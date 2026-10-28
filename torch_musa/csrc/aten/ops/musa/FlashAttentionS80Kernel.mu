#include <ATen/core/Scalar.h>
#include <ATen/ops/empty.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

// Flash Attention S80: Software-level tiled attention with online softmax
// Designed for MTT S80 (CC 2.1) without hardware Flash Attention support

#define TILE_L 64
#define TILE_S 64
#define MAX_THREADS 256
#define WARP_SIZE 32

// Online softmax state
struct OnlineSoftmaxState {
  float m;  // max value
  float l;  // sum of exp
};

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Block-level reduction for max
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[MAX_THREADS / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_max(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < (MAX_THREADS / WARP_SIZE)) ? shared[lane] : -FLT_MAX;
    val = warp_reduce_max(val);
  }

  return val;
}

// Block-level reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[MAX_THREADS / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_sum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < (MAX_THREADS / WARP_SIZE)) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
  }

  return val;
}

// Compute Q @ K^T for a tile using simple matmul
__device__ __forceinline__ void compute_qk_tile(
    const float* Q_tile,  // [TILE_L, D]
    const float* K_tile,  // [TILE_S, D]
    float* S_tile,        // [TILE_L, TILE_S]
    int D,
    float scale) {
  int row = threadIdx.y;
  int col = threadIdx.x;

  if (row < TILE_L && col < TILE_S) {
    float sum = 0.0f;
    for (int k = 0; k < D; ++k) {
      sum += Q_tile[row * D + k] * K_tile[col * D + k];
    }
    S_tile[row * TILE_S + col] = sum * scale;
  }
}

// Apply causal mask
__device__ __forceinline__ void apply_causal_mask(
    float* S_tile,
    int q_idx,
    int k_idx,
    int TILE_S_size) {
  int row = threadIdx.y;
  int col = threadIdx.x;

  if (row < TILE_L && col < TILE_S_size) {
    int global_q = q_idx + row;
    int global_k = k_idx + col;
    if (global_k > global_q) {
      S_tile[row * TILE_S + col] = -FLT_MAX;
    }
  }
}

// Online softmax update
__device__ __forceinline__ void online_softmax_update(
    const float* S_tile,
    float* O_tile,        // [TILE_L, D]
    const float* V_tile,  // [TILE_S, D]
    float* m,             // max values [TILE_L]
    float* l,             // sum values [TILE_L]
    int D,
    int TILE_S_size) {
  int row = threadIdx.y;

  if (row < TILE_L) {
    // Find new max for this row
    float m_new = m[row];
    for (int col = threadIdx.x; col < TILE_S_size; col += blockDim.x) {
      m_new = fmaxf(m_new, S_tile[row * TILE_S + col]);
    }

    // Reduce max across threads
    __shared__ float m_shared[MAX_THREADS];
    m_shared[threadIdx.x + threadIdx.y * blockDim.x] = m_new;
    __syncthreads();

    if (threadIdx.x == 0) {
      m_new = -FLT_MAX;
      for (int t = 0; t < blockDim.x; ++t) {
        m_new = fmaxf(m_new, m_shared[t + row * blockDim.x]);
      }
      m_shared[row] = m_new;
    }
    __syncthreads();
    m_new = m_shared[row];

    // Compute exp(S - m_new) and update l
    float l_new = 0.0f;
    float correction = expf(m[row] - m_new);

    for (int col = threadIdx.x; col < TILE_S_size; col += blockDim.x) {
      float p = expf(S_tile[row * TILE_S + col] - m_new);
      l_new += p;
      S_tile[row * TILE_S + col] = p;  // Store P = exp(S - m_new)
    }

    // Reduce sum across threads
    __shared__ float l_shared[MAX_THREADS];
    l_shared[threadIdx.x + threadIdx.y * blockDim.x] = l_new;
    __syncthreads();

    if (threadIdx.x == 0) {
      l_new = 0.0f;
      for (int t = 0; t < blockDim.x; ++t) {
        l_new += l_shared[t + row * blockDim.x];
      }
      l_shared[row] = l_new;
    }
    __syncthreads();
    l_new = l_shared[row];

    // Update l: l_new = exp(m_old - m_new) * l_old + sum(P)
    l[row] = correction * l[row] + l_new;

    // Update O: O_new = correction * O_old + P @ V
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float pv = 0.0f;
      for (int k = 0; k < TILE_S_size; ++k) {
        pv += S_tile[row * TILE_S + k] * V_tile[k * D + d];
      }
      O_tile[row * D + d] = correction * O_tile[row * D + d] + pv;
    }

    // Update m
    m[row] = m_new;
  }
}

// Main Flash Attention kernel for S80
template <typename T>
__global__ void flash_attention_s80_kernel(
    const T* Q,      // [N, H, L, D]
    const T* K,      // [N, H, S, D]
    const T* V,      // [N, H, S, D]
    T* O,            // [N, H, L, D]
    const float scale,
    const int N,
    const int H,
    const int L,
    const int S,
    const int D,
    const bool causal) {
  // Each block processes one (batch, head, tile_l) combination
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int tile_l_idx = blockIdx.x;

  int q_start = tile_l_idx * TILE_L;
  if (q_start >= L) return;

  int q_end = min(q_start + TILE_L, L);
  int actual_tile_l = q_end - q_start;

  // Shared memory for tiles
  extern __shared__ char smem[];
  float* Q_tile = (float*)smem;
  float* K_tile = Q_tile + TILE_L * D;
  float* V_tile = K_tile + TILE_S * D;
  float* S_tile = V_tile + TILE_S * D;
  float* O_tile = S_tile + TILE_L * TILE_S;
  float* m = O_tile + TILE_L * D;
  float* l = m + TILE_L;

  // Initialize output and state
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * blockDim.y;

  for (int i = tid; i < actual_tile_l * D; i += total_threads) {
    O_tile[i] = 0.0f;
  }
  for (int i = tid; i < actual_tile_l; i += total_threads) {
    m[i] = -FLT_MAX;
    l[i] = 0.0f;
  }
  __syncthreads();

  // Load Q tile
  const T* Q_ptr = Q + ((batch_idx * H + head_idx) * L + q_start) * D;
  for (int i = tid; i < actual_tile_l * D; i += total_threads) {
    Q_tile[i] = (float)Q_ptr[i];
  }
  __syncthreads();

  // Loop over K/V tiles
  int num_s_tiles = (S + TILE_S - 1) / TILE_S;
  for (int s_tile_idx = 0; s_tile_idx < num_s_tiles; ++s_tile_idx) {
    int s_start = s_tile_idx * TILE_S;
    int s_end = min(s_start + TILE_S, S);
    int actual_tile_s = s_end - s_start;

    // Load K tile
    const T* K_ptr = K + ((batch_idx * H + head_idx) * S + s_start) * D;
    for (int i = tid; i < actual_tile_s * D; i += total_threads) {
      K_tile[i] = (float)K_ptr[i];
    }
    __syncthreads();

    // Compute S = Q @ K^T
    compute_qk_tile(Q_tile, K_tile, S_tile, D, scale);
    __syncthreads();

    // Apply causal mask if needed
    if (causal) {
      apply_causal_mask(S_tile, q_start, s_start, actual_tile_s);
      __syncthreads();
    }

    // Load V tile
    const T* V_ptr = V + ((batch_idx * H + head_idx) * S + s_start) * D;
    for (int i = tid; i < actual_tile_s * D; i += total_threads) {
      V_tile[i] = (float)V_ptr[i];
    }
    __syncthreads();

    // Online softmax update
    online_softmax_update(S_tile, O_tile, V_tile, m, l, D, actual_tile_s);
    __syncthreads();
  }

  // Normalize output: O = O / l
  for (int i = tid; i < actual_tile_l * D; i += total_threads) {
    int row = i / D;
    O_tile[i] = O_tile[i] / l[row];
  }
  __syncthreads();

  // Write output
  T* O_ptr = O + ((batch_idx * H + head_idx) * L + q_start) * D;
  for (int i = tid; i < actual_tile_l * D; i += total_threads) {
    O_ptr[i] = (T)O_tile[i];
  }
}

// Host function to launch Flash Attention S80 kernel
at::Tensor flash_attention_s80_forward(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    bool causal,
    c10::optional<double> scale_opt) {
  // Validate inputs
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [N, H, L, D]");
  TORCH_CHECK(K.dim() == 4, "K must be 4D [N, H, S, D]");
  TORCH_CHECK(V.dim() == 4, "V must be 4D [N, H, S, D]");

  int N = Q.size(0);
  int H = Q.size(1);
  int L = Q.size(2);
  int D = Q.size(3);
  int S = K.size(2);

  TORCH_CHECK(K.size(0) == N && K.size(1) == H && K.size(3) == D,
              "K shape mismatch");
  TORCH_CHECK(V.size(0) == N && V.size(1) == H && V.size(2) == S && V.size(3) == D,
              "V shape mismatch");

  TORCH_CHECK(D <= 256, "Head dimension must be <= 256");
  TORCH_CHECK(D == K.size(3), "Q and K must have same head dimension");

  float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(static_cast<float>(D));

  // Allocate output
  auto O = at::empty({N, H, L, D}, Q.options());

  // Get current device and stream
  int cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&cur_device));
  const musaStream_t stream = c10::musa::getCurrentMUSAStream(cur_device);

  // Calculate shared memory size
  size_t smem_size = (TILE_L * D + TILE_S * D + TILE_S * D +
                      TILE_L * TILE_S + TILE_L * D + TILE_L + TILE_L) * sizeof(float);

  // Launch kernel
  dim3 grid((L + TILE_L - 1) / TILE_L, H, N);
  dim3 block(32, 8);  // 256 threads total

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "flash_attention_s80_kernel", [&] {
        flash_attention_s80_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            scale,
            N, H, L, S, D,
            causal);
        C10_MUSA_KERNEL_LAUNCH_CHECK();
      });

  return O;
}

}  // namespace musa
}  // namespace at
