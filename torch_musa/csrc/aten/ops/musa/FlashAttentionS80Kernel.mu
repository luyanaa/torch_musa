#include <ATen/core/Scalar.h>
#include <ATen/ops/empty.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cmath>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

#define FA_BR 8
#define FA_BC 32
#define FA_WARP_SIZE 32
#define FA_NUM_WARPS 8
#define FA_THREADS (FA_NUM_WARPS * FA_WARP_SIZE)

template <typename QKV_STYPE>
__device__ __forceinline__ int k_pad() {
  return (sizeof(QKV_STYPE) == 2) ? 2 : 1;
}

template <typename QKV_STYPE>
__device__ __forceinline__ int k_stride(int D) {
  return D + k_pad<QKV_STYPE>();
}

__device__ __forceinline__ int fa_warp_id() {
  return threadIdx.x / FA_WARP_SIZE;
}

__device__ __forceinline__ int fa_lane_id() {
  return threadIdx.x % FA_WARP_SIZE;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = FA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = FA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename QKV_STYPE>
struct SharedMemLayout {
  QKV_STYPE* Q_smem;
  QKV_STYPE* K_smem;
  QKV_STYPE* V_smem;
  float* S_smem;
  float* O_smem;
  float* m_smem;
  float* l_smem;

  static size_t total_size(int D) {
    size_t q = FA_BR * D * sizeof(QKV_STYPE);
    size_t k = FA_BC * (D + k_pad<QKV_STYPE>()) * sizeof(QKV_STYPE);
    size_t v = FA_BC * D * sizeof(QKV_STYPE);
    size_t s = FA_BR * FA_BC * sizeof(float);
    size_t o = FA_BR * D * sizeof(float);
    size_t ml = FA_BR * 2 * sizeof(float);
    return q + k + v + s + o + ml;
  }

  static SharedMemLayout create(char* smem, int D) {
    SharedMemLayout layout;
    char* ptr = smem;
    layout.Q_smem = reinterpret_cast<QKV_STYPE*>(ptr);
    ptr += FA_BR * D * sizeof(QKV_STYPE);
    layout.K_smem = reinterpret_cast<QKV_STYPE*>(ptr);
    ptr += FA_BC * (D + k_pad<QKV_STYPE>()) * sizeof(QKV_STYPE);
    layout.V_smem = reinterpret_cast<QKV_STYPE*>(ptr);
    ptr += FA_BC * D * sizeof(QKV_STYPE);
    layout.S_smem = reinterpret_cast<float*>(ptr);
    ptr += FA_BR * FA_BC * sizeof(float);
    layout.O_smem = reinterpret_cast<float*>(ptr);
    ptr += FA_BR * D * sizeof(float);
    layout.m_smem = reinterpret_cast<float*>(ptr);
    ptr += FA_BR * sizeof(float);
    layout.l_smem = reinterpret_cast<float*>(ptr);
    return layout;
  }
};

__device__ __forceinline__ float load_as_float(half val) {
  return __half2float(val);
}

__device__ __forceinline__ float load_as_float(float val) {
  return val;
}

template <typename QKV_STYPE>
__device__ __forceinline__ QKV_STYPE store_qkv(float val);

template <>
__device__ __forceinline__ half store_qkv<half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ float store_qkv<float>(float val) {
  return val;
}

template <typename T, typename QKV_STYPE>
__global__ void flash_attention_s80_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const float scale,
    const int N,
    const int H,
    const int L,
    const int S,
    const int D,
    const bool causal) {
  const int warp_id = fa_warp_id();
  const int lane = fa_lane_id();

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int tile_l_idx = blockIdx.x;

  const int q_start = tile_l_idx * FA_BR;
  if (q_start >= L) return;

  const int row = warp_id;
  const int global_row = q_start + row;

  extern __shared__ char smem[];
  auto sm = SharedMemLayout<QKV_STYPE>::create(smem, D);

  for (int i = threadIdx.x; i < FA_BR; i += FA_THREADS) {
    sm.m_smem[i] = -INFINITY;
    sm.l_smem[i] = 0.0f;
  }
  for (int i = threadIdx.x; i < FA_BR * D; i += FA_THREADS) {
    sm.O_smem[i] = 0.0f;
  }
  __syncthreads();

  const auto batch_head_q_offset =
      (static_cast<int64_t>(batch_idx) * H + head_idx) * L * D;
  const auto batch_head_kv_offset =
      (static_cast<int64_t>(batch_idx) * H + head_idx) * S * D;

  const int actual_br = min(FA_BR, L - q_start);

  for (int i = threadIdx.x; i < actual_br * D; i += FA_THREADS) {
    int r = i / D;
    int d = i % D;
    int global_r = q_start + r;
    float val = load_as_float(Q[batch_head_q_offset + global_r * D + d]);
    sm.Q_smem[r * D + d] = store_qkv<QKV_STYPE>(val);
  }
  __syncthreads();

  const int num_s_tiles = (S + FA_BC - 1) / FA_BC;

  for (int s_tile_idx = 0; s_tile_idx < num_s_tiles; ++s_tile_idx) {
    const int s_start = s_tile_idx * FA_BC;
    const int actual_bc = min(FA_BC, S - s_start);

    bool skip_tile = causal && (s_start > (q_start + actual_br - 1));

    if (!skip_tile) {
      const int K_STRIDE = k_stride<QKV_STYPE>(D);
      for (int i = threadIdx.x; i < actual_bc * D; i += FA_THREADS) {
        int r = i / D;
        int d = i % D;
        int global_r = s_start + r;
        float kval = load_as_float(K[batch_head_kv_offset + global_r * D + d]);
        float vval = load_as_float(V[batch_head_kv_offset + global_r * D + d]);
        sm.K_smem[r * K_STRIDE + d] = store_qkv<QKV_STYPE>(kval);
        sm.V_smem[r * D + d] = store_qkv<QKV_STYPE>(vval);
      }
    }
    __syncthreads();

    if (!skip_tile && global_row < L) {
      bool row_skip = causal && (s_start > global_row);

      if (!row_skip) {
        const int K_STRIDE = k_stride<QKV_STYPE>(D);
        for (int j = lane; j < actual_bc; j += FA_WARP_SIZE) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            dot += load_as_float(sm.Q_smem[row * D + d]) *
                   load_as_float(sm.K_smem[j * K_STRIDE + d]);
          }
          float s_val = dot * scale;

          if (causal) {
            int global_k = s_start + j;
            if (global_k > global_row) {
              s_val = -INFINITY;
            }
          }

          sm.S_smem[row * FA_BC + j] = s_val;
        }
        
        __shfl_xor_sync(0xFFFFFFFF, 0, 1);

        float row_max = -INFINITY;
        for (int j = lane; j < actual_bc; j += FA_WARP_SIZE) {
          row_max = fmaxf(row_max, sm.S_smem[row * FA_BC + j]);
        }
        row_max = warp_reduce_max(row_max);

        float m_old = sm.m_smem[row];
        float m_new = fmaxf(m_old, row_max);

        float correction = expf(m_old - m_new);

        float local_sum = 0.0f;
        for (int j = lane; j < actual_bc; j += FA_WARP_SIZE) {
          float p = expf(sm.S_smem[row * FA_BC + j] - m_new);
          sm.S_smem[row * FA_BC + j] = p;
          local_sum += p;
        }
        local_sum = warp_reduce_sum(local_sum);

        if (lane == 0) {
          sm.l_smem[row] = correction * sm.l_smem[row] + local_sum;
          sm.m_smem[row] = m_new;
        }

        for (int d_base = 0; d_base < D; d_base += FA_WARP_SIZE) {
          int d = d_base + lane;
          if (d < D) {
            float pv = 0.0f;
            for (int j = 0; j < actual_bc; ++j) {
              pv += sm.S_smem[row * FA_BC + j] *
                    load_as_float(sm.V_smem[j * D + d]);
            }
            sm.O_smem[row * D + d] =
                correction * sm.O_smem[row * D + d] + pv;
          }
        }
      }
    }
    __syncthreads();
  }

  if (global_row < L) {
    float l_val = sm.l_smem[row];
    if (l_val > 0.0f) {
      for (int d_base = 0; d_base < D; d_base += FA_WARP_SIZE) {
        int d = d_base + lane;
        if (d < D) {
          float out_val = sm.O_smem[row * D + d] / l_val;
          int64_t out_idx = batch_head_q_offset + global_row * D + d;
          O[out_idx] = static_cast<T>(out_val);
        }
      }
    } else {
      for (int d_base = 0; d_base < D; d_base += FA_WARP_SIZE) {
        int d = d_base + lane;
        if (d < D) {
          int64_t out_idx = batch_head_q_offset + global_row * D + d;
          O[out_idx] = static_cast<T>(0.0f);
        }
      }
    }
  }
}

at::Tensor flash_attention_s80_forward(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    bool causal,
    c10::optional<double> scale_opt) {
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [N, H, L, D]");
  TORCH_CHECK(K.dim() == 4, "K must be 4D [N, H, S, D]");
  TORCH_CHECK(V.dim() == 4, "V must be 4D [N, H, S, D]");

  TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
              "Q, K, V must be on the same device");

  int N = Q.size(0);
  int H = Q.size(1);
  int L = Q.size(2);
  int D = Q.size(3);
  int S = K.size(2);

  TORCH_CHECK(K.size(0) == N && K.size(1) == H && K.size(3) == D,
              "K shape mismatch");
  TORCH_CHECK(V.size(0) == N && V.size(1) == H && V.size(2) == S &&
                  V.size(3) == D,
              "V shape mismatch");
  TORCH_CHECK(D <= 256, "Head dimension must be <= 256");
  TORCH_CHECK(D > 0, "Head dimension must be > 0");

  float scale = scale_opt.has_value()
                    ? static_cast<float>(scale_opt.value())
                    : 1.0f / sqrtf(static_cast<float>(D));

  auto O = at::empty({N, H, L, D}, Q.options());

  if (L == 0 || S == 0) {
    O.zero_();
    return O;
  }

  int cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&cur_device));
  TORCH_CHECK(Q.get_device() == cur_device,
              "Q must be on current MUSA device");

  const musaStream_t stream = c10::musa::getCurrentMUSAStream(cur_device);

  bool use_float_storage = (Q.scalar_type() == at::ScalarType::Float && D <= 128);

  dim3 grid((L + FA_BR - 1) / FA_BR, H, N);
  dim3 block(FA_THREADS);

  if (use_float_storage) {
    size_t smem_size = SharedMemLayout<float>::total_size(D);
    TORCH_CHECK(smem_size <= 48 * 1024,
                "Shared memory requirement (", smem_size,
                " bytes) exceeds S80 limit (49152 bytes)");

    AT_DISPATCH_FLOATING_TYPES(
        Q.scalar_type(), "flash_attention_s80_kernel", [&] {
          flash_attention_s80_kernel<scalar_t, float>
              <<<grid, block, smem_size, stream>>>(
                  Q.data_ptr<scalar_t>(),
                  K.data_ptr<scalar_t>(),
                  V.data_ptr<scalar_t>(),
                  O.data_ptr<scalar_t>(),
                  scale,
                  N, H, L, S, D,
                  causal);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        });
  } else {
    size_t smem_size = SharedMemLayout<half>::total_size(D);
    TORCH_CHECK(smem_size <= 48 * 1024,
                "Shared memory requirement (", smem_size,
                " bytes) exceeds S80 limit (49152 bytes)");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(), "flash_attention_s80_kernel", [&] {
          flash_attention_s80_kernel<scalar_t, half>
              <<<grid, block, smem_size, stream>>>(
                  Q.data_ptr<scalar_t>(),
                  K.data_ptr<scalar_t>(),
                  V.data_ptr<scalar_t>(),
                  O.data_ptr<scalar_t>(),
                  scale,
                  N, H, L, S, D,
                  causal);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        });
  }

  return O;
}

}  // namespace musa
}  // namespace at
