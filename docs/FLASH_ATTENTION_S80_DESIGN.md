# Flash Attention-like Operator for MTT S80 Design Proposal

## Executive Summary

Based on analysis of ggml/llama.cpp Flash Attention implementation and MUSA SDK capabilities, this document proposes a software-level Flash Attention-like operator for MTT S80 (CC 2.1) that achieves memory efficiency through tiling and online softmax, despite lacking hardware Flash Attention support.

## Key Findings

### 1. ggml MUSA Flash Attention Architecture

**Kernel Selection Logic** (from `fattn.cu:704-718`):
- MUSA falls back to `BEST_FATTN_KERNEL_TILE` or `BEST_FATTN_KERNEL_VEC`
- MMA_F16 kernel (hardware tensor core) is explicitly disabled for MUSA
- TILE kernel is the generic fallback when no tensor cores are available

**TILE Kernel Characteristics** (from `fattn-tile.cuh:793-992`):
- Uses shared memory tiling to reduce HBM access
- Implements online softmax algorithm (numerically stable, single-pass)
- Templated for different head dimensions (64, 128, 256, etc.)
- Configuration parameters: `nbatch_fa`, `nbatch_K`, `nwarps`, `ncols`

### 2. S80 Hardware Constraints

- **Architecture**: CC 2.1 (no native Flash Attention hardware)
- **Shared Memory**: Limited compared to newer architectures
- **Tensor Cores**: No Turing/Ampere-style MMA instructions
- **BF16**: Not supported for compute (only storage)
- **Memory Bandwidth**: ~400 GB/s (theoretical)

### 3. Available MUSA Primitives

**muDNN Operations**:
- `MatMul`: High-performance matrix multiplication
- `Softmax`: Optimized softmax with various modes
- `Reduce`: Sum/max reductions
- `Unary`: Element-wise operations (EXP, etc.)

**muBLASLt**:
- Epilogue fusion (RELU, GELU, BIAS)
- Batched operations
- Mixed precision support (FP32/FP16 compute)

**MUSA Runtime**:
- Shared memory management
- Warp-level primitives
- Cooperative groups

## Proposed Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Flash Attention S80                    │
├─────────────────────────────────────────────────────────┤
│  Input: Q [N,H,L,D], K [N,H,S,D], V [N,H,S,D]          │
│  Output: O [N,H,L,D]                                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐                                         │
│  │ Tile Loop   │  For each tile of K/V (block of S)     │
│  │ (over S)    │                                         │
│  └──────┬──────┘                                         │
│         │                                                │
│         ├─► Load K_tile, V_tile to shared memory        │
│         │                                                │
│         ├─► Compute S_tile = Q @ K_tile^T / sqrt(d)    │
│         │    (using muBLASLt MatMul)                    │
│         │                                                │
│         ├─► Apply mask (if causal)                      │
│         │                                                │
│         ├─► Online Softmax Update:                      │
│         │    m_new = max(m_old, max(S_tile))            │
│         │    P_tile = exp(S_tile - m_new)               │
│         │    l_new = exp(m_old - m_new) * l_old + ΣP   │
│         │    O_new = exp(m_old - m_new) * O_old + P@V  │
│         │                                                │
│         └─► Update accumulators (m, l, O)               │
│                                                           │
│  Final: O = O / l                                        │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Tiling Strategy

**Tile Sizes** (optimized for S80 shared memory ~48KB):
- `TILE_S = 64` (KV sequence length per tile)
- `TILE_L = 64` (Q sequence length per block)
- `D = 128` (head dimension, common for LLMs)

**Shared Memory Layout**:
```
Q_tile:     [TILE_L, D] = 64 × 128 × 2B = 16 KB (FP16)
K_tile:     [TILE_S, D] = 64 × 128 × 2B = 16 KB (FP16)
V_tile:     [TILE_S, D] = 64 × 128 × 2B = 16 KB (FP16)
S_tile:     [TILE_L, TILE_S] = 64 × 64 × 4B = 16 KB (FP32)
─────────────────────────────────────────────────────────
Total: 64 KB (fits in S80 shared memory with margin)
```

#### 2. Online Softmax Algorithm

**Numerical Stability**:
```cpp
// Per-tile update (Flash Attention paper, Algorithm 1)
m_new = max(m_old, max_j(S_tile[i,j]))
P_tile = exp(S_tile - m_new)
l_new = exp(m_old - m_new) * l_old + sum_j(P_tile[i,j])
O_new = exp(m_old - m_new) * O_old + P_tile @ V_tile
```

**Benefits**:
- Single-pass over K/V (no need to store full S matrix)
- Numerically stable (no overflow in exp)
- Memory efficient (O(L×D) instead of O(L×S))

#### 3. Compute Strategy

**Option A: muBLASLt MatMul + Custom Softmax**
```cpp
// Use muBLASLt for Q @ K^T and P @ V
mublasLtMatmul(handle, Q_tile, K_tile, S_tile, ...);
// Custom kernel for online softmax update
online_softmax_kernel<<<...>>>(S_tile, m, l, O);
mublasLtMatmul(handle, P_tile, V_tile, O_tile, ...);
```

**Option B: Fully Custom Kernel**
```cpp
// Single fused kernel with warp-level matmul
__global__ void flash_attention_s80_kernel(...) {
    // Load tiles to shared memory
    // Warp-level matmul using wmma-like pattern
    // Online softmax update
    // Accumulate output
}
```

**Recommendation**: Start with Option A (hybrid approach) for faster development and validation, then optimize with Option B if needed.

#### 4. Memory Access Pattern

**Coalesced Access**:
- Q/K/V stored in row-major format
- Each warp loads contiguous 128-byte chunks
- Use `__ldg()` for read-only cache

**Shared Memory Banking**:
- Pad tiles to avoid bank conflicts
- Use 128-bit aligned loads/stores

### Implementation Plan

#### Phase 1: Core Kernel (2-3 weeks)

1. **Basic TILE kernel** for FP16:
   - Single-head, single-batch
   - No mask support
   - Fixed tile sizes (64×64)

2. **Validation**:
   - Compare against standard attention (math backend)
   - Test numerical accuracy (rtol=1e-3, atol=1e-3)
   - Benchmark on typical LLM shapes

#### Phase 2: Feature Completion (1-2 weeks)

3. **Causal mask support**:
   - Upper triangular mask for autoregressive models
   - Optimize for decode (L=1) vs prefill (L>1)

4. **Multi-head/batch support**:
   - Grid-stride loop over heads
   - Batch dimension handling

5. **FP32 fallback**:
   - For architectures without fast FP16

#### Phase 3: Optimization (2-3 weeks)

6. **Warp-level matmul**:
   - Custom MMA-like instructions using warp shuffle
   - Reduce shared memory traffic

7. **Tuning**:
   - Tile size selection based on head dimension
   - Occupancy optimization
   - Memory bandwidth utilization

8. **Advanced features**:
   - Variable-length sequences (padding optimization)
   - Sliding window attention
   - GQA (Grouped Query Attention) optimization

### Expected Performance

**Memory Savings**:
- Standard attention: O(N×H×L×S) for S matrix
- Flash Attention S80: O(N×H×L×D) for output only
- **Reduction**: ~10-100x for typical LLM shapes (L,S=2048-8192, D=128)

**Compute Efficiency**:
- Theoretical peak: ~10 TFLOPS (FP16) on S80
- Expected utilization: 40-60% (limited by memory bandwidth)
- **Speedup vs math backend**: 2-3x for long sequences (L,S > 2048)

**Benchmark Targets** (Llama-2 7B, batch=1, L=S=2048, H=32, D=128):
- Memory: ~1 GB (vs ~8 GB for standard attention)
- Latency: ~10-15 ms (vs ~30-40 ms for math backend)
- Throughput: ~150 TFLOPS/s (effective)

### Integration with torch_musa

#### API Design

```python
# Python API
def flash_attention_s80(
    query: torch.Tensor,  # [N, H, L, D]
    key: torch.Tensor,    # [N, H, S, D]
    value: torch.Tensor,  # [N, H, S, D]
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:  # [N, H, L, D]
    """
    Flash Attention-like operator for MTT S80.
    
    Falls back to math backend if not supported or disabled.
    """
    return torch.ops.aten.musa_flash_attention_s80(
        query, key, value, causal, scale
    )
```

#### C++ Dispatch

```cpp
// Register in musa_functions.yaml
- func: musa_flash_attention_s80
  dispatch:
    PrivateUse1: FlashAttentionS80

// Implementation in FlashAttention.cpp
at::Tensor FlashAttentionS80(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool causal,
    c10::optional<double> scale) {
  
  // Check architecture
  if (at::musa::getMUSAArch() < 210) {
    TORCH_WARN("Flash Attention S80 requires CC >= 2.1");
    return at::scaled_dot_product_attention(query, key, value, ...);
  }
  
  // Launch kernel
  return flash_attention_s80_kernel(query, key, value, causal, scale);
}
```

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Shared memory overflow | Kernel fails to launch | Reduce tile sizes, use occupancy calculator |
| Numerical instability | Incorrect results | Use FP32 accumulators, validate against reference |
| Low occupancy | Poor performance | Tune block size, reduce register usage |
| Memory bandwidth bottleneck | Limited speedup | Optimize coalescing, use L2 cache hints |
| muBLASLt overhead | Launch latency | Fuse operations in custom kernel (Phase 3) |

### Success Criteria

1. **Correctness**: Passes all validation tests (rtol=1e-3, atol=1e-3)
2. **Memory**: Uses < 2 GB for Llama-2 7B attention (batch=1, L=S=4096)
3. **Performance**: 2x speedup vs math backend for L,S >= 2048
4. **Integration**: Seamlessly integrates with torch_musa SDPA backend selection

## References

1. **Flash Attention Paper**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
2. **ggml Implementation**: https://github.com/ggml-org/ggml/blob/master/src/ggml-cuda/fattn-tile.cuh
3. **MUSA SDK Documentation**: MooreThreads MUSA Toolkit 4.3.0
4. **muDNN API Reference**: muDNN 3.1.0 Programmer's Guide

## Appendix: Configuration Parameters

### Recommended Tile Sizes for S80

| Head Dim (D) | TILE_L | TILE_S | Shared Mem (FP16) | Occupancy |
|--------------|--------|--------|-------------------|-----------|
| 64           | 128    | 128    | 48 KB             | 2 blocks/SM |
| 128          | 64     | 64     | 48 KB             | 2 blocks/SM |
| 256          | 32     | 32     | 48 KB             | 1 block/SM |

### Warp Configuration

- **Block size**: 256 threads (8 warps)
- **Warps per tile**: 8 warps × 32 threads = 256 threads
- **Threads per row**: 32 (one warp)
- **Rows per tile**: 8 (TILE_L / threads_per_row)

---

**Document Version**: 1.0  
**Date**: 2026-10-28  
**Author**: torch_musa team  
**Status**: Proposal (pending implementation)
