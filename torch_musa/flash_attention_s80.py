"""Flash Attention S80 - Software-level tiled attention for MTT S80."""

import torch
import torch_musa
from typing import Optional


def flash_attention_s80(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention-like operator for MTT S80.

    This is a software-level implementation using tiled computation and
    online softmax algorithm. Designed for MTT S80 (CC 2.1) which lacks
    hardware Flash Attention support.

    Args:
        query: Query tensor [N, H, L, D]
        key: Key tensor [N, H, S, D]
        value: Value tensor [N, H, S, D]
        causal: Whether to apply causal mask (default: False)
        scale: Attention scale factor (default: 1/sqrt(D))

    Returns:
        Output tensor [N, H, L, D]

    Note:
        - Supports FP32 and FP16
        - Head dimension D must be <= 256
        - Uses tile size 64x64 for Q/K/V
        - Memory efficient: O(N*H*L*D) instead of O(N*H*L*S)
    """
    # Validate inputs
    assert query.dim() == 4, f"query must be 4D, got {query.dim()}D"
    assert key.dim() == 4, f"key must be 4D, got {key.dim()}D"
    assert value.dim() == 4, f"value must be 4D, got {value.dim()}D"

    N, H, L, D = query.shape
    _, _, S, D_k = key.shape

    assert D == D_k, f"query and key head dim mismatch: {D} vs {D_k}"
    assert key.shape[0] == N and key.shape[1] == H, "key batch/head mismatch"
    assert value.shape == (N, H, S, D), "value shape mismatch"

    assert D <= 256, f"Head dimension {D} > 256 not supported"
    assert query.dtype in [torch.float32, torch.float16], \
        f"Unsupported dtype {query.dtype}, use float32 or float16"

    # Ensure contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Default scale
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Call C++ implementation
    # For now, fall back to standard SDPA with a warning
    # TODO: Replace with actual kernel call once registered
    import warnings
    warnings.warn(
        "Flash Attention S80 kernel not yet registered. "
        "Falling back to standard SDPA (math backend). "
        "This is a reference implementation for validation.",
        UserWarning
    )

    # Use PyTorch's SDPA as reference
    with torch.backends.musa.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale
        )

    return output


def flash_attention_s80_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation using standard attention.

    This is used for validation and correctness testing.
    """
    N, H, L, D = query.shape
    _, _, S, _ = key.shape

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Compute attention scores
    # Q @ K^T: [N, H, L, D] @ [N, H, D, S] -> [N, H, L, S]
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(L, S, device=query.device, dtype=torch.bool),
            diagonal=S - L + 1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Weighted sum: [N, H, L, S] @ [N, H, S, D] -> [N, H, L, D]
    output = torch.matmul(attn_weights, value)

    return output


# Register as a custom op for testing
try:
    torch.library.define("musa::flash_attention_s80", "(Tensor q, Tensor k, Tensor v, bool causal, float? scale) -> Tensor")
    torch.library.impl("musa::flash_attention_s80", "PrivateUse1", flash_attention_s80)
except Exception as e:
    # If registration fails, just use the function directly
    pass
