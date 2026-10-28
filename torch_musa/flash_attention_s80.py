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
        - Uses tile size 8x32 for Q/K/V
        - For FP32 with D<=128: full precision (float storage)
        - For FP32 with D>128 or FP16: mixed precision (half storage for Q/K/V)
        - Memory efficient: O(N*H*L*D) instead of O(N*H*L*S)
    """
    # Ensure contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    
    # Call the registered C++ operator
    return torch.ops.aten.flash_attention_s80(query, key, value, causal, scale)


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

    # Apply causal mask (top-left: position i can attend to j <= i)
    if causal:
        causal_mask = torch.triu(
            torch.ones(L, S, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Weighted sum: [N, H, L, S] @ [N, H, S, D] -> [N, H, L, D]
    output = torch.matmul(attn_weights, value)

    return output
