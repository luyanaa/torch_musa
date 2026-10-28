"""Test flash_attention_s80 operator."""

import torch
import torch_musa
import pytest
from torch_musa.flash_attention_s80 import flash_attention_s80_reference


def test_flash_attention_s80_basic():
    """Test basic functionality with small inputs."""
    print("Testing flash_attention_s80 basic...")
    
    N, H, L, S, D = 1, 1, 8, 8, 64
    query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
    key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    
    # Call the C++ operator
    output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
    
    # Compare with reference
    ref_output = flash_attention_s80_reference(query, key, value, False, None)
    
    torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
    print("  Basic FP32: PASS")


def test_flash_attention_s80_dtypes():
    """Test different data types."""
    print("\nTesting flash_attention_s80 dtypes...")
    
    N, H, L, S, D = 2, 4, 16, 16, 64
    
    for dtype in [torch.float32, torch.float16]:
        query = torch.randn(N, H, L, D, device="musa", dtype=dtype)
        key = torch.randn(N, H, S, D, device="musa", dtype=dtype)
        value = torch.randn(N, H, S, D, device="musa", dtype=dtype)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
        ref_output = flash_attention_s80_reference(query, key, value, False, None)
        
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        atol = 1e-3 if dtype == torch.float32 else 1e-2
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)
        print(f"  {dtype}: PASS")


def test_flash_attention_s80_head_dims():
    """Test different head dimensions (D)."""
    print("\nTesting flash_attention_s80 head dimensions...")
    
    N, H, L, S = 1, 2, 16, 16
    
    for D in [32, 64, 128, 256]:
        query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
        key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
        ref_output = flash_attention_s80_reference(query, key, value, False, None)
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
        print(f"  D={D}: PASS")


def test_flash_attention_s80_sequence_lengths():
    """Test different sequence lengths (L, S) including edge cases."""
    print("\nTesting flash_attention_s80 sequence lengths...")
    
    N, H, D = 1, 2, 64
    
    test_cases = [
        (8, 8),      # Small, equal
        (16, 32),    # L < S
        (32, 16),    # L > S
        (7, 13),     # Non-power-of-2
        (33, 65),    # Non-multiple of tile size (32)
        (1, 1),      # Minimum
        (64, 64),    # Tile size
    ]
    
    for L, S in test_cases:
        query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
        key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
        ref_output = flash_attention_s80_reference(query, key, value, False, None)
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
        print(f"  L={L}, S={S}: PASS")


def test_flash_attention_s80_causal():
    """Test causal masking."""
    print("\nTesting flash_attention_s80 causal...")
    
    N, H, L, S, D = 2, 4, 16, 16, 64
    
    for dtype in [torch.float32, torch.float16]:
        query = torch.randn(N, H, L, D, device="musa", dtype=dtype)
        key = torch.randn(N, H, S, D, device="musa", dtype=dtype)
        value = torch.randn(N, H, S, D, device="musa", dtype=dtype)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, True, None)
        ref_output = flash_attention_s80_reference(query, key, value, True, None)
        
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        atol = 1e-3 if dtype == torch.float32 else 1e-2
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)
        print(f"  Causal {dtype}: PASS")


def test_flash_attention_s80_causal_edge_cases():
    """Test causal masking with edge cases."""
    print("\nTesting flash_attention_s80 causal edge cases...")
    
    N, H, D = 1, 1, 64
    
    test_cases = [
        (8, 16),   # L < S
        (16, 8),   # L > S
        (7, 13),   # Non-power-of-2
        (33, 33),  # Non-multiple of tile size
    ]
    
    for L, S in test_cases:
        query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
        key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, True, None)
        ref_output = flash_attention_s80_reference(query, key, value, True, None)
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
        print(f"  Causal L={L}, S={S}: PASS")


def test_flash_attention_s80_custom_scale():
    """Test custom scale parameter."""
    print("\nTesting flash_attention_s80 custom scale...")
    
    N, H, L, S, D = 1, 2, 16, 16, 64
    query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
    key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    
    custom_scale = 0.125
    
    output = torch.ops.aten.flash_attention_s80(query, key, value, False, custom_scale)
    ref_output = flash_attention_s80_reference(query, key, value, False, custom_scale)
    
    torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
    print("  Custom scale: PASS")


def test_flash_attention_s80_batch_head_variation():
    """Test different batch and head counts."""
    print("\nTesting flash_attention_s80 batch/head variation...")
    
    L, S, D = 16, 16, 64
    
    test_cases = [
        (1, 1),
        (2, 4),
        (4, 8),
        (1, 16),
    ]
    
    for N, H in test_cases:
        query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
        key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        
        output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
        ref_output = flash_attention_s80_reference(query, key, value, False, None)
        
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
        print(f"  N={N}, H={H}: PASS")


def test_flash_attention_s80_empty_inputs():
    """Test edge cases with zero-length sequences."""
    print("\nTesting flash_attention_s80 empty inputs...")
    
    N, H, D = 1, 1, 64
    
    # L=0
    query = torch.randn(N, H, 0, D, device="musa", dtype=torch.float32)
    key = torch.randn(N, H, 10, D, device="musa", dtype=torch.float32)
    value = torch.randn(N, H, 10, D, device="musa", dtype=torch.float32)
    
    output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
    assert output.shape == (N, H, 0, D)
    print("  L=0: PASS")
    
    # S=0
    query = torch.randn(N, H, 10, D, device="musa", dtype=torch.float32)
    key = torch.randn(N, H, 0, D, device="musa", dtype=torch.float32)
    value = torch.randn(N, H, 0, D, device="musa", dtype=torch.float32)
    
    output = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
    assert output.shape == (N, H, 10, D)
    print("  S=0: PASS")


def test_flash_attention_s80_vs_pytorch_wrapper():
    """Test that torch.flash_attention_s80 wrapper calls the C++ op."""
    print("\nTesting flash_attention_s80 wrapper consistency...")
    
    N, H, L, S, D = 2, 4, 16, 16, 64
    query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
    key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
    
    # Call via wrapper
    output_wrapper = torch_musa.flash_attention_s80(query, key, value, False, None)
    
    # Call via C++ op directly
    output_cpp = torch.ops.aten.flash_attention_s80(query, key, value, False, None)
    
    torch.testing.assert_close(output_wrapper.cpu(), output_cpp.cpu(), rtol=1e-6, atol=1e-6)
    print("  Wrapper vs C++ op: PASS")


def test_flash_attention_s80_causal_vs_sdpa():
    """Test causal masking matches PyTorch SDPA for L≠S."""
    print("\nTesting flash_attention_s80 causal vs SDPA...")
    
    N, H, D = 1, 2, 64
    
    test_cases = [
        (8, 16),
        (16, 8),
        (7, 13),
        (33, 33),
    ]
    
    for L, S in test_cases:
        query = torch.randn(N, H, L, D, device="musa", dtype=torch.float32)
        key = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        value = torch.randn(N, H, S, D, device="musa", dtype=torch.float32)
        
        # Our implementation
        output = torch.ops.aten.flash_attention_s80(query, key, value, True, None)
        
        # PyTorch SDPA
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / (D ** 0.5)
        )
        
        torch.testing.assert_close(output.cpu(), sdpa_output.cpu(), rtol=1e-3, atol=1e-3)
        print(f"  Causal L={L}, S={S} vs SDPA: PASS")


if __name__ == "__main__":
    test_flash_attention_s80_basic()
    test_flash_attention_s80_dtypes()
    test_flash_attention_s80_head_dims()
    test_flash_attention_s80_sequence_lengths()
    test_flash_attention_s80_causal()
    test_flash_attention_s80_causal_edge_cases()
    test_flash_attention_s80_causal_vs_sdpa()
    test_flash_attention_s80_custom_scale()
    test_flash_attention_s80_batch_head_variation()
    test_flash_attention_s80_empty_inputs()
    test_flash_attention_s80_vs_pytorch_wrapper()
    print("\n✓ All flash_attention_s80 tests passed!")
