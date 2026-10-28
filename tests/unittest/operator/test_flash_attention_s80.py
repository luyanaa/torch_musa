"""Test Flash Attention S80 implementation."""

import torch
import torch_musa
import pytest
from torch_musa.flash_attention_s80 import flash_attention_s80, flash_attention_s80_reference


class TestFlashAttentionS80:
    """Test suite for Flash Attention S80."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("causal", [False, True])
    def test_basic_shapes(self, dtype, causal):
        """Test basic attention shapes."""
        N, H, L, S, D = 2, 8, 64, 64, 128

        query = torch.randn(N, H, L, D, dtype=dtype, device="musa")
        key = torch.randn(N, H, S, D, dtype=dtype, device="musa")
        value = torch.randn(N, H, S, D, dtype=dtype, device="musa")

        # Test implementation
        output = flash_attention_s80(query, key, value, causal=causal)

        # Test reference
        ref_output = flash_attention_s80_reference(query, key, value, causal=causal)

        # Compare
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("L,S", [(32, 32), (64, 128), (128, 64)])
    def test_different_seq_lengths(self, L, S):
        """Test different sequence lengths."""
        N, H, D = 1, 4, 64

        query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa")
        key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")
        value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")

        output = flash_attention_s80(query, key, value, causal=False)
        ref_output = flash_attention_s80_reference(query, key, value, causal=False)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_different_head_dims(self, D):
        """Test different head dimensions."""
        N, H, L, S = 1, 4, 64, 64

        query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa")
        key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")
        value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")

        output = flash_attention_s80(query, key, value, causal=False)
        ref_output = flash_attention_s80_reference(query, key, value, causal=False)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)

    def test_custom_scale(self):
        """Test custom scale factor."""
        N, H, L, S, D = 1, 4, 64, 64, 128

        query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa")
        key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")
        value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")

        scale = 0.5
        output = flash_attention_s80(query, key, value, causal=False, scale=scale)
        ref_output = flash_attention_s80_reference(query, key, value, causal=False, scale=scale)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)

    def test_causal_mask(self):
        """Test causal masking."""
        N, H, L, S, D = 1, 4, 128, 128, 64

        query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa")
        key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")
        value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")

        output = flash_attention_s80(query, key, value, causal=True)
        ref_output = flash_attention_s80_reference(query, key, value, causal=True)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)

    def test_llama_like_shape(self):
        """Test Llama-like attention shape."""
        # Llama-2 7B: H=32, D=128, typical L=S=2048
        # Use smaller size for testing
        N, H, L, S, D = 1, 8, 512, 512, 128

        query = torch.randn(N, H, L, D, dtype=torch.float16, device="musa")
        key = torch.randn(N, H, S, D, dtype=torch.float16, device="musa")
        value = torch.randn(N, H, S, D, dtype=torch.float16, device="musa")

        output = flash_attention_s80(query, key, value, causal=True)
        ref_output = flash_attention_s80_reference(query, key, value, causal=True)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-2, atol=1e-2)

    def test_gqa_shape(self):
        """Test Grouped Query Attention shape."""
        # GQA: fewer KV heads than Q heads
        N, H_q, H_kv, L, S, D = 1, 32, 8, 64, 64, 128

        query = torch.randn(N, H_q, L, D, dtype=torch.float32, device="musa")
        key = torch.randn(N, H_kv, S, D, dtype=torch.float32, device="musa")
        value = torch.randn(N, H_kv, S, D, dtype=torch.float32, device="musa")

        # Expand K/V to match Q heads
        key_expanded = key.repeat_interleave(H_q // H_kv, dim=1)
        value_expanded = value.repeat_interleave(H_q // H_kv, dim=1)

        output = flash_attention_s80(query, key_expanded, value_expanded, causal=False)
        ref_output = flash_attention_s80_reference(query, key_expanded, value_expanded, causal=False)

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)

    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        N, H, L, S, D = 1, 4, 64, 64, 128

        # Use large values to test numerical stability
        query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa") * 10.0
        key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa") * 10.0
        value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa") * 10.0

        output = flash_attention_s80(query, key, value, causal=False)
        ref_output = flash_attention_s80_reference(query, key, value, causal=False)

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # Run basic test
    print("Testing Flash Attention S80...")

    N, H, L, S, D = 1, 4, 64, 64, 128
    query = torch.randn(N, H, L, D, dtype=torch.float32, device="musa")
    key = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")
    value = torch.randn(N, H, S, D, dtype=torch.float32, device="musa")

    print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")

    output = flash_attention_s80(query, key, value, causal=True)
    ref_output = flash_attention_s80_reference(query, key, value, causal=True)

    print(f"Output shape: {output.shape}")
    print(f"Max diff: {(output - ref_output).abs().max().item()}")
    print(f"Mean diff: {(output - ref_output).abs().mean().item()}")

    torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
    print("✓ Test passed!")
