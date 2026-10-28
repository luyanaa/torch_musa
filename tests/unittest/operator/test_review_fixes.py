"""Verification tests for review fixes."""

import torch
import torch_musa
from torch_musa import testing


def test_hardtanh_non_contiguous():
    """Test HardTanh with non-contiguous Float/Half inputs."""
    print("Testing HardTanh non-contiguous...")
    
    # Test Float
    base = torch.randn(8, 16, device="musa", dtype=torch.float32)
    x = base[:, :4]  # stride (16, 1), non-contiguous
    assert not x.is_contiguous(), "x should be non-contiguous"
    
    y = torch.nn.functional.hardtanh(x, -1, 1)
    ref = torch.nn.functional.hardtanh(x.cpu(), -1, 1)
    
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-5, atol=1e-5)
    print("  Float non-contiguous: PASS")
    
    # Test Half
    base_half = torch.randn(8, 16, device="musa", dtype=torch.float16)
    x_half = base_half[:, :4]
    y_half = torch.nn.functional.hardtanh(x_half, -1, 1)
    ref_half = torch.nn.functional.hardtanh(x_half.cpu(), -1, 1)
    
    torch.testing.assert_close(y_half.cpu(), ref_half, rtol=1e-3, atol=1e-3)
    print("  Half non-contiguous: PASS")
    
    # Test strided view with stride > 1
    base2 = torch.randn(8, 16, device="musa", dtype=torch.float32)
    x2 = base2[:, ::2]  # stride (16, 2)
    y2 = torch.nn.functional.hardtanh(x2, -0.5, 0.5)
    ref2 = torch.nn.functional.hardtanh(x2.cpu(), -0.5, 0.5)
    
    torch.testing.assert_close(y2.cpu(), ref2, rtol=1e-5, atol=1e-5)
    print("  Strided view: PASS")


def test_gated_silu_shapes():
    """Test GatedSiLU with various shapes and edge cases."""
    print("\nTesting GatedSiLU shapes...")
    
    # Test typical shapes
    shapes = [
        (4, 128, 1536),  # typical decode
        (1, 1, 128),     # minimal
        (3, 7, 10),      # odd dimensions
    ]
    
    for shape in shapes:
        # Float32
        x = torch.randn(shape, dtype=torch.float32, device="musa")
        y = torch.gated_silu(x)
        
        hidden = shape[-1] // 2
        x_part = x[..., :hidden]
        gate = x[..., hidden:]
        ref = torch.nn.functional.silu(x_part) * gate
        
        torch.testing.assert_close(y.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
        print(f"  Shape {shape} float32: PASS")
        
        # Float16
        x_half = torch.randn(shape, dtype=torch.float16, device="musa")
        y_half = torch.gated_silu(x_half)
        
        x_part_half = x_half[..., :hidden]
        gate_half = x_half[..., hidden:]
        ref_half = torch.nn.functional.silu(x_part_half) * gate_half
        
        torch.testing.assert_close(y_half.cpu(), ref_half.cpu(), rtol=1e-2, atol=1e-2)
        print(f"  Shape {shape} float16: PASS")
    
    # Test BF16 on arch >= 220
    if testing.get_musa_arch() >= 22:
        print("  Testing BF16 (arch >= 220)...")
        x_bf16 = torch.randn(4, 128, 1536, dtype=torch.bfloat16, device="musa")
        y_bf16 = torch.gated_silu(x_bf16)
        
        hidden = 768
        x_part_bf16 = x_bf16[..., :hidden]
        gate_bf16 = x_bf16[..., hidden:]
        ref_bf16 = torch.nn.functional.silu(x_part_bf16) * gate_bf16
        
        torch.testing.assert_close(y_bf16.cpu(), ref_bf16.cpu(), rtol=1e-2, atol=1e-2)
        print("  BF16: PASS")
    else:
        print("  BF16: SKIPPED (arch < 220)")


def test_clamp_non_contiguous():
    """Test ClampMin_/ClampMax_ with non-contiguous views."""
    print("\nTesting Clamp non-contiguous...")
    
    # Test clamp_min_ with strided view
    base = torch.randn(8, 16, device="musa", dtype=torch.float32)
    x = base[:, ::2]  # stride (16, 2), non-contiguous
    assert not x.is_contiguous(), "x should be non-contiguous"
    
    x_cpu = x.cpu().clone()
    
    x.clamp_min_(0)
    x_cpu.clamp_min_(0)
    
    torch.testing.assert_close(x.cpu(), x_cpu, rtol=1e-5, atol=1e-5)
    print("  clamp_min_ strided: PASS")
    
    # Test clamp_max_ with strided view
    base2 = torch.randn(8, 16, device="musa", dtype=torch.float32)
    x2 = base2[:, ::3]  # stride (16, 3)
    x2_cpu = x2.cpu().clone()
    
    x2.clamp_max_(0.5)
    x2_cpu.clamp_max_(0.5)
    
    torch.testing.assert_close(x2.cpu(), x2_cpu, rtol=1e-5, atol=1e-5)
    print("  clamp_max_ strided: PASS")
    
    # Test with slice view
    base3 = torch.randn(8, 16, device="musa", dtype=torch.float32)
    x3 = base3[:, :4]  # stride (16, 1), non-contiguous
    x3_cpu = x3.cpu().clone()
    
    x3.clamp_min_(-0.5)
    x3.clamp_max_(0.5)
    x3_cpu.clamp_min_(-0.5)
    x3_cpu.clamp_max_(0.5)
    
    torch.testing.assert_close(x3.cpu(), x3_cpu, rtol=1e-5, atol=1e-5)
    print("  clamp_min_ + clamp_max_ slice: PASS")


if __name__ == "__main__":
    test_hardtanh_non_contiguous()
    test_gated_silu_shapes()
    test_clamp_non_contiguous()
    print("\n✓ All verification tests passed!")
