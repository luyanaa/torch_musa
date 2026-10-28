#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

// Forward declaration of kernel launcher
at::Tensor flash_attention_s80_forward(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    bool causal,
    c10::optional<double> scale);

// Main dispatch function
at::Tensor FlashAttentionS80(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    bool causal,
    c10::optional<double> scale) {
  // Check input dtypes - support FP32 and FP16
  TORCH_CHECK(
      Q.scalar_type() == at::ScalarType::Float ||
      Q.scalar_type() == at::ScalarType::Half,
      "FlashAttentionS80 only supports Float32 and Float16, got ",
      Q.scalar_type());

  TORCH_CHECK(
      Q.scalar_type() == K.scalar_type() && Q.scalar_type() == V.scalar_type(),
      "Q, K, V must have same dtype");

  // Ensure inputs are contiguous
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  // Device guard
  const c10::musa::OptionalMUSAGuard device_guard(device_of(Q));

  // Launch kernel
  return flash_attention_s80_forward(Q, K, V, causal, scale);
}

}  // namespace musa
}  // namespace at
