#pragma once

#include <metal_stdlib>

#include "../../../generated/gemm.h"
#include "../../../generated/matmul.h"
#include "../../common/mxu_fragment/integer_formats.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace operands {

template <GemmAPrologueKind PROLOGUE, typename Element, ushort ACTIVATION_GROUP_SIZE>
struct LeftOperand {
  UZU_CONST bool QUANTIZED = PROLOGUE == GemmAPrologueKind::Int8Symmetric;
  UZU_CONST ushort BITS = QUANTIZED ? 8 : 0;
  UZU_CONST ushort GROUP_SIZE = QUANTIZED ? ACTIVATION_GROUP_SIZE : 0;
  static_assert(
      QUANTIZED == (ACTIVATION_GROUP_SIZE != 0),
      "activation group size must be present exactly for int8 activations"
  );

  using CodeElement = int8_t;
  using ScaleElement = float;
  using DenseElement = Element;
  using ElementType = metal::conditional_t<QUANTIZED, CodeElement, DenseElement>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;

  template <ushort BLOCK_K>
  static constexpr ushort outer_block_k() {
    if constexpr (QUANTIZED) {
      static_assert(ACTIVATION_GROUP_SIZE % MXU_SIMDGROUP_BLOCK_K == 0, "activation groups must contain MMA chunks");
      return ACTIVATION_GROUP_SIZE;
    } else {
      return BLOCK_K;
    }
  }
};

template <GemmBPrologueKind PROLOGUE, ushort BITS_, ushort GROUP_SIZE_, typename Element>
struct RightOperand {
  UZU_CONST bool QUANTIZED = PROLOGUE != GemmBPrologueKind::FullPrecision;
  /// The weights are hashed out of a bit tape rather than read as codes, so the
  /// operand is DECODED INTO THREADGROUP MEMORY once per K block and the K loop
  /// reads its fragments from there. `GROUP_SIZE` is that staging block, not a
  /// quantization group: there is one scale per row.
  UZU_CONST bool STAGED = PROLOGUE == GemmBPrologueKind::Trellis;
  UZU_CONST ushort BITS = QUANTIZED ? BITS_ : 0;
  UZU_CONST ushort GROUP_SIZE = QUANTIZED ? GROUP_SIZE_ : 0;
  UZU_CONST GemmBPrologueKind SCHEME = PROLOGUE;
  UZU_CONST bool NEEDS_CORRECTION = QUANTIZED && !STAGED && PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant;

  static_assert(!QUANTIZED || BITS_ == 4 || BITS_ == 8, "quantized integer weights must use 4 or 8 bits");
  static_assert(!QUANTIZED || PROLOGUE != GemmBPrologueKind::FullPrecision, "quantized weights need a scheme");
  static_assert(!STAGED || BITS_ == 8, "a staged decode hands the MXU int8");

  using CodeElement = int8_t;
  using ScaleElement = Element;
  using DenseElement = Element;
  /// What the MMA fragments hold. The trellis decode emits int8 directly, so the
  /// staged tile IS the operand; every other scheme dequantizes into `Element`.
  using ElementType = metal::conditional_t<STAGED, CodeElement, DenseElement>;
  using Format = metal::conditional_t<
      QUANTIZED,
      uzu::matmul::IntegerFormat<BITS_, uzu::matmul::Signedness::Signed>,
      uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>>;

  template <ushort BLOCK_K>
  static constexpr ushort outer_block_k() {
    if constexpr (QUANTIZED) {
      static_assert(GROUP_SIZE_ % MXU_SIMDGROUP_BLOCK_K == 0, "weight groups must contain complete MMA chunks");
      static_assert(BLOCK_K % GROUP_SIZE_ == 0, "tile block K must contain complete weight groups");
      return GROUP_SIZE_;
    } else {
      return BLOCK_K;
    }
  }
};

template <typename Left>
struct LeftStorage {
  const device typename Left::DenseElement* values;
  const device int8_t* codes;
  const device float* scales;
  const device int32_t* group_sums;

  METAL_FUNC const device int32_t* correction_sums() const thread {
    static_assert(Left::QUANTIZED, "correction_sums is only valid for int8 activations");
    return group_sums;
  }
};

template <typename Right>
struct RightStorage {
  const device typename Right::DenseElement* dense;
  const device uint8_t* codes;
  const device typename Right::ScaleElement* scales;
  const device typename Right::ScaleElement* biases;
  const device uint8_t* zero_points;
  /// The bit tape, and the window/hash/scale constants that read it. Both null
  /// unless `Right::STAGED`; `scales` doubles as the one-scale-per-row vector.
  const device uint* tape;
  const constant uzu::matmul::TrellisParams* trellis;
  bool signed_codes;

  METAL_FUNC const device typename Right::ScaleElement* bias() const thread {
    static_assert(Right::SCHEME == GemmBPrologueKind::ScaleBiasDequant, "bias is only valid for ScaleBiasDequant");
    return biases;
  }

  METAL_FUNC const device uint8_t* zp() const thread {
    static_assert(
        Right::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant,
        "zp is only valid for ScaleZeroPointDequant"
    );
    return zero_points;
  }
};

template <typename Left, typename Element>
METAL_FUNC LeftStorage<Left> pack_left(
    const device Element* a,
    const device int8_t* codes,
    const device float* scales,
    const device int32_t* group_sums
) {
  if constexpr (Left::QUANTIZED) {
    return {nullptr, codes, scales, group_sums};
  } else {
    return {a, nullptr, nullptr, nullptr};
  }
}

template <typename Right, typename Element>
METAL_FUNC RightStorage<Right> pack_right(
    const device Element* dense,
    const device Element* scales,
    const device Element* biases,
    const device uint8_t* zero_points,
    const constant uzu::matmul::TrellisParams* trellis,
    const bool signed_codes
) {
  if constexpr (!Right::QUANTIZED) {
    return {dense, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};
  } else if constexpr (Right::STAGED) {
    return {nullptr, nullptr, scales, nullptr, nullptr, reinterpret_cast<const device uint*>(dense), trellis, true};
  } else {
    return {
        nullptr,
        reinterpret_cast<const device uint8_t*>(dense),
        scales,
        Right::SCHEME == GemmBPrologueKind::ScaleBiasDequant ? biases : nullptr,
        Right::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant ? zero_points : nullptr,
        nullptr,
        nullptr,
        signed_codes
    };
  }
}

/// The threadgroup block the schedule stages `B` into: the `uint`-typed
/// allocation for a trellis tile, `b_shared` for every other scheme. See
/// `GEMM_TRELLIS_TG_WORDS` in `gemm.metal` for why the tile needs its own.
template <typename Right, typename Element>
METAL_FUNC threadgroup typename Right::ElementType* stage_block(
    threadgroup Element* b_shared,
    threadgroup uint* b_trellis
) {
  if constexpr (Right::STAGED) {
    return reinterpret_cast<threadgroup typename Right::ElementType*>(b_trellis);
  } else {
    return b_shared;
  }
}

template <GemmAPrologueKind PROLOGUE, typename Element, ushort ACTIVATION_GROUP_SIZE>
using LeftOperandFor = LeftOperand<PROLOGUE, Element, ACTIVATION_GROUP_SIZE>;

template <GemmBPrologueKind PROLOGUE, ushort BITS, ushort GROUP_SIZE, typename Element>
using RightOperandFor = RightOperand<PROLOGUE, BITS, GROUP_SIZE, Element>;

} // namespace operands
} // namespace gemm
} // namespace uzu
