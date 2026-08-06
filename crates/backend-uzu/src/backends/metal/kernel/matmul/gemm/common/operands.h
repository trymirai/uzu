#pragma once

#include <metal_stdlib>

#include "../../../generated/gemm.h"
#include "../../common/mxu_fragment/integer_formats.h"
#include "../../common/quant_pack.h"
#include "../../common/quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace operands {

namespace detail {

template <bool WITH_GROUP_SUMS>
struct QuantizedLeftArgs;

template <>
struct QuantizedLeftArgs<false> {
  const device int8_t* values;
  const device float* scales;

  METAL_FUNC void seek_rows(const size_t row, const int leading_dimension, const uint k_offset) thread {
    values += row * leading_dimension + k_offset;
  }

  METAL_FUNC void advance_k(const ushort elements) thread { values += elements; }
};

template <>
struct QuantizedLeftArgs<true> {
  const device int8_t* values;
  const device float* scales;
  const device int32_t* group_sums;

  METAL_FUNC void seek_rows(const size_t row, const int leading_dimension, const uint k_offset) thread {
    values += row * leading_dimension + k_offset;
  }

  METAL_FUNC void advance_k(const ushort elements) thread { values += elements; }
};

template <GemmBPrologueKind SCHEME, ushort BITS, typename ElementType>
struct QuantizedRightArgs;

template <ushort BITS>
struct QuantizedStorage {
  const device uint8_t* values;
  bool signed_codes;

  QuantizedStorage(const device uint8_t* values_, const bool signed_codes_)
      : values(values_), signed_codes(signed_codes_) {}

  METAL_FUNC void seek_block(const size_t block_column, const uint k_offset, const int row_stride_bytes) thread {
    values += block_column * row_stride_bytes + int(k_offset) * get_bytes_per_pack<BITS>() / get_pack_factor<BITS>();
  }

  METAL_FUNC void seek_columns(const ushort tile_column_offset, const int row_stride_bytes) thread {
    values += size_t(tile_column_offset) * row_stride_bytes;
  }

  METAL_FUNC void advance_k(const int bytes) thread { values += bytes; }
};

// Constructors keep the specialized signedness flag mandatory; aggregate
// initialization would silently default an omitted trailing flag to false.

template <ushort BITS, typename ElementType>
struct QuantizedRightArgs<GemmBPrologueKind::ScaleSymmetricDequant, BITS, ElementType> {
  QuantizedStorage<BITS> storage;
  const device ElementType* scales;

  QuantizedRightArgs(const device uint8_t* values_, const device ElementType* scales_, const bool signed_codes_)
      : storage(values_, signed_codes_), scales(scales_) {}
};

template <ushort BITS, typename ElementType>
struct QuantizedRightArgs<GemmBPrologueKind::ScaleBiasDequant, BITS, ElementType> {
  QuantizedStorage<BITS> storage;
  const device ElementType* scales;
  const device ElementType* biases;

  QuantizedRightArgs(
      const device uint8_t* values_,
      const device ElementType* scales_,
      const device ElementType* biases_,
      const bool signed_codes_
  )
      : storage(values_, signed_codes_), scales(scales_), biases(biases_) {}
};

template <ushort BITS, typename ElementType>
struct QuantizedRightArgs<GemmBPrologueKind::ScaleZeroPointDequant, BITS, ElementType> {
  QuantizedStorage<BITS> storage;
  const device ElementType* scales;
  const device uint8_t* zero_points;

  QuantizedRightArgs(
      const device uint8_t* values_,
      const device ElementType* scales_,
      const device uint8_t* zero_points_,
      const bool signed_codes_
  )
      : storage(values_, signed_codes_), scales(scales_), zero_points(zero_points_) {}
};

} // namespace detail

template <GemmBPrologueKind SCHEME, ushort BITS, typename ElementType>
static METAL_FUNC void seek_quantized_metadata(
    thread detail::QuantizedRightArgs<SCHEME, BITS, ElementType>& args,
    const size_t block_column,
    const int groups_per_row,
    const int first_group
) {
  args.scales += block_column * groups_per_row + first_group;
  if constexpr (SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    args.biases += block_column * groups_per_row + first_group;
  } else if constexpr (SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
    args.zero_points +=
        block_column * zero_point_row_stride<BITS>(groups_per_row) + ((BITS == 4) ? (first_group / 2) : first_group);
  }
}

template <typename ElementType_>
struct Dense {
  using ElementType = ElementType_;
};

template <typename Format_, ushort GROUP_SIZE_, GemmBPrologueKind SCHEME_, typename ScaleElementType_>
struct Quantized {
  using Format = Format_;
  using ScaleElementType = ScaleElementType_;
  UZU_CONST ushort BITS = Format::BITS;
  UZU_CONST ushort GROUP_SIZE = GROUP_SIZE_;
  UZU_CONST GemmBPrologueKind SCHEME = SCHEME_;

  static_assert(BITS == 4 || BITS == 8, "quantized integer bits must be 4 or 8");
  static_assert(GROUP_SIZE > 0, "quantized groups must be non-empty");
  static_assert(SCHEME != GemmBPrologueKind::FullPrecision, "quantized operands need a quantization scheme");
};

template <typename ElementType>
struct DenseArgs {
  const device ElementType* values;

  METAL_FUNC void seek_rows(const size_t row, const int leading_dimension, const uint k_offset) thread {
    values += row * leading_dimension + k_offset;
  }

  template <bool TRANSPOSED>
  METAL_FUNC void seek_columns(const size_t column, const uint k_offset, const int leading_dimension) thread {
    values += (TRANSPOSED ? column * leading_dimension : column) +
              (TRANSPOSED ? k_offset : k_offset * uint(leading_dimension));
  }

  METAL_FUNC void advance_k(const ushort elements) thread { values += elements; }
};

template <typename Operand>
struct RightBinding;

template <typename Element>
struct RightBinding<Dense<Element>> {
  using ElementType = Element;
  using Args = DenseArgs<Element>;
  UZU_CONST bool NEEDS_CORRECTION = false;
};

template <typename Format, ushort GROUP_SIZE, GemmBPrologueKind SCHEME, typename ScaleElementType>
struct RightBinding<Quantized<Format, GROUP_SIZE, SCHEME, ScaleElementType>> {
  using ElementType = ScaleElementType;
  using Args = detail::QuantizedRightArgs<SCHEME, Format::BITS, ScaleElementType>;
  UZU_CONST bool NEEDS_CORRECTION = SCHEME != GemmBPrologueKind::ScaleSymmetricDequant;
};

template <typename LeftOperand, typename RightOperand>
struct LeftBinding;

template <typename Element, typename RightOperand>
struct LeftBinding<Dense<Element>, RightOperand> {
  using ElementType = Element;
  using Args = DenseArgs<Element>;
};

template <typename Format, ushort GROUP_SIZE, typename ScaleElementType, typename RightOperand>
struct LeftBinding<
    Quantized<Format, GROUP_SIZE, GemmBPrologueKind::ScaleSymmetricDequant, ScaleElementType>,
    RightOperand> {
  static_assert(Format::BITS == 8, "integer MMA left operands must be 8-bit");
  static_assert(
      metal::is_same<typename Format::UnpackedElement, int8_t>::value,
      "integer MMA left operands must be signed"
  );
  using ElementType = typename Format::UnpackedElement;
  using Args = detail::QuantizedLeftArgs<RightBinding<RightOperand>::NEEDS_CORRECTION>;
};

template <GemmAPrologueKind A_PROLOGUE, typename ElementType>
struct LeftOperandFor;

template <typename ElementType>
struct LeftOperandFor<GemmAPrologueKind::FullPrecision, ElementType> {
  using type = Dense<ElementType>;
};

template <typename ElementType>
struct LeftOperandFor<GemmAPrologueKind::Int8Symmetric, ElementType> {
  using type = Quantized<
      uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>,
      32,
      GemmBPrologueKind::ScaleSymmetricDequant,
      float>;
};

template <GemmBPrologueKind B_PROLOGUE, ushort BITS, ushort GROUP_SIZE, typename ElementType>
struct RightOperandFor {
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant || B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant ||
          B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant,
      "unsupported quantized right operand scheme"
  );
  using type =
      Quantized<uzu::matmul::IntegerFormat<BITS, uzu::matmul::Signedness::Signed>, GROUP_SIZE, B_PROLOGUE, ElementType>;
};

template <ushort BITS, ushort GROUP_SIZE, typename ElementType>
struct RightOperandFor<GemmBPrologueKind::FullPrecision, BITS, GROUP_SIZE, ElementType> {
  using type = Dense<ElementType>;
};

template <typename LeftOperand, typename RightOperand, typename ElementType>
static METAL_FUNC typename LeftBinding<LeftOperand, RightOperand>::Args make_left_args(
    const device ElementType* a,
    const device int8_t* a_int8,
    const device float* a_scales,
    const device int32_t* a_group_sums
) {
  if constexpr (metal::is_same<LeftOperand, Dense<ElementType>>::value) {
    return {a};
  } else if constexpr (RightBinding<RightOperand>::NEEDS_CORRECTION) {
    return {a_int8, a_scales, a_group_sums};
  } else {
    return {a_int8, a_scales};
  }
}

template <typename RightOperand, typename ElementType>
static METAL_FUNC typename RightBinding<RightOperand>::Args make_right_args(
    const device ElementType* b,
    const device ElementType* scales,
    const device ElementType* biases,
    const device uint8_t* zero_points,
    const bool signed_codes
) {
  using Binding = RightBinding<RightOperand>;
  if constexpr (metal::is_same<RightOperand, Dense<ElementType>>::value) {
    return {b};
  } else {
    const device uint8_t* storage = reinterpret_cast<const device uint8_t*>(b);
    if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
      return typename Binding::Args(storage, scales, biases, signed_codes);
    } else if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
      return typename Binding::Args(storage, scales, zero_points, signed_codes);
    } else if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleSymmetricDequant) {
      return typename Binding::Args(storage, scales, signed_codes);
    } else {
      static_assert(
          RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant ||
              RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant ||
              RightOperand::SCHEME == GemmBPrologueKind::ScaleSymmetricDequant,
          "unsupported right operand scheme"
      );
    }
  }
}

} // namespace operands
} // namespace gemm
} // namespace uzu
