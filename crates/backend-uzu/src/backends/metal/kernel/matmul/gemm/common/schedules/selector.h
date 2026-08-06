#pragma once

#include "../../../../generated/gemm.h"
#include "../operands.h"
#include "dense.h"
#include "integer.h"
#include "staged.h"

namespace uzu {
namespace gemm {
namespace schedules {

template <typename LeftOperand, typename RightOperand>
struct ScheduleFor;

template <typename LeftElementType, typename RightElementType>
struct ScheduleFor<operands::Dense<LeftElementType>, operands::Dense<RightElementType>> {
  using type = DenseSchedule;
};

template <typename LeftElementType, typename Format, ushort GROUP_SIZE, GemmBPrologueKind SCHEME, typename ElementType>
struct ScheduleFor<operands::Dense<LeftElementType>, operands::Quantized<Format, GROUP_SIZE, SCHEME, ElementType>> {
  using type = StagedSchedule;
};

template <
    typename LeftFormat,
    ushort LEFT_GROUP_SIZE,
    typename RightFormat,
    ushort GROUP_SIZE,
    GemmBPrologueKind SCHEME,
    typename ElementType>
struct ScheduleFor<
    operands::Quantized<LeftFormat, LEFT_GROUP_SIZE, GemmBPrologueKind::ScaleSymmetricDequant, float>,
    operands::Quantized<RightFormat, GROUP_SIZE, SCHEME, ElementType>> {
  using type = IntegerSchedule<
      operands::Quantized<LeftFormat, LEFT_GROUP_SIZE, GemmBPrologueKind::ScaleSymmetricDequant, float>,
      operands::Quantized<RightFormat, GROUP_SIZE, SCHEME, ElementType>>;
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
