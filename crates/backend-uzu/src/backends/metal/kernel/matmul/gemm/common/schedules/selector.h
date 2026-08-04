#pragma once

#include "../../../../generated/gemm.h"
#include "dense.h"
#include "integer.h"
#include "staged.h"

namespace uzu {
namespace gemm {
namespace schedules {

template <
    GemmAPrologueKind A_PROLOGUE,
    GemmBPrologueKind B_PROLOGUE,
    ushort BITS,
    ushort RIGHT_GROUP_SIZE,
    ushort LEFT_GROUP_SIZE = 32>
struct ScheduleFor;

template <GemmBPrologueKind B_PROLOGUE, ushort BITS, ushort RIGHT_GROUP_SIZE, ushort LEFT_GROUP_SIZE>
struct ScheduleFor<GemmAPrologueKind::Int8Symmetric, B_PROLOGUE, BITS, RIGHT_GROUP_SIZE, LEFT_GROUP_SIZE> {
  using type = IntegerSchedule<BITS, RIGHT_GROUP_SIZE, LEFT_GROUP_SIZE, B_PROLOGUE>;
};

template <ushort BITS, ushort RIGHT_GROUP_SIZE, ushort LEFT_GROUP_SIZE>
struct ScheduleFor<
    GemmAPrologueKind::FullPrecision,
    GemmBPrologueKind::FullPrecision,
    BITS,
    RIGHT_GROUP_SIZE,
    LEFT_GROUP_SIZE> {
  using type = DenseSchedule;
};

template <ushort BITS, ushort RIGHT_GROUP_SIZE, ushort LEFT_GROUP_SIZE>
struct ScheduleFor<
    GemmAPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    BITS,
    RIGHT_GROUP_SIZE,
    LEFT_GROUP_SIZE> {
  using type = StagedSchedule<BITS, RIGHT_GROUP_SIZE>;
};

template <ushort BITS, ushort RIGHT_GROUP_SIZE, ushort LEFT_GROUP_SIZE>
struct ScheduleFor<
    GemmAPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleZeroPointDequant,
    BITS,
    RIGHT_GROUP_SIZE,
    LEFT_GROUP_SIZE> {
  using type = StagedSchedule<BITS, RIGHT_GROUP_SIZE>;
};

template <ushort BITS, ushort RIGHT_GROUP_SIZE, ushort LEFT_GROUP_SIZE>
struct ScheduleFor<
    GemmAPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleSymmetricDequant,
    BITS,
    RIGHT_GROUP_SIZE,
    LEFT_GROUP_SIZE> {
  using type = StagedSchedule<BITS, RIGHT_GROUP_SIZE>;
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
