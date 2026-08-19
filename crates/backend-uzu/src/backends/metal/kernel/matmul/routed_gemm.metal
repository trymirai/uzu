#include <metal_stdlib>

#include "../common/dsl.h"
#include "../common/soft_cap.h"
#include "../generated/gemm.h"
#include "common/microfloat.h"

using namespace metal;
using namespace uzu::gemm;

#define ROUTED_COLUMNS 128
#define ROUTED_ROWS 4

template <typename AT, typename BT, typename DT, bool MICROFLOAT, uint GROUP_SIZE, bool EXPERT_ROUTED>
VARIANTS(AT, half, bfloat, float)
VARIANTS(BT, half, bfloat, float)
VARIANTS(DT, half, bfloat, float)
VARIANTS(MICROFLOAT, false, true)
VARIANTS(GROUP_SIZE, 0, 16, 32)
VARIANTS(EXPERT_ROUTED, false, true)
CONSTRAINT(MICROFLOAT == (GROUP_SIZE != 0))
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
CONSTRAINT(EXPERT_ROUTED || MICROFLOAT)
KERNEL(RoutedGemm)(
    device const uchar* b,
    device const uchar* scales OPTIONAL(MICROFLOAT),
    device const BT* global_scales OPTIONAL(MICROFLOAT),
    device const AT* a,
    device DT* d,
    device const BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    device const BT* expert_biases OPTIONAL(expert_bias),
    device const uint* offsets OPTIONAL(EXPERT_ROUTED),
    device const uint* grouped_routes OPTIONAL(EXPERT_ROUTED),
    constant uint& route_count,
    constant uint& n,
    constant uint& k,
    constant uint& routes_per_token,
    constant uint& expert_count,
    constant bool& route_inputs,
    constant uint& row_partitions,
    constant float& ab_scale,
    constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool expert_bias SPECIALIZE,
    const uint column_tile GROUPS(n.div_ceil(ROUTED_COLUMNS)),
    const uint expert GROUPS(expert_count),
    const uint partition GROUPS(row_partitions),
    const uint lane THREADS(ROUTED_COLUMNS)
) {
  const uint column = column_tile * ROUTED_COLUMNS + lane;
  if (column >= n || expert >= expert_count) {
    return;
  }

  const uint begin = EXPERT_ROUTED ? offsets[expert] : 0;
  const uint end = EXPERT_ROUTED ? offsets[expert + 1] : route_count;
  const ulong bank_row = ulong(expert) * n + column;
  const device BT* full_precision_weights = reinterpret_cast<const device BT*>(b);
  float global_scale = 1.0f;
  if constexpr (MICROFLOAT) {
    global_scale = float(global_scales[expert]);
  }

  const uint grouped_stride = row_partitions * ROUTED_ROWS;
  for (uint grouped_base = begin + partition * ROUTED_ROWS;
       grouped_base < end;
       grouped_base += grouped_stride) {
    uint routes[ROUTED_ROWS];
    uint input_rows[ROUTED_ROWS];
    bool valid_rows[ROUTED_ROWS];
    float values[ROUTED_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma clang loop unroll(full)
    for (uint row = 0; row < ROUTED_ROWS; ++row) {
      const uint grouped = grouped_base + row;
      const bool in_bounds = grouped < end;
      const uint route = in_bounds ? (EXPERT_ROUTED ? grouped_routes[grouped] : grouped) : route_count;
      routes[row] = route;
      input_rows[row] = route_inputs ? route : route / routes_per_token;
      valid_rows[row] = route < route_count;
    }

    // Keep four rows live so each decoded expert weight is fetched once per tile.
    for (uint inner = 0; inner < k; ++inner) {
      float weight;
      if constexpr (MICROFLOAT) {
        const uchar packed = b[bank_row * (k / 2) + inner / 2];
        const uint code = (inner & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
        const uint exponent = scales[bank_row * (k / GROUP_SIZE) + inner / GROUP_SIZE];
        weight = decode_mxfp4(code, exponent, global_scale);
      } else {
        weight = float(full_precision_weights[bank_row * k + inner]);
      }
      #pragma clang loop unroll(full)
      for (uint row = 0; row < ROUTED_ROWS; ++row) {
        if (valid_rows[row]) {
          const ulong input_index = ulong(input_rows[row]) * k + inner;
          values[row] = fma(float(a[input_index]), weight, values[row]);
        }
      }
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < ROUTED_ROWS; ++row) {
      if (!valid_rows[row]) {
        continue;
      }
      const uint route = routes[row];
      float value = values[row] * ab_scale;
      if (output_transform.contains(GemmDTransform::ACCUMULATE)) {
        value += float(d[ulong(route) * n + column]);
      }
      if (output_transform.contains(GemmDTransform::BIAS)) {
        value += float(output_bias[column]);
      }
      if (expert_bias) {
        value += float(expert_biases[ulong(expert) * n + column]);
      }
      if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
        value = uzu::apply_soft_cap(value, soft_cap);
      }
      d[ulong(route) * n + column] = DT(value);
    }
  }
}
