#include <metal_stdlib>

#include "../common/dsl.h"
#include "../common/soft_cap.h"
#include "../generated/gemm.h"
#include "common/microfloat.h"

using namespace metal;
using namespace uzu::gemm;

#define ROUTED_COLUMNS_PER_SIMDGROUP 4
#define ROUTED_SIMDGROUPS 4
#define ROUTED_COLUMNS 16
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
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(ROUTED_SIMDGROUPS)
) {
  if (expert >= expert_count) {
    return;
  }

  const uint begin = EXPERT_ROUTED ? offsets[expert] : 0;
  const uint end = EXPERT_ROUTED ? offsets[expert + 1] : route_count;
  const uint column_base = column_tile * ROUTED_COLUMNS + simd_group * ROUTED_COLUMNS_PER_SIMDGROUP;
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
    float values[ROUTED_ROWS][ROUTED_COLUMNS_PER_SIMDGROUP] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    #pragma clang loop unroll(full)
    for (uint row = 0; row < ROUTED_ROWS; ++row) {
      const uint grouped = grouped_base + row;
      const bool in_bounds = grouped < end;
      const uint route = in_bounds ? (EXPERT_ROUTED ? grouped_routes[grouped] : grouped) : route_count;
      routes[row] = route;
      input_rows[row] = route_inputs ? route : route / routes_per_token;
      valid_rows[row] = route < route_count;
    }

    // Split K across each simdgroup and keep four route rows live so a
    // coalesced weight vector feeds the complete row tile.
    for (uint inner = simd_lane * 4; inner < k; inner += 32 * 4) {
      float4 input_values[ROUTED_ROWS] = {
          float4(0.0f),
          float4(0.0f),
          float4(0.0f),
          float4(0.0f),
      };
      #pragma clang loop unroll(full)
      for (uint row = 0; row < ROUTED_ROWS; ++row) {
        if (!valid_rows[row]) {
          continue;
        }
        const device AT* input = a + ulong(input_rows[row]) * k + inner;
        if (inner + 4 <= k) {
          input_values[row] = static_cast<float4>(*reinterpret_cast<const device vec<AT, 4>*>(input));
          continue;
        }
        for (uint offset = 0; offset < k - inner; ++offset) {
          input_values[row][offset] = float(input[offset]);
        }
      }

      #pragma clang loop unroll(full)
      for (uint output = 0; output < ROUTED_COLUMNS_PER_SIMDGROUP; ++output) {
        const uint column = column_base + output;
        if (column >= n) {
          continue;
        }
        const ulong bank_row = ulong(expert) * n + column;
        float4 weight;
        if constexpr (MICROFLOAT) {
          const device uchar* codes = b + bank_row * (k / 2) + inner / 2;
          const ushort packed = *reinterpret_cast<const device ushort*>(codes);
          const uint exponent = scales[bank_row * (k / GROUP_SIZE) + inner / GROUP_SIZE];
          const float scale = decode_e8m0(exponent) * global_scale;
          weight = scale * float4(
              decode_e2m1(packed & 0x0fu),
              decode_e2m1((packed >> 4u) & 0x0fu),
              decode_e2m1((packed >> 8u) & 0x0fu),
              decode_e2m1(packed >> 12u)
          );
        } else if (inner + 4 <= k) {
          const device BT* weights = full_precision_weights + bank_row * k + inner;
          weight = static_cast<float4>(*reinterpret_cast<const device vec<BT, 4>*>(weights));
        } else {
          weight = float4(0.0f);
          const device BT* weights = full_precision_weights + bank_row * k + inner;
          for (uint offset = 0; offset < k - inner; ++offset) {
            weight[offset] = float(weights[offset]);
          }
        }
        #pragma clang loop unroll(full)
        for (uint row = 0; row < ROUTED_ROWS; ++row) {
          values[row][output] += dot(input_values[row], weight);
        }
      }
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < ROUTED_ROWS; ++row) {
      if (!valid_rows[row]) {
        continue;
      }
      const uint route = routes[row];
      #pragma clang loop unroll(full)
      for (uint output = 0; output < ROUTED_COLUMNS_PER_SIMDGROUP; ++output) {
        const uint column = column_base + output;
        const float total = simd_sum(values[row][output]);
        if (simd_lane != 0 || column >= n) {
          continue;
        }
        float value = total * ab_scale;
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
}
