#include <metal_stdlib>

#include "../common/dsl.h"
#include "../common/soft_cap.h"
#include "../generated/gemm.h"

using namespace metal;
using namespace uzu::gemm;

#define ROUTED_COLUMNS 128

template <typename AT, typename BT, typename DT>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
KERNEL(RoutedGemm)(
    device const BT* b,
    device const AT* a,
    device DT* d,
    device const BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    device const BT* expert_biases OPTIONAL(expert_bias),
    device const uint* offsets,
    device const uint* grouped_routes,
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

  const uint begin = offsets[expert];
  const uint end = offsets[expert + 1];
  const device BT* weight_row = b + (ulong(expert) * n + column) * k;

  for (uint grouped = begin + partition; grouped < end; grouped += row_partitions) {
    const uint route = grouped_routes[grouped];
    if (route >= route_count) {
      continue;
    }
    const uint a_row = route_inputs ? route : route / routes_per_token;
    const device AT* input_row = a + ulong(a_row) * k;
    float value = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
      value = fma(float(input_row[inner]), float(weight_row[inner]), value);
    }

    value *= ab_scale;
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
