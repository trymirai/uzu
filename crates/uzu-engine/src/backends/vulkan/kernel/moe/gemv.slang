import definitions;
import common.hadamard;
import generated.gemm_b_layout;
import generated.hadamard_order;
import generated.quantization_method;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_b_source;
import matmul.gemv.common.reduce;
import matmul.gemv.common.tile;

static const uint MOE_GEMV_SIMDGROUPS = 8;
static const uint MOE_GEMV_THREADS = MOE_GEMV_SIMDGROUPS * SIMD_SIZE;

groupshared float moe_gemv_outputs[HADAMARD_TRANSFORM_BLOCK_SIZE];

[[Variants("T", "bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "GROUP_SIZE == 64 && [4, 8].contains(&BITS)")]]
[[Precondition("encode", "in_vec_size.is_multiple_of(if BITS == 4 { 512 } else { 256 }) && out_vec_size.is_multiple_of(32) && group_count_x == out_vec_size / 32")]]
func MoeGemv<T: IStorageFloat>(
  Ptr<uint8_t, Access.Read> b,
  Ptr<T, Access.Read> scales,
  [[Optional("BITS == 4")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<T, Access.Read> a,
  Ptr<uint, Access.Read> expert_ids,
  Ptr<T, Access.ReadWrite> d,
  Ptr<int, Access.Read> hadamard_factors,
  uint in_vec_size,
  uint out_vec_size,
  uint num_routes,
  uint active_experts_per_token,
  uint group_count_x,
  uint b_expert_stride_bytes,
  uint scale_expert_stride,
  uint zero_point_expert_stride,
  uint scale_output_stride,
  uint scale_group_stride,
  uint zero_point_output_stride,
  uint zero_point_group_stride,
  [[Groups("num_routes")]] uint route,
  [[Groups("group_count_x")]] uint output_tile,
  [[Threads("32")]] uint simd_lane,
  [[Threads("8")]] uint simd_group,
  [[Specialize]] uint GROUP_SIZE,
  [[Specialize]] uint BITS,
  [[Specialize]] bool a_per_token,
) {
  let expert = expert_ids[route];
  let a_row = a_per_token ? route / active_experts_per_token : route;
  let u4 = BITS == 4;
  QuantizedWeights<T> weights = {
    b + expert * b_expert_stride_bytes, scales + expert * scale_expert_stride, scales,
    zero_points + expert * zero_point_expert_stride, GROUP_SIZE, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, u4, false,
    u4 ? QuantizationMethod.ScaleZeroPoint : QuantizationMethod.ScaleSymmetric, in_vec_size, GemmBLayout.RowMajor
  };
  GemvOperands<T, T, T> ops = {
    Ptr<uint, Access.Read>(weights.codes), a + a_row * in_vec_size, d + route * out_vec_size, scales,
    Ptr<uint, Access.Read>(scales)
  };
  GemvParams params = { in_vec_size, out_vec_size, 1, 1.0, 0.0, false, false, false, false };
  let Tile = GemvTile(1, HADAMARD_TRANSFORM_BLOCK_SIZE, GROUP_SIZE * BITS / 64, MOE_GEMV_SIMDGROUPS);
  let tile = OutputTile.make(Tile, true, output_tile, 0, out_vec_size);
  float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < GEMV_MAX_ROWS_PER_LANE; R++) {
      result[I][R] = 0.0;
    }
  }
  QuantBSource.accumulate(result, weights, ops, params, Tile, tile, true, true);
  Reduce.run(result, Tile);

  if (tile.reduction_lane == 0) {
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      moe_gemv_outputs[tile.local_row + R] = T.fromFloat(result[0][R]).toFloat();
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let invocation = subgroup_invocation();
  let block = invocation < HADAMARD_TRANSFORM_BLOCK_SIZE;
  let value = hadamard_transform<MOE_GEMV_THREADS>(block ? moe_gemv_outputs[invocation] : 0.0);
  if (block) {
    let column = output_tile * HADAMARD_TRANSFORM_BLOCK_SIZE + invocation;
    let factor = hadamard_factors[expert * out_vec_size + column];
    d[route * out_vec_size + column] = T.fromFloat(hadamard_factor(value, factor));
  }
}
