#pragma once

#include "../../../common/soft_cap.h"
#include "../../gemm/common/gemm_rht.h"
#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, bool FULL_TILE>
struct Epilogue {
  using U = float;

  static METAL_FUNC void store(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      threadgroup U* shared_results
  ) {
    const bool use_hadamard =
        Tile::OUTPUT_ROWS >= METAL_SIMD_SIZE && params.output_transform.contains(GemmDTransform::RHT);
    const bool output_writer = tile.writer && tile.reduction_lane == 0;
    if (!use_hadamard && !output_writer) {
      return;
    }

    if (!use_hadamard) {
      write_results<false>(result, ops, params, tile, shared_results);
      return;
    }
    if (output_writer) {
      write_results<true>(result, ops, params, tile, shared_results);
    }

    hadamard_from_shared(ops, params, tile, shared_results);
  }

  static METAL_FUNC void hadamard_from_shared(
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      threadgroup U* shared_results
  ) {
    if constexpr (Tile::OUTPUT_ROWS >= METAL_SIMD_SIZE) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const ushort rows =
          FULL_TILE ? ushort(Tile::INPUT_ROWS) : ushort(min(Tile::INPUT_ROWS, params.batch_size - tile.input_row));
      const ushort cols =
          FULL_TILE ? ushort(Tile::OUTPUT_ROWS) : ushort(min(Tile::OUTPUT_ROWS, params.out_vec_size - tile.tile_row));
      const device BT* bias_block = ops.output_bias;
      if (params.output_transform.contains(GemmDTransform::BIAS)) {
        bias_block += tile.tile_row;
      }
      ThreadContext rht_context{};
      rht_context.simd_lane_id = tile.simd_lane;
      rht_context.simdgroup_index = tile.simd_group;
      apply_output_random_hadamard_transform(
          shared_results,
          ops.d + tile.input_row * params.out_vec_size + tile.tile_row,
          ops.hadamard_factors + tile.tile_row,
          bias_block,
          params.output_transform.contains(GemmDTransform::BIAS),
          rows,
          cols,
          Tile::OUTPUT_ROWS,
          params.out_vec_size,
          Tile::NUM_SIMDGROUPS,
          rht_context
      );
    }
  }

private:
  template <bool STAGE_HADAMARD>
  static METAL_FUNC void write_results(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      threadgroup U* shared_results
  ) {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      const uint input_row = tile.input_row + I;
      if (!(FULL_TILE || input_row < params.batch_size)) {
        return;
      }
      device DT* output = ops.d + input_row * params.out_vec_size;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const uint global_row = params.output_row(tile.row0 + R);
        if (!OutputTile<Tile, FULL_TILE>::row_in_range(global_row, params.out_vec_size)) {
          return;
        }
        if constexpr (STAGE_HADAMARD) {
          shared_results[I * Tile::OUTPUT_ROWS + params.output_row(tile.local_row + R)] =
              transform<false>(result[I][R], global_row, ops, params);
        } else {
          output[global_row] = static_cast<DT>(transform<true>(result[I][R], global_row, ops, params));
        }
      });
    });
  }

public:
  template <bool APPLY_BIAS>
  static METAL_FUNC U
  transform(U value, uint global_row, const thread GemvOperands<AT, BT, DT>& ops, const thread GemvParams& params) {
    if (params.output_transform.contains(GemmDTransform::SCALE)) {
      value *= params.ab_scale;
    }
    if constexpr (APPLY_BIAS) {
      if (params.output_transform.contains(GemmDTransform::BIAS)) {
        value += static_cast<U>(ops.output_bias[global_row]);
      }
    }
    if (params.output_transform.contains(GemmDTransform::SOFT_CAP)) {
      value = apply_soft_cap(value, params.soft_cap);
    }
    return value;
  }
};

} // namespace gemm
} // namespace uzu
