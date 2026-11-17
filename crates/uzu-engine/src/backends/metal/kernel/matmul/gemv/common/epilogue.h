#pragma once

#include "../../../common/soft_cap.h"
#include "../../../hadamard_transform/hadamard_transform.h"
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

    if constexpr (Tile::OUTPUT_ROWS >= METAL_SIMD_SIZE) {
      static_assert(Tile::K_SPLIT == 1 || Tile::INPUT_ROWS == 1, "RHT reuses shared results after Reduce");
      threadgroup_barrier(mem_flags::mem_threadgroup);
      constexpr uint VECTORS_PER_INPUT = Tile::OUTPUT_ROWS / hadamard::VECTOR_SIZE;
      constexpr uint VECTORS = Tile::INPUT_ROWS * VECTORS_PER_INPUT;
      using Rounded = vec<DT, hadamard::VECTOR_SIZE>;
      for (uint first = tile.simd_group * METAL_SIMD_SIZE; first < VECTORS;
           first += Tile::NUM_SIMDGROUPS * METAL_SIMD_SIZE) {
        const uint vector_index = first + tile.simd_lane;
        const uint input_index = vector_index / VECTORS_PER_INPUT;
        const uint local_row = vector_index % VECTORS_PER_INPUT * hadamard::VECTOR_SIZE;
        const uint input_row = tile.input_row + input_index;
        const uint global_row = tile.tile_row + local_row;
        const bool active = vector_index < VECTORS &&
                            (FULL_TILE || (input_row < params.batch_size && global_row < params.out_vec_size));
        const float4 values =
            active ? float4(Rounded(load_vec4(shared_results + input_index * Tile::OUTPUT_ROWS + local_row))) : 0.0f;
        const float4 rotated = hadamard::simdgroup_transform_vector(
            static_cast<ushort>(tile.simd_lane),
            values,
            HADAMARD_TRANSFORM_BLOCK_SIZE
        );
        if (active) {
          auto result = Rounded(rotated * float4(load_vec4(ops.hadamard_factors + global_row)));
          if (params.output_transform.contains(GemmDTransform::BIAS)) {
            result = Rounded(float4(result) + float4(load_vec4(ops.output_bias + global_row)));
          }
          store_vec4(ops.d + input_row * params.output_row_stride + global_row, result);
        }
      }
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
      device DT* output = ops.d + input_row * params.output_row_stride;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const uint global_row = params.output_row(tile.row0 + R);
        if (!OutputTile<Tile, FULL_TILE>::row_in_range(global_row, params.out_vec_size)) {
          return;
        }
        if (params.gathered && !params.gathers(ops.gather_indices[input_row * params.out_vec_size + global_row])) {
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
