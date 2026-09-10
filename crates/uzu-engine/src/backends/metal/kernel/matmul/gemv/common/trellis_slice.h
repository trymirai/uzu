#pragma once

#include "../../../common/integral_constant.h"
#include "../../../generated/matmul.h"
#include "../../common/trellis_decode.h"
#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

/// The trellis GEMV's per-row metadata: ONE scale for the whole row.
///
/// Every other B source reloads a scale per quantization group, which is why
/// `QuantMetadata` is folded once per group. A tape has no groups -- one scale
/// covers the row -- so this is loaded once before the K loop and folded once
/// after it, and the K loop accumulates the raw integer dot product.
/// `codebook_scale` (`1 / rms(codebook)`) rides on the row scale for the same
/// reason: the decode emits small integers and this keeps the per-weight cost at
/// one fma.
template <typename Tile, typename AT, typename BT, typename DT>
struct TrellisMetadata {
private:
  float scale[Tile::ROWS_PER_LANE];

public:
  METAL_FUNC void load(
      const thread uint (&weight_rows)[Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const constant uzu::matmul::TrellisParams& trellis
  ) thread {
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      scale[R] = float(ops.scales[weight_rows[R]]) * trellis.codebook_scale;
    });
  }

  METAL_FUNC void fold(
      thread float (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread float (&partial)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE]
  ) const thread {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        result[I][R] = fma(scale[R], partial[I][R], result[I][R]);
      });
    });
  }
};

/// One lane's slice of a trellis tape: a CONTIGUOUS run of `STATES_PER_LANE`
/// trellis states per weight row, read with one load and decoded in registers.
///
/// `QuantSlice`'s lane owns a contiguous run of CODES; this lane owns a
/// contiguous run of STATES, which is the same thing once the tape is the only
/// addressable form the weights have. Two consequences, both wanted:
///
///   * the run spans `(STATES_PER_LANE - 1) * KV + L` bits, so ONE load of
///     `RAW_WORDS` words and one funnel shift serve every state in it, against
///     two device loads per state for a lane-strided split;
///   * the lane's `VALUES_PER_LANE` activations are contiguous, so the slot
///     loads merge into wide ones, and the simdgroup still covers one contiguous
///     activation region -- just at a coarser granularity.
///
/// THE 128-BIT WINDOW. `TrellisConfig::is_valid` guarantees `KV <= L <= 32`, so
/// with `STATES_PER_LANE == 4` the run spans at most `3*32 + 32 == 128` bits and
/// the normalized run always fits in two `ulong`s. That is what lets `L` and
/// `KV` stay RUNTIME values: the states come out of a 128-bit register pair by
/// shifting it down `KV` bits at a time, so nothing here needs a compile-time
/// bit offset, a word index or a branch. A wider run would need one.
///
/// ADDRESS ARITHMETIC. One block of `REDUCTION_LANES * STATES_PER_LANE` steps is
/// `REDUCTION_LANES * STATES_PER_LANE * KV` bits, which for `V == 4` and eight
/// reduction lanes is always a whole number of words. `bit_offset & 31` is
/// therefore loop-invariant and the row pointers just walk down, so the K loop
/// never recomputes a `>> 5`, a `& 31` or a row product.
/// `KV` is the tape's bits-per-step when the caller knows it at compile time and
/// 0 when it does not; see the state extraction in `accumulate`.
template <typename Tile, typename AT, typename BT, typename DT, bool FULL_TILE, uint KV>
struct TrellisSlice {
  using U = float;

public:
  /// Weights per trellis step. Fixed by the codebook: a 32-bit hash has four
  /// bytes.
  UZU_CONST uint V = uzu::trellis::TRELLIS_V;
  /// Trellis states one lane decodes per K block. Four is the widest run that
  /// still fits a 128-bit window at `KV == L == 32`; see above.
  UZU_CONST uint STATES_PER_LANE = 4;
  UZU_CONST uint VALUES_PER_LANE = STATES_PER_LANE * V;
  /// K values one full reduction step covers.
  UZU_CONST uint BLOCK_VALUES = Tile::REDUCTION_LANES * VALUES_PER_LANE;

private:
  /// Bits the run spans, once normalized to bit 0: `STATES_PER_LANE - 1` steps
  /// of `KV` plus one window. With a compile-time `KV` that is 68 bits at
  /// `KV == 12`, not the 128 a runtime `KV <= 32` has to be sized for, so the
  /// run is a word shorter and one device load cheaper as well.
  UZU_CONST uint STEP_BITS = (KV != 0u) ? KV : uzu::trellis::TRELLIS_MAX_L;
  UZU_CONST uint SPAN_BITS = (STATES_PER_LANE - 1u) * STEP_BITS + uzu::trellis::TRELLIS_MAX_L;
  /// Words of the normalized run, and words loaded to produce them: the funnel
  /// shift reads one word past the last.
  UZU_CONST uint RUN_WORDS = (SPAN_BITS + 31u) / 32u;
  UZU_CONST uint RAW_WORDS = RUN_WORDS + 1;

  static_assert(Tile::GROUP_LANES == 1, "a trellis lane owns whole trellis states");
  // 32 steps of KV bits is KV whole words for any KV, which is what makes
  // `block_words` exact and `shift` loop-invariant.
  static_assert((BLOCK_VALUES / V) % 32 == 0, "one K block must advance the tape by whole words");

  /// The run normalized to bit 0, little-endian.
  uint run[Tile::ROWS_PER_LANE][RUN_WORDS];
  /// This lane's tape word for each weight row, and the bit of the run's base
  /// inside it -- loop-invariant, see above.
  const device uint* head[Tile::ROWS_PER_LANE];
  uint shift;
  uint block_words;

public:
  /// `weight_rows` are the tape rows this lane reduces; `reduction_lane` picks
  /// its run inside the first K block.
  static METAL_FUNC TrellisSlice make(
      const device uint* tape,
      const thread uint (&weight_rows)[Tile::ROWS_PER_LANE],
      uint reduction_lane,
      uint in_vec_size,
      const constant uzu::matmul::TrellisParams& trellis
  ) {
    const uint kv = trellis.k_bits_per_step;
    const uint steps = in_vec_size / V;
    const uzu::trellis::Walk run =
        uzu::trellis::walk(steps, STATES_PER_LANE * (reduction_lane + 1u) - 1u, BLOCK_VALUES / V, kv);
    const uint stride = trellis.row_stride_words;

    TrellisSlice slice;
    slice.shift = run.shift;
    slice.block_words = run.block_words;
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      slice.head[R] = tape + (ulong)weight_rows[R] * (ulong)stride + (ulong)run.word_offset;
    });
    return slice;
  }

  /// One load per weight row, then one funnel shift that normalizes the run to
  /// bit 0. Every load is issued before any shift, so the runs arrive together.
  METAL_FUNC void load_weights() thread {
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      uint raw[RAW_WORDS];
      uzu::const_for_loop<0, int(RAW_WORDS), 1>([&](auto i) UZU_ALWAYS_INLINE { raw[i.value] = head[R][i.value]; });
      uzu::const_for_loop<0, int(RUN_WORDS), 1>([&](auto i) UZU_ALWAYS_INLINE {
        run[R][i.value] = uint((((ulong)raw[i.value + 1] << 32) | (ulong)raw[i.value]) >> shift);
      });
    });
  }

  /// Step to the next K block. A block is a whole number of tape words, so the
  /// shift stays put and only the pointers move.
  METAL_FUNC void advance() thread {
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      head[R] -= block_words;
    });
  }

  /// Decode this lane's run and accumulate it into the batch. `column` is the
  /// first K column of the run.
  METAL_FUNC void accumulate(
      thread U (&partial)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      uint column,
      uint batch_remaining,
      const constant uzu::matmul::TrellisParams& trellis
  ) const thread {
    const uint state_mask = uzu::trellis::state_mask(trellis.l);

    // The run's bit offset RISES as the step FALLS, so slot
    // `STATES_PER_LANE - 1 - i` is the state at bit `i * KV`.
    //
    // With a compile-time `KV` that bit is a compile-time word and offset, so
    // the four states are four INDEPENDENT extractions off the run -- which is
    // the point, in a kernel whose limiter is latency rather than issue. The
    // runtime form has to walk a 128-bit register pair down `kv` bits per
    // state, and each state then waits on the one before it.
    uint states[Tile::ROWS_PER_LANE][STATES_PER_LANE];
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      if constexpr (KV != 0u) {
        uzu::const_for_loop<0, int(STATES_PER_LANE), 1>([&](auto i) UZU_ALWAYS_INLINE {
          constexpr uint BIT = uint(i.value) * KV;
          constexpr uint WORD = BIT >> 5u;
          constexpr uint OFFSET = BIT & 31u;
          // A window that straddles a word needs the next one, and it always
          // exists: straddling means `32*(WORD+1) < BIT + L <= SPAN_BITS`.
          static_assert(OFFSET == 0u || WORD + 1u < RUN_WORDS, "run is too short for its top window");
          if constexpr (OFFSET == 0u) {
            states[R][STATES_PER_LANE - 1u - uint(i.value)] = run[R][WORD] & state_mask;
          } else {
            states[R][STATES_PER_LANE - 1u - uint(i.value)] =
                ((run[R][WORD] >> OFFSET) | (run[R][WORD + 1u] << (32u - OFFSET))) & state_mask;
          }
        });
      } else {
        const uint kv = trellis.k_bits_per_step;
        // `window` is 64 bits wide, so this is what `spare` has to be lifted by
        // to refill the bits `window` just shifted out. `KV >= 1`, so it is a
        // legal shift.
        const uint refill = 64u - kv;
        ulong window = (ulong)run[R][0] | ((ulong)run[R][1] << 32);
        ulong spare = (ulong)run[R][2] | ((ulong)run[R][3] << 32);
        uzu::const_for_loop<0, int(STATES_PER_LANE), 1>([&](auto i) UZU_ALWAYS_INLINE {
          states[R][STATES_PER_LANE - 1u - uint(i.value)] = uint(window) & state_mask;
          window = (window >> kv) | (spare << refill);
          spare >>= kv;
        });
      }
    });

    uzu::const_for_loop<0, int(STATES_PER_LANE), 1>([&](auto slot) UZU_ALWAYS_INLINE {
      constexpr uint S = uint(decltype(slot)::value);
      // One hash per state, then the table-free SWAR level map: four int8
      // weights whose single scale rides on the row scale, so there is no
      // per-weight multiply here at all.
      float weights[Tile::ROWS_PER_LANE][V];
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const char4 levels = as_type<char4>(
            uzu::trellis::map_bytes(uzu::trellis::state_hash(states[R][S], trellis.hash_a, trellis.hash_b))
        );
        uzu::const_for_loop<0, int(V), 1>([&](auto c)
                                              UZU_ALWAYS_INLINE { weights[R][c.value] = float(levels[c.value]); });
      });
      Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
        constexpr uint I = decltype(input_index)::value;
        // Padding batch slots read the last real row rather than branching: the
        // decode above is already paid, so their arithmetic is free and only the
        // store is guarded.
        const uint source_row = FULL_TILE ? I : min(I, batch_remaining - 1);
        const device AT* input = ops.a + (tile.input_row + source_row) * params.in_vec_size + column + S * V;
        const vec<AT, V> input_values = *reinterpret_cast<const device vec<AT, V>*>(input);
        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          uzu::const_for_loop<0, int(V), 1>([&](auto c) UZU_ALWAYS_INLINE {
            partial[I][R] = fma(weights[R][c.value], float(input_values[c.value]), partial[I][R]);
          });
        });
      });
    });
  }
};

} // namespace gemm
} // namespace uzu
