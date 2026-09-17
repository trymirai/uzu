#pragma once

#include <metal_stdlib>

#include "../../../common/thread_context.h"
#include "../../../generated/matmul.h"
#include "../../common/fragment.h"
#include "../../common/trellis_decode.h"
#include "operands.h"
#include "quantized/cursor.h"
#include "schedules/tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace quantized {

/// The right-hand operand cursor for a trellis tape.
///
/// Every other weight scheme reads codes straight out of device memory, so its
/// cursor is a pointer walk. A tape has no addressable codes: a weight only
/// exists once its trellis state has been hashed. So this cursor DECODES a
/// `THREADGROUP_BLOCK_N x THREADGROUP_BLOCK_K` int8 block into threadgroup
/// memory once per K group, and the K loop then reads MMA fragments out of that
/// block exactly as it would from device memory. `THREADGROUP_BLOCK_K` is the
/// `RightOperand::GROUP_SIZE` the integer schedule already steps in, so the
/// schedule's loop structure, its `aligned_inner_iterations` and the shipped
/// split-K policy all apply unchanged.
///
/// ADDRESS GENERATION is the thing to get right; a naive version recomputes a
/// 64-bit `row * row_stride_words` product, a `>> 5` and a `& 31` per state and
/// shows up as an address-generation limiter. Two facts collapse it to one
/// pointer add per state:
///
///   * `THREADS` is a multiple of `STEPS_PER_BLOCK`, so a thread's `SLOTS`
///     states all sit at the SAME trellis step and differ only by weight ROW.
///     One shift and one word offset serve all of them, and consecutive slots
///     are a fixed number of tape words apart.
///   * one K group advances the step by `STEPS_PER_BLOCK`, i.e. the bit offset
///     falls by `THREADGROUP_BLOCK_K * k` bits, which for `V == 4` is always a
///     whole number of 32-bit words. `bit_offset & 31` is therefore
///     loop-invariant and `bit_offset >> 5` is an induction variable.
///
/// so `stage()` never recomputes a shift, a mask, the out-of-range row test or
/// the threadgroup destination.
///
/// N RAGGEDNESS is handled by the `live` mask rather than by the fragment load:
/// weight rows past `N` stage an exact zero, so the MMA needs no N guard and the
/// fragment read is always the unbounded one. `live` is derived from `params->N`
/// alone, so it is the same in every simdgroup of the threadgroup — which
/// matters, because the two `ALIGNED_N` instantiations of the schedule must
/// agree on how many barriers they execute.
template <typename Core>
struct TrellisCursor {
  using Ops = typename Core::FragmentOps;
  using Fragment = uzu::matmul::Fragment<int8_t, Core::TILES_N, Core::TILES_K, Ops, uzu::matmul::ReadDirect, true>;

  UZU_CONST ushort BLOCK_N = Core::THREADGROUP_BLOCK_N;
  UZU_CONST ushort BLOCK_K = Core::THREADGROUP_BLOCK_K;
  UZU_CONST ushort THREADS = Core::THREADGROUP_THREADS;
  UZU_CONST ushort STEPS_PER_BLOCK = BLOCK_K / ushort(uzu::trellis::TRELLIS_V);
  UZU_CONST ushort SLOTS = (BLOCK_N * STEPS_PER_BLOCK) / THREADS;
  /// Words per staged weight row. `SHARED_STRIDE_B` already pads the row against
  /// threadgroup bank conflicts; the decode needs it to be a multiple of four so
  /// its 32-bit stores stay aligned.
  UZU_CONST ushort ROW_WORDS = Core::SHARED_STRIDE_B / 4;
  /// Weight rows between two of this thread's slots.
  UZU_CONST ushort SLOT_ROWS = THREADS / STEPS_PER_BLOCK;
  UZU_CONST ushort SLOT_WORDS = SLOT_ROWS * ROW_WORDS;
  UZU_CONST uint ALL_LIVE = (SLOTS >= 32) ? 0xFFFFFFFFu : ((1u << SLOTS) - 1u);

  static_assert(SLOTS * THREADS == BLOCK_N * STEPS_PER_BLOCK, "weight block must divide evenly across the threadgroup");
  static_assert(THREADS % STEPS_PER_BLOCK == 0, "a thread's slots must share a trellis step");
  static_assert(SLOTS <= 32, "the live-row mask is 32 bits");
  static_assert(Core::SHARED_STRIDE_B % 4 == 0, "the decode stores whole words into the staged row");

  /// The staged block, as words. Its allocation is `uint`-typed for exactly this
  /// reason; see `operands::stage_block`.
  threadgroup uint* block;
  /// Tape word holding the low bits of slot 0's state for the current K group.
  const device uint* word;
  /// Bit position of the state inside `word` -- loop-invariant, see above.
  uint shift;
  /// Tape words between two slots, and between two K groups.
  uint slot_words;
  uint words_per_group;
  uint state_mask;
  uint hash_a;
  uint hash_b;
  /// Threadgroup word index of slot 0.
  ushort destination;
  /// Bit `i` is set when slot `i`'s weight row is inside `N`.
  uint live;
  /// This simdgroup's read window into the staged block.
  uzu::matmul::FragmentSource<threadgroup int8_t*> source;
  ushort simd_lane_id;

  template <typename Storage>
  static METAL_FUNC TrellisCursor make(
      const Storage right,
      threadgroup int8_t* shared,
      const constant uzu::matmul::GemmParams* params,
      const schedules::TileContext tile,
      const thread ThreadContext& thread_context
  ) {
    const uint kv = right.trellis->k_bits_per_step;
    const uint row_stride_words = right.trellis->row_stride_words;
    const uint steps = uint(params->K) / uint(uzu::trellis::TRELLIS_V);
    const uint thread_index = thread_context.simdgroup_index * METAL_SIMD_SIZE + thread_context.simd_lane_id;
    const ushort step = ushort(thread_index % STEPS_PER_BLOCK);
    const ushort slot_row = ushort(thread_index / STEPS_PER_BLOCK);
    const uzu::trellis::Walk tape =
        uzu::trellis::walk(steps, tile.k_offset / uint(uzu::trellis::TRELLIS_V) + uint(step), STEPS_PER_BLOCK, kv);
    const uint block_row_base = uint(tile.block_col);

    TrellisCursor cursor;
    cursor.block = reinterpret_cast<threadgroup uint*>(shared);
    cursor.word = right.tape + (ulong)(block_row_base + slot_row) * (ulong)row_stride_words + (ulong)tape.word_offset;
    cursor.shift = tape.shift;
    cursor.slot_words = uint(SLOT_ROWS) * row_stride_words;
    cursor.words_per_group = tape.block_words;
    cursor.state_mask = uzu::trellis::state_mask(right.trellis->l);
    cursor.hash_a = right.trellis->hash_a;
    cursor.hash_b = right.trellis->hash_b;
    cursor.destination = slot_row * ROW_WORDS + step;
    cursor.live = 0u;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < SLOTS; ++i) {
      if (block_row_base + slot_row + i * SLOT_ROWS < uint(params->N)) {
        cursor.live |= 1u << i;
      }
    }
    cursor.source =
        uzu::matmul::fragment_source(shared + tile.tile_col_offset * Core::SHARED_STRIDE_B, int(Core::SHARED_STRIDE_B));
    cursor.simd_lane_id = ushort(thread_context.simd_lane_id);
    return cursor;
  }

  /// Decode this thread's `SLOTS` states into the staged block and step to the
  /// next K group. The barrier pair is the block's, and it is why this is a
  /// method rather than something the schedule open-codes: both `ALIGNED_N`
  /// instantiations must run it the same number of times.
  METAL_FUNC void stage() thread {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The N guard, hoisted out of the unroll. `live` does not change across K
    // groups, so the common case -- every slot inside N -- can run the whole
    // unroll with NO per-slot conditional at all, and only the one ragged N
    // block per grid pays a test per slot. `integer_and_conditional` is this
    // kernel's second limiter and SLOTS of it per K group were conditionals.
    if (live == ALL_LIVE) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < SLOTS; ++i) {
        const uint state = uzu::trellis::state_at_word(word + i * slot_words, shift, state_mask);
        block[destination + i * SLOT_WORDS] = uzu::trellis::map_bytes(uzu::trellis::state_hash(state, hash_a, hash_b));
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < SLOTS; ++i) {
        uint packed = 0u;
        if ((live >> i) & 1u) {
          const uint state = uzu::trellis::state_at_word(word + i * slot_words, shift, state_mask);
          packed = uzu::trellis::map_bytes(uzu::trellis::state_hash(state, hash_a, hash_b));
        }
        block[destination + i * SLOT_WORDS] = packed;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    word -= words_per_group;
  }

  METAL_FUNC Fragment load(const uint chunk_index) const thread {
    Fragment tile;
    tile.load_from(simd_lane_id, source.advanced(int(chunk_index) * int(Core::SIMDGROUP_BLOCK_K)));
    return tile;
  }

  /// The staged block is re-decoded per K group and read by chunk index, so
  /// there is no pointer to walk.
  METAL_FUNC void advance() thread {}

  METAL_FUNC void begin_k_group(const uint) thread {}
};

/// The right-hand cursor for whichever weight scheme is compiled: a tape decoder
/// for `Trellis`, the ordinary device-memory code cursor otherwise.
template <bool HOIST, typename Core, typename RightOperand, bool ALIGNED_N, typename Storage>
static METAL_FUNC auto make_right_cursor(
    const Storage right,
    threadgroup typename Core::RightElementType* shared,
    const constant uzu::matmul::GemmParams* params,
    const schedules::TileContext tile,
    const thread ThreadContext& thread_context
) {
  if constexpr (RightOperand::STAGED) {
    return TrellisCursor<Core>::make(right, shared, params, tile, thread_context);
  } else {
    return make_cursor<Axis::Columns, HOIST, Core, typename RightOperand::Format, ALIGNED_N>(
        right,
        params,
        tile,
        thread_context
    );
  }
}

} // namespace quantized
} // namespace gemm
} // namespace uzu
