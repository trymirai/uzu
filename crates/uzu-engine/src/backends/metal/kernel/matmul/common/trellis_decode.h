#pragma once

// QTIP bitshift-trellis weight decode: tape -> state -> four int8 weights.
//
// `backends/common/kernel/matmul/trellis_format.rs` is the NORMATIVE definition
// of the format -- the recurrence, packing v2, the codebook -- and the oracle
// these kernels are tested against.  What matters on this side is the one
// consequence packing v2 exists for: `s_t` is one L-bit window of the row, at
// the offset `Walk` below spells out, so a state costs one shift and one mask
// at a computed offset.  A gather, not a scan: no per-step reassembly and no
// special case for the first steps of a tape.
//
// THE CODEBOOK is one hash per STATE whose four RAW BYTES become the state's
// four weights through a byte-separable SWAR chain, so a step costs one hash
// plus ~14 whole-word ALU ops and lands directly in int8 -- no table, no
// threadgroup codebook, no float dequantization on the way into the MXU.  The
// alternative that reads a 256-entry Gaussian-quantile table per byte gives the
// same marginals for fewer ALU ops but four threadgroup gathers, and measured
// 6.5-13.4% SLOWER at every M: a gather on Apple prices out at ~3.5 ALU ops.
//
// Consumed by `gemm/common/trellis_loader.h` and `gemv/common/trellis_slice.h`.

#include <metal_stdlib>

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace trellis {

/// Weights produced per trellis step. Fixed: a 32-bit hash has four bytes.
UZU_CONST uint TRELLIS_V = 4;

/// Widest window this header is sized for.
UZU_CONST uint TRELLIS_MAX_L = 32;

/// `L` is a runtime value everywhere below, so callers hoist its mask once.
METAL_FUNC uint state_mask(uint l) { return (l >= TRELLIS_MAX_L) ? 0xFFFFFFFFu : ((1u << l) - 1u); }

// ---------------------------------------------------------------------------
// state -> weights
// ---------------------------------------------------------------------------

/// `state -> ` the raw 32-bit hash: a 32-bit LCG followed by one fmix32
/// avalanche round.
///
/// The avalanche is what makes the 2**KV successors of a state spread out in
/// R^V; a bare LCG makes consecutive states nearly collinear.  Both halves of
/// the round are load bearing: dropping the trailing xorshift saves two ops and
/// still clears the lag-correlation gate at L=16 V=2 k=3, but fails it at V=4
/// k=3, where three of the four coordinates land at 0.049..0.072 against a 0.05
/// bar.
METAL_FUNC uint state_hash(uint s, uint a, uint b) {
  // One Murmur3 fmix32 round, verbatim from `qtip/codebooks.h`.
  uint x = s * a + b;
  x ^= x >> 16u;
  x *= 0x85EBCA6Bu;
  x ^= x >> 16u;
  return x;
}

/// The four 2-bit fields of each byte, summed, in 7 whole-word ops.
///
/// The standard SWAR pairwise tree, not four masked shifts summed: stage one
/// leaves `f0 + f1 <= 6` in each low nibble and `f2 + f3 <= 6` in each high one,
/// so `(s + (s >> 4)) & 0x0f0f0f0f` sums them at `<= 12` -- still a nibble, so
/// nothing carries out of a byte lane.  `qtip-research/REPORT_E8.md` section 1.
METAL_FUNC uint byte_pairs(uint x) {
  const uint s = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
  return (s + (s >> 4u)) & 0x0f0f0f0fu;
}

/// An already-hashed word -> the step's four int8 weights, packed
/// little-endian.  E8's k = 3 tier-A map, `w = 8 * pairs + ((3 * n0) & 15) - 54`
/// in 14 ops: 74 levels reaching 2.85 sigma.
///
/// No step can carry out of a byte lane, which is what makes the whole chain one
/// dependency chain of whole-word ops: `u <= 8 * 12 + 15 = 111`, so the `+ 0x4a`
/// bias tops out at 185.  `^ 0x80808080` is the `- 128` that turns the
/// offset-binary composite into its int8 level.  `trellis_format::
/// codebook_table` is the same map as a closed form, and the GPU parity tests
/// are what pin the two together.
METAL_FUNC uint map_bytes(uint x) {
  const uint p = byte_pairs(x);
  const uint d = ((x & 0x0f0f0f0fu) * 3u) & 0x0f0f0f0fu;
  return ((p << 3u) + d + 0x4a4a4a4au) ^ 0x80808080u;
}

// ---------------------------------------------------------------------------
// tape -> state
// ---------------------------------------------------------------------------

/// The L-bit window starting at `p` shifted by `shift` bits.  Two adjacent
/// 32-bit loads cover it for any shift, since L <= 32 leaves 64 - 31 = 33 valid
/// bits.  Reads one word past the window, so a row needs a word of slack past
/// its last bit; `TrellisConfig::row_stride_words` gives four.
///
/// Takes a word pointer and a shift rather than a bit offset because every
/// state a `TrellisCursor` thread decodes in one K group sits at the same bit
/// inside its word and differs only by a row count: it takes the pair from
/// [`Walk`] once per group and walks the pointer.
METAL_FUNC uint state_at_word(const device uint* p, uint shift, uint state_mask) {
  const ulong window = (ulong)p[0] | ((ulong)p[1] << 32u);
  return uint(window >> shift) & state_mask;
}

/// A lane's place on a tape row: the bit offset of its base state, and what
/// one K group of `group_steps` steps does to it.
///
/// A row is `tape_steps`-step tapes, `tape_bits` apart, each packed DESCENDING
/// (`trellis_format.rs`), so step `t`'s window starts at
///
/// ```text
/// (t / tape_steps) * tape_bits + (tape_steps - 1 - t % tape_steps) * KV
/// ```
///
/// One K group either covers a whole number of tapes -- then every group moves
/// the offset UP by the same `advance_bits` and the wrap never fires -- or is a
/// fraction of one, and then walks DOWN its tape by `group_steps * KV` bits and
/// jumps to the next tape every `wrap_period` groups. On lalamo's 64-column
/// tapes every kernel is in the first case; the whole-row tape is the second
/// with a wrap that never comes, which is what the fast path of the one-tape
/// format turned into: two selects per K group.
///
/// The advance is not a whole number of words in general (136 bits per GEMM
/// group on a `k = 2` tape), so `word()` and `shift()` are taken from `bit` per
/// K group: one shift and one mask per group, not per state.
struct Walk {
  /// Bit offset of the lane's base state from the start of the row.
  uint bit;
  /// What one K group adds to `bit`, as a wrapping `uint`.
  uint advance_bits;
  /// What the jump to the next tape adds on top, and every how many groups.
  uint wrap_bits;
  uint wrap_period;
  /// Groups already walked in the current tape; uniform across the
  /// threadgroup, which is what keeps the wrap select uniform.
  uint groups_in_tape;

  METAL_FUNC uint word() const thread { return bit >> 5u; }
  METAL_FUNC uint shift() const thread { return bit & 31u; }

  METAL_FUNC void advance() thread {
    groups_in_tape += 1u;
    const bool wrap = groups_in_tape == wrap_period;
    bit += advance_bits + (wrap ? wrap_bits : 0u);
    groups_in_tape = wrap ? 0u : groups_in_tape;
  }
};

/// The walk for a lane whose base state is step `group_first_step + lane_step`
/// of a row and whose K group is `group_steps` steps. `group_first_step` is the
/// group's first step, the same for every lane of the threadgroup;
/// `trellis_format::TrellisConfig::is_valid` keeps a tape and a group
/// commensurate, so the group's position inside its tape is what seeds the
/// wrap counter.
METAL_FUNC Walk
walk(uint group_first_step, uint lane_step, uint group_steps, uint kv, uint tape_steps, uint tape_bits) {
  const uint base_step = group_first_step + lane_step;
  const uint tape = base_step / tape_steps;
  const uint position = base_step - tape * tape_steps;
  Walk w;
  w.bit = tape * tape_bits + (tape_steps - 1u - position) * kv;
  if (group_steps >= tape_steps) {
    w.advance_bits = (group_steps / tape_steps) * tape_bits;
    w.wrap_bits = 0u;
    w.wrap_period = 0xFFFFFFFFu;
    w.groups_in_tape = 0u;
  } else {
    // Down the tape, as a wrapping subtraction; the wrap adds the tape back
    // and lands on the next one.
    w.advance_bits = 0u - group_steps * kv;
    w.wrap_bits = tape_bits + tape_steps * kv;
    w.wrap_period = tape_steps / group_steps;
    w.groups_in_tape = (group_first_step % tape_steps) / group_steps;
  }
  return w;
}

} // namespace trellis
} // namespace uzu
