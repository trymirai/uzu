#pragma once

// QTIP bitshift-trellis weight decode: tape -> state -> four int8 weights.
//
// `backends/common/kernel/matmul/trellis_format.rs` is the NORMATIVE definition
// of the format -- the recurrence, packing v2, the codebook -- and the oracle
// these kernels are tested against.  What matters on this side is the one
// consequence packing v2 exists for: `s_t` is the L-bit window at bit offset
// `(T-1-t)*KV` of the row, so a state costs one shift and one mask at a
// computed offset.  A gather, not a scan: no per-step reassembly and no special
// case for the first steps of a row.
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
/// Takes a word pointer and a shift rather than a bit offset because one K block
/// of a staged tile advances the bit offset by a whole number of WORDS, so
/// consecutive blocks share `shift` and differ by a constant word count.
/// Callers hoist the shift with [`walk`] and walk the pointer instead of
/// recomputing `>>5` and `&31` per state.
METAL_FUNC uint state_at_word(const device uint* p, uint shift, uint state_mask) {
  const ulong window = (ulong)p[0] | ((ulong)p[1] << 32u);
  return uint(window >> shift) & state_mask;
}

/// Where a run of states starts inside a tape row, and how far one block of
/// `steps_per_block` steps moves it.
///
/// The tape DESCENDS -- `s_t` sits at bit `(steps - 1 - t) * KV` -- so a run is
/// based on its LAST step, and one block is always a whole number of 32-bit
/// words (V == 4), which is what makes `shift` loop-invariant.
struct Walk {
  /// Words from the start of the row to the word holding the run's first state.
  uint word_offset;
  /// Bit of that state inside that word.
  uint shift;
  /// Words one block of `steps_per_block` steps advances the row pointer by.
  uint block_words;
};

METAL_FUNC Walk walk(uint steps, uint last_step, uint steps_per_block, uint kv) {
  const uint bit_offset = (steps - 1u - last_step) * kv;
  return Walk{bit_offset >> 5u, bit_offset & 31u, (steps_per_block * kv) / 32u};
}

} // namespace trellis
} // namespace uzu
