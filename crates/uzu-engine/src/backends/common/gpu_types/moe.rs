//! Mixture-of-experts kernel types.

/// Expert weight encodings decoded by the mixture-of-experts kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeExpertFormat {
    /// E2M1 codes two per byte (low nibble first), one E4M3 scale per 16 inputs and one float32 global scale per
    /// expert.
    Nvfp4,
    /// Unsigned 4-bit codes two per byte (low nibble first) with one bfloat16 (scale, bias) pair per 64 inputs.
    AffineU4,
}

/// Largest number of experts a token selects.
pub const MOE_MAX_TOP_K: u32 = 8;
/// Largest number of routed experts the routing kernels hold, eight per lane of a simdgroup.
pub const MOE_MAX_EXPERTS: u32 = 256;
/// Largest batch the fused router and the GEMV expert kernels take; larger batches use the grouped GEMM path.
pub const MOE_GEMV_MAX_TOKENS: u32 = 32;
/// Simdgroups per threadgroup of the GEMV expert kernels; each walks its own share of the active experts.
pub const MOE_GEMV_SIMDGROUPS: u32 = 8;
/// Hidden units one GEMV up threadgroup produces per expert (each lane quartet of a simdgroup owns one).
pub const MOE_UP_TILE_HIDDEN: u32 = 4;
/// Output rows one GEMV down threadgroup produces.
pub const MOE_DOWN_TILE_ROWS: u32 = 16;
/// Experts whose router logits one routing threadgroup computes.
pub const MOE_ROUTE_EXPERTS_PER_GROUP: u32 = 4;
/// Threads of the single threadgroup that routes a large batch.
pub const MOE_ROUTE_FINALIZE_THREADS: u32 = 1024;
/// Bits of an active-expert entry holding the expert; the next MOE_ACTIVE_COUNT_BITS hold its route count and the
/// rest its first route, so that the expert kernels read one word before streaming weights. The first route fits for
/// batches of up to 1024 tokens with 8 picks; only the small-batch expert kernels, at most MOE_GEMV_MAX_TOKENS tokens,
/// read it.
pub const MOE_ACTIVE_EXPERT_BITS: u32 = 8;
pub const MOE_ACTIVE_COUNT_BITS: u32 = 11;

/// An entry of the active-expert table: `expert | count << 8 | first_route << 19`.
pub fn moe_active_entry(
    expert: u32,
    count: u32,
    first_route: u32,
) -> u32 {
    debug_assert!(expert < (1 << MOE_ACTIVE_EXPERT_BITS) && count < (1 << MOE_ACTIVE_COUNT_BITS));
    expert
        | (count << MOE_ACTIVE_EXPERT_BITS)
        | first_route.wrapping_shl(MOE_ACTIVE_EXPERT_BITS + MOE_ACTIVE_COUNT_BITS)
}

/// The (expert, count, first route) of an active-expert entry.
pub fn moe_active_entry_parts(entry: u32) -> (u32, u32, u32) {
    (
        entry & ((1 << MOE_ACTIVE_EXPERT_BITS) - 1),
        (entry >> MOE_ACTIVE_EXPERT_BITS) & ((1 << MOE_ACTIVE_COUNT_BITS) - 1),
        entry >> (MOE_ACTIVE_EXPERT_BITS + MOE_ACTIVE_COUNT_BITS),
    )
}

/// Bit positions (magnitude, sign) of the four codes in each 16-bit half of a repacked NVFP4 word.
fn nvfp4_lane_fields(slot: u32) -> (u32, u32) {
    match slot {
        0 => (9, 15),
        1 => (6, 12),
        2 => (3, 13),
        3 => (0, 14),
        _ => unreachable!("a 16-bit half holds four codes"),
    }
}

/// Repacks one little-endian word of eight NVFP4 codes (code i in bits [4i, 4i + 4), sign in the top bit) for the
/// expert kernels. Code j of each 16-bit half keeps its half but moves its three magnitude bits and its sign to the
/// positions [`nvfp4_lane_fields`] gives, so that the kernels place codes j and j + 4 into the magnitude bits [9, 12)
/// and the sign bit of a half2 with at most two shifts.
pub fn nvfp4_repack_word(word: u32) -> u32 {
    let lane = |codes: u32| {
        (0..4).fold(0u32, |packed, slot| {
            let code = (codes >> (4 * slot)) & 0xF;
            let (magnitude_bit, sign_bit) = nvfp4_lane_fields(slot);
            packed | ((code & 0x7) << magnitude_bit) | ((code >> 3) << sign_bit)
        })
    };
    lane(word & 0xFFFF) | (lane(word >> 16) << 16)
}

/// [`nvfp4_repack_word`] on both 16-bit halves at once, without loops: what the loader runs on every word.
pub fn nvfp4_repack_word_fast(word: u32) -> u32 {
    ((word & 0x0007_0007) << 9)
        | ((word & 0x0008_0008) << 12)
        | (((word >> 4) & 0x0007_0007) << 6)
        | ((word & 0x0080_0080) << 5)
        | (((word >> 8) & 0x0007_0007) << 3)
        | ((word & 0x0800_0800) << 2)
        | ((word >> 12) & 0x0007_0007)
        | ((word & 0x8000_8000) >> 1)
}

/// Code `index` (0..8) of a word that [`nvfp4_repack_word`] produced.
pub fn nvfp4_repacked_code(
    word: u32,
    index: u32,
) -> u8 {
    let lane = (word >> (16 * (index / 4))) & 0xFFFF;
    let (magnitude_bit, sign_bit) = nvfp4_lane_fields(index % 4);
    (((lane >> magnitude_bit) & 0x7) | (((lane >> sign_bit) & 1) << 3)) as u8
}
