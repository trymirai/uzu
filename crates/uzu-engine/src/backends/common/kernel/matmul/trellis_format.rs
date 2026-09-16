//! QTIP bitshift-trellis weight format: host-side packing and the decode
//! oracle.
//!
//! This module is the definition of the format *and* the oracle the kernels are
//! tested against; the kernels never call it. The device half is
//! `metal/kernel/matmul/common/trellis_decode.h`, and the GEMM that consumes it
//! is the [`GemmBPrologueKind::Trellis`] arm of `gemm.metal`.
//!
//! [`GemmBPrologueKind::Trellis`]: crate::backends::common::gpu_types::gemm::GemmBPrologueKind::Trellis
//!
//! # Format
//!
//! A tape is `T` trellis steps. With window `L`, [`TRELLIS_V`] weights per step
//! and `k` bits per weight (`KV = k*V` bits injected per step):
//!
//! ```text
//! s_0 = header
//! s_t = ((s_{t-1} << KV) mod 2**L) | c_t
//! w[row][t*V + v] = codebook(s_t)[v] * row_scale[row]
//! ```
//!
//! Bit `j` of a row lives in byte `j >> 3` at bit `j & 7`, LSB first; a
//! width-`n` field at bit `p` is `sum_i bit(p+i) << i`. Layout (packing **v2**,
//! codes in DESCENDING step order with the header above them):
//!
//! ```text
//! bits [(T-1-t)*KV, (T-t)*KV)     c_t,  t = 1 .. T-1
//! bits [(T-1)*KV, (T-1)*KV + L)   s_0
//! ```
//!
//! The payoff is that `s_t` is then literally one L-bit LSB-first field:
//!
//! ```text
//! s_t = (tape_as_integer >> ((T-1-t)*KV)) & ((1 << L) - 1)
//! ```
//!
//! so a kernel reads a state with one shift and one mask, with no block
//! reassembly and no special case for the first steps of a tape. See the
//! "WHY DESCENDING" section of `qtip/oracle.py` for the derivation: it is
//! forced, given that `s_t`'s low `KV` bits are `c_t` and the next `KV` are
//! `c_{t-1}`.
//!
//! A weight row is `cols / restart_columns` such tapes, each of
//! `T = restart_columns / V` steps, bit-concatenated and nothing between them:
//! tape `b` starts at bit `b * (L + (T-1)*KV)`. Without a restart the row is one
//! tape of `cols / V` steps, and every formula above holds with `T = steps`.
//! Step `t` of a row is therefore the window at
//!
//! ```text
//! (t / T) * (L + (T-1)*KV) + (T-1 - t % T) * KV
//! ```
//!
//! which is what [`TrellisConfig::window_bit_offset`] and the device `Walk`
//! both implement. Every tape restarts the recurrence from its own `s_0`, and
//! that is the point: the fitter buys a free choice of path at every tape start
//! for `L - KV` bits per tape.
//!
//! Rate is `k + (L - KV) / restart_columns` bits per weight; the `s_0` header is
//! the overhead term. Dropping it and seeding the window with zero is a real
//! bug — every recovered state would disagree with the fitted stream — which is
//! why it is stored explicitly.
//!
//! # The codebook
//!
//! One hash per STATE, whose four raw bytes become four int8 weights through a
//! byte-separable map: a quarter of the hashes a per-coordinate codebook would
//! need, no table, and no dequantization on the way into the MXU. The map is
//! E8's k = 3 tier A, `8 * pairs(b) + ((3 * (b & 15)) & 15) - 54`, written here
//! as the CLOSED FORM it was fitted as — the device computes the same values
//! with a SWAR chain, and the GPU parity tests are what pin the two together.
use std::sync::LazyLock;

use crate::backends::common::gpu_types::TrellisParams;

/// The seed the shipped configs are fitted with.
const DEFAULT_SEED: u64 = 1234;

/// Weights produced per trellis step. Fixed: a 32-bit hash has four bytes.
pub const TRELLIS_V: u32 = 4;

/// Columns of K one threadgroup decodes into shared memory per iteration.
///
/// The GEMM reports this as the weight "group size" so that the shipped integer
/// schedule, its split-K policy and `GemmParams::aligned_inner_iterations` — all
/// of which are written against `outer_block_k() == GROUP_SIZE` — apply to a
/// trellis tape unchanged. There is no group of weights sharing a scale here;
/// there is one scale per row and this is a staging block.
pub const TRELLIS_BLOCK_K: u32 = 64;

/// Columns of the narrowest K unit a kernel walks: one GEMV lane's run of
/// `TrellisSlice::STATES_PER_LANE` states. A tape is whole runs, so a run never
/// straddles a header.
pub const TRELLIS_TAPE_UNIT: u32 = 16;

/// Columns of the widest K group a kernel walks: the 32-lane GEMV's block,
/// `gemv::policy::trellis_k_block(32)`. The GEMM group ([`TRELLIS_BLOCK_K`])
/// and the eight-lane GEMV block (128) sit between this and
/// [`TRELLIS_TAPE_UNIT`], all powers of two, so a tape that divides this or is
/// whole multiples of it nests with every group the dispatch can pick: a group
/// is whole tapes, or a tape is whole groups.
pub const TRELLIS_WIDEST_K_GROUP: u32 = 512;

/// The splitmix64 finalizer (Steele/Lea), verbatim from `qtip/codebooks.h`.
fn splitmix64(
    x: u64,
    seed: u64,
) -> u64 {
    let mut z = x.wrapping_add(seed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `(a, b)` 32-bit LCG pair the state hash uses, `a` forced odd.
///
/// `codebooks.py` derives one pair per coordinate from the seed; this codebook
/// takes one hash per STATE, so only coordinate 0's pair exists and it is a
/// constant of the seed rather than of the config.
pub fn hash_params() -> (u32, u32) {
    ((splitmix64(0, DEFAULT_SEED) as u32) | 1, splitmix64(1, DEFAULT_SEED) as u32)
}

/// A `(L, k)` trellis configuration and how a row is cut into tapes. `V` is
/// [`TRELLIS_V`] everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrellisConfig {
    /// Window width in bits; the state is `L` bits wide, `1 <= L <= 32`.
    pub l: u32,
    /// Bits per weight; `k * V` bits are injected per step.
    pub k: u32,
    /// Columns per tape, or `None` for one tape spanning the row. See the
    /// module doc for the layout; the same field on lalamo's `TrellisSpec`
    /// writes it.
    ///
    /// The kernels walk K in groups of their own, so a tape has to nest with
    /// those: whole [`TRELLIS_TAPE_UNIT`]s, and either a divisor of
    /// [`TRELLIS_WIDEST_K_GROUP`] or whole multiples of it.
    pub restart_columns: Option<u32>,
}

impl TrellisConfig {
    pub const fn new(
        l: u32,
        k: u32,
    ) -> Self {
        Self {
            l,
            k,
            restart_columns: None,
        }
    }

    pub const fn with_restart(
        self,
        columns: u32,
    ) -> Self {
        Self {
            restart_columns: Some(columns),
            ..self
        }
    }

    /// Bits injected per step.
    pub const fn kv(&self) -> u32 {
        self.k * TRELLIS_V
    }

    const fn state_mask(&self) -> u32 {
        if self.l >= 32 {
            u32::MAX
        } else {
            (1u32 << self.l) - 1
        }
    }

    const fn is_valid(&self) -> bool {
        let restart_ok = match self.restart_columns {
            None => true,
            Some(columns) => {
                columns >= TRELLIS_TAPE_UNIT
                    && columns.is_multiple_of(TRELLIS_TAPE_UNIT)
                    && (TRELLIS_WIDEST_K_GROUP.is_multiple_of(columns)
                        || columns.is_multiple_of(TRELLIS_WIDEST_K_GROUP))
            },
        };
        1 <= self.l && self.l <= 32 && self.k >= 1 && self.kv() <= self.l && restart_ok
    }

    /// Whether a `cols`-column row is a whole number of tapes.
    pub const fn divides(
        &self,
        cols: u32,
    ) -> bool {
        match self.restart_columns {
            None => cols.is_multiple_of(TRELLIS_V),
            Some(columns) => cols.is_multiple_of(columns),
        }
    }

    pub const fn steps(
        &self,
        cols: u32,
    ) -> u32 {
        cols / TRELLIS_V
    }

    /// Steps in one tape: the whole row without a restart.
    pub const fn tape_steps(
        &self,
        cols: u32,
    ) -> u32 {
        match self.restart_columns {
            None => self.steps(cols),
            Some(columns) => columns / TRELLIS_V,
        }
    }

    pub const fn tapes(
        &self,
        cols: u32,
    ) -> u32 {
        match self.restart_columns {
            None => 1,
            Some(columns) => cols / columns,
        }
    }

    /// Bits of one tape: its header and the codes of every later step.
    pub const fn tape_bits(
        &self,
        cols: u32,
    ) -> u32 {
        self.l + (self.tape_steps(cols) - 1) * self.kv()
    }

    pub const fn bits_per_row(
        &self,
        cols: u32,
    ) -> u32 {
        self.tapes(cols) * self.tape_bits(cols)
    }

    pub const fn bytes_per_row(
        &self,
        cols: u32,
    ) -> u32 {
        self.bits_per_row(cols).div_ceil(8)
    }

    /// `u32`s per tape row. The window read is a two-word load at a bit offset
    /// inside the row, so a row needs one word of slack past its last bit;
    /// four keeps every row 16-byte aligned as well. `trellis_decode.h`'s
    /// `row_stride_words` is not a second copy of this — the host computes it
    /// once and passes it down in `TrellisParams`.
    pub const fn row_stride_words(
        &self,
        cols: u32,
    ) -> u32 {
        self.bits_per_row(cols).div_ceil(32) + 4
    }

    /// Bit offset of the L-bit window that spells out `s_t` (packing v2): the
    /// tape holding step `t`, then the descending position inside it.
    pub const fn window_bit_offset(
        &self,
        t: u32,
        cols: u32,
    ) -> u32 {
        let tape_steps = self.tape_steps(cols);
        (t / tape_steps) * self.tape_bits(cols) + (tape_steps - 1 - t % tape_steps) * self.kv()
    }
}

// ---------------------------------------------------------------------------
// the codebook: state -> four int8 weights
// ---------------------------------------------------------------------------

/// `state -> ` the raw 32-bit hash: a 32-bit LCG followed by one fmix32
/// avalanche round.
pub fn state_hash(
    state: u32,
    a: u32,
    b: u32,
) -> u32 {
    let mut x = state.wrapping_mul(a).wrapping_add(b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^ (x >> 16)
}

/// The induced 256-entry `byte -> int8` map, as E8's closed form rather than as
/// the SWAR chain `trellis_decode.h` computes it with.
///
/// `pairs` is the sum of the byte's four 2-bit fields; 74 levels over
/// `[-54, 55]`, reaching 2.85 sigma.
pub fn codebook_table() -> Vec<i8> {
    (0..256u32)
        .map(|b| {
            let pairs = (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3);
            let level = (8 * pairs + (((b & 15) * 3) & 15)) as i32 - 54;
            i8::try_from(level).expect("trellis level must fit int8")
        })
        .collect()
}

/// `1 / rms(codebook_table())` — the scalar folded into the per-row weight
/// scale, which makes the decoded codebook unit variance. The four hash bytes
/// are uniform and independent after the fmix, so the codebook marginal IS the
/// induced table's.
pub fn codebook_scale() -> f32 {
    static SCALE: LazyLock<f32> = LazyLock::new(|| {
        let table = codebook_table();
        let mean_square: f64 = table.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / table.len() as f64;
        (1.0 / mean_square.sqrt()) as f32
    });
    *SCALE
}

/// `state -> ` the four int8 codes the kernel stages, in column order.
pub fn state_codes(
    state: u32,
    a: u32,
    b: u32,
) -> [i8; 4] {
    let table = codebook_table();
    let hash = state_hash(state, a, b);
    [0u32, 1, 2, 3].map(|j| table[((hash >> (8 * j)) & 255) as usize])
}

// ---------------------------------------------------------------------------
// bit tape
// ---------------------------------------------------------------------------

/// A row-major bit tape: `rows` rows of `config.row_stride_words(cols)` `u32`s.
/// Little-endian word order makes `words` byte-identical to the reference
/// `uint8` tape for the first `bytes_per_row` bytes of each row; the remainder
/// is zero padding.
#[derive(Debug, Clone)]
pub struct TrellisTape {
    pub config: TrellisConfig,
    pub rows: u32,
    pub cols: u32,
    pub words: Vec<u32>,
}

#[inline]
fn write_field(
    row: &mut [u32],
    bit_offset: u32,
    width: u32,
    value: u32,
) {
    debug_assert!(width <= 32);
    let masked = if width == 32 {
        value
    } else {
        value & ((1u32 << width) - 1)
    };
    let word = (bit_offset >> 5) as usize;
    let shift = bit_offset & 31;
    row[word] |= masked << shift;
    if shift + width > 32 {
        row[word + 1] |= masked >> (32 - shift);
    }
}

#[inline]
fn read_field(
    row: &[u32],
    bit_offset: u32,
    width: u32,
) -> u32 {
    debug_assert!(width <= 32);
    let word = (bit_offset >> 5) as usize;
    let shift = bit_offset & 31;
    let low = u64::from(row[word]) | (u64::from(row[word + 1]) << 32);
    let value = (low >> shift) as u32;
    if width == 32 {
        value
    } else {
        value & ((1u32 << width) - 1)
    }
}

impl TrellisTape {
    /// A deterministic pseudo-random tape, packed the way `qtip/oracle.py`
    /// packs a fitted one (a fitted tape's statistics are irrelevant to kernel
    /// correctness or to decode cost).
    pub fn random(
        config: TrellisConfig,
        rows: u32,
        cols: u32,
        seed: u64,
    ) -> Self {
        assert!(config.is_valid(), "invalid trellis config {config:?}");
        assert!(config.divides(cols), "cols must be a whole number of tapes");
        let steps = config.steps(cols);
        let tape_steps = config.tape_steps(cols);
        let stride = config.row_stride_words(cols);
        let kv = config.kv();
        let code_mask = (1u64 << kv) - 1;
        let state_mask = u64::from(config.state_mask());
        let mut state = seed | 1;
        let mut next = move || {
            state = splitmix64(state, 0x9E37_79B9_7F4A_7C15);
            state
        };

        let mut words = vec![0u32; (rows * stride) as usize];
        for row in 0..rows as usize {
            let destination = &mut words[row * stride as usize..(row + 1) * stride as usize];
            for step in 0..steps {
                let offset = config.window_bit_offset(step, cols);
                if step.is_multiple_of(tape_steps) {
                    write_field(destination, offset, config.l, (next() & state_mask) as u32);
                } else {
                    write_field(destination, offset, kv, (next() & code_mask) as u32);
                }
            }
        }
        Self {
            config,
            rows,
            cols,
            words,
        }
    }

    /// A tape from rows as a fitter writes them: `bytes_per_row` bytes each,
    /// back to back, nothing between rows. lalamo's `TrellisMatrix.export` is
    /// this, and here is the one place the kernels' `row_stride_words` padding
    /// is put in.
    pub fn from_packed_rows(
        config: TrellisConfig,
        rows: u32,
        cols: u32,
        bytes: &[u8],
    ) -> Self {
        assert!(config.is_valid(), "invalid trellis config {config:?}");
        assert!(config.divides(cols), "cols must be a whole number of tapes");
        let row_bytes = config.bytes_per_row(cols) as usize;
        assert_eq!(bytes.len(), rows as usize * row_bytes, "expected {rows} rows of {row_bytes} bytes");
        let stride = config.row_stride_words(cols) as usize;
        let mut words = vec![0u32; rows as usize * stride];
        for (row, source) in bytes.chunks(row_bytes).enumerate() {
            for (index, chunk) in source.chunks(4).enumerate() {
                let mut padded = [0u8; 4];
                padded[..chunk.len()].copy_from_slice(chunk);
                words[row * stride + index] = u32::from_le_bytes(padded);
            }
        }
        Self {
            config,
            rows,
            cols,
            words,
        }
    }

    fn row(
        &self,
        row: u32,
    ) -> &[u32] {
        let stride = self.config.row_stride_words(self.cols) as usize;
        &self.words[row as usize * stride..(row as usize + 1) * stride]
    }

    /// `[rows][steps]` states, one window read per step — the same access the
    /// kernels make (`oracle.py:unpack`).
    pub fn states(&self) -> Vec<u32> {
        let steps = self.config.steps(self.cols);
        let mut out = vec![0u32; (self.rows * steps) as usize];
        for row in 0..self.rows {
            let source = self.row(row);
            for step in 0..steps {
                out[(row * steps + step) as usize] =
                    read_field(source, self.config.window_bit_offset(step, self.cols), self.config.l);
            }
        }
        out
    }

    /// `[rows][steps]` states obtained by running the trellis recurrence over
    /// the unpacked code stream, restarted from the header of every tape.
    /// Independent of [`Self::states`], which reads windows; the two agreeing
    /// is what pins packing v2 down.
    pub fn states_by_recurrence(&self) -> Vec<u32> {
        let steps = self.config.steps(self.cols);
        let tape_steps = self.config.tape_steps(self.cols);
        let kv = self.config.kv();
        let mask = self.config.state_mask();
        let mut out = vec![0u32; (self.rows * steps) as usize];
        for row in 0..self.rows {
            let source = self.row(row);
            let mut state = 0u32;
            for step in 0..steps {
                let offset = self.config.window_bit_offset(step, self.cols);
                if step.is_multiple_of(tape_steps) {
                    state = read_field(source, offset, self.config.l);
                } else {
                    state = (state.wrapping_shl(kv) & mask) | read_field(source, offset, kv);
                }
                out[(row * steps + step) as usize] = state;
            }
        }
        out
    }

    /// `[rows][cols]` int8 codes — the weights in the basis the MXU sees, before
    /// `row_scale * codebook_scale()`.
    pub fn codes(&self) -> Vec<i8> {
        let (a, b) = hash_params();
        let table = codebook_table();
        let states = self.states();
        let steps = self.config.steps(self.cols) as usize;
        let cols = self.cols as usize;
        let mut out = vec![0i8; (self.rows * self.cols) as usize];
        for row in 0..self.rows as usize {
            for step in 0..steps {
                let hash = state_hash(states[row * steps + step], a, b);
                for j in 0..4 {
                    out[row * cols + step * 4 + j] = table[((hash >> (8 * j)) & 255) as usize];
                }
            }
        }
        out
    }
}

/// The device-side constants for `config` over `k`-column rows, or `None` if
/// `config` is not one this format can express or `k` is not whole tapes.
///
/// Both kernel paths build their `TrellisParams` here, so the tape stride, the
/// hash pair and the codebook scale have exactly one definition and no caller
/// can pass a set that disagrees with the tape it points at.
pub fn trellis_params(
    config: TrellisConfig,
    k: u32,
) -> Option<TrellisParams> {
    if !config.is_valid() || !config.divides(k) {
        return None;
    }
    let (hash_a, hash_b) = hash_params();
    Some(TrellisParams {
        l: config.l,
        k_bits_per_step: config.kv(),
        row_stride_words: config.row_stride_words(k),
        hash_a,
        hash_b,
        codebook_scale: codebook_scale(),
        tape_steps: config.tape_steps(k),
        tape_bits: config.tape_bits(k),
    })
}
