//! Cross-checks of the Rust trellis format port against the Python oracle.
//!
//! Three things need pinning, and they are pinned separately because they can
//! drift separately: the TAPE LAYOUT (packing v2 — a change here makes uzu read
//! an exported model's bytes wrong), the STATE HASH (the codebook was fitted
//! with it), and the LEVEL MAP (a change here silently stops the codebook being
//! unit variance).

use uzu_engine_macros::uzu_test;

use crate::backends::common::kernel::matmul::trellis_format::*;

/// One row of `mad1_golden.json`, dumped from `qtip/oracle.py` by
/// `tools/qtip_dump_golden.py` at the shipped `L = 32, V = 4, k = 3`: the packed
/// tape bytes and the 17 states Python reads back out of them.
///
/// Kept as bytes rather than as a re-packing, because what needs pinning is that
/// uzu's READ of an exported tape agrees with Python's WRITE of it. uzu never
/// writes a production tape.
const GOLDEN_TAPE: &str = "ddaff0c82b87c44e1b5f5b3e1155aa14f1fc7060c927348522e27966";
const GOLDEN_STATES: [u32; 17] = [
    0x6679_e222,
    0x9e22_2853,
    0x2285_3427,
    0x5342_7c96,
    0x27c9_6070,
    0x9607_0fcf,
    0x70fc_f114,
    0xcf11_4aa5,
    0x14aa_5511,
    0xa551_13e5,
    0x113e_5b5f,
    0xe5b5_f1b4,
    0x5f1b_4ec4,
    0xb4ec_4872,
    0xc487_2bc8,
    0x72bc_8f0a,
    0xc8f0_afdd,
];

#[uzu_test]
fn tape_layout_matches_python_oracle() {
    let config = TrellisConfig::new(32, 3);
    let cols = GOLDEN_STATES.len() as u32 * TRELLIS_V;
    let bytes: Vec<u8> =
        (0..GOLDEN_TAPE.len() / 2).map(|i| u8::from_str_radix(&GOLDEN_TAPE[2 * i..2 * i + 2], 16).unwrap()).collect();
    assert_eq!(config.bytes_per_row(cols) as usize, bytes.len(), "rate formula disagrees with the fixture");

    let stride = config.row_stride_words(cols) as usize;
    let mut words = vec![0u32; stride];
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let mut padded = [0u8; 4];
        padded[..chunk.len()].copy_from_slice(chunk);
        words[index] = u32::from_le_bytes(padded);
    }
    let tape = TrellisTape {
        config,
        rows: 1,
        cols,
        words,
    };

    // The one-shift window read packing v2 exists for, and the trellis
    // recurrence. Both must reproduce Python, and each other.
    assert_eq!(tape.states(), GOLDEN_STATES, "window read");
    assert_eq!(tape.states_by_recurrence(), GOLDEN_STATES, "recurrence");
}

/// `state_hash` against the Python oracle it was fitted with.
///
/// The fixture is dumped from `run_e6.py` driven by OUR hash, as the four bytes
/// PR #800's `q2dither` map induces — that map is not what uzu ships, so the
/// four lines that reproduce it live here rather than in the format module, and
/// what this pins is [`state_hash`], and the `hash_params` pair it is called with. The dump script itself checks all `2**16`
/// states run_e6 can enumerate; these 85 are what that check leaves behind.
const STATE_HASH_PYTHON_GOLDEN: [(u32, u32); 85] = [
    (0, 0xe7ef_0421),
    (733, 0xeddf_fbf5),
    (1466, 0xf90c_e50b),
    (2199, 0xc2f6_1602),
    (2932, 0x2ee9_0fc7),
    (3665, 0xfcf5_2df3),
    (4398, 0x082e_d2f8),
    (5131, 0xfa03_2d00),
    (5864, 0x0be3_1cdb),
    (6597, 0x1700_1dea),
    (7330, 0x1823_061a),
    (8063, 0xfefc_f44f),
    (8796, 0x0cec_fadd),
    (9529, 0x1704_041b),
    (10262, 0x39d8_1a04),
    (10995, 0xeaf1_010c),
    (11728, 0x26fe_00ea),
    (12461, 0x1609_d230),
    (13194, 0x07e6_fbfa),
    (13927, 0xbce6_fa0b),
    (14660, 0xfd09_eff7),
    (15393, 0xd019_01eb),
    (16126, 0x2828_001a),
    (16859, 0xd803_13fb),
    (17592, 0x1408_f012),
    (18325, 0x1500_ff0a),
    (19058, 0xfb1d_03e5),
    (19791, 0xe8fb_05ee),
    (20524, 0x01e5_f639),
    (21257, 0xe6da_2306),
    (21990, 0x1afc_1417),
    (22723, 0xee2e_17df),
    (23456, 0x4ffc_fbd2),
    (24189, 0xfa11_f30d),
    (24922, 0x1e30_f516),
    (25655, 0xfded_1814),
    (26388, 0x04e6_23fe),
    (27121, 0x04fc_d2f4),
    (27854, 0x03de_fb23),
    (28587, 0xf9f6_bc1a),
    (29320, 0x21d7_e228),
    (30053, 0x0e30_25ea),
    (30786, 0xe90d_db17),
    (31519, 0x1119_e90c),
    (32252, 0xe6f4_f60a),
    (32985, 0x0239_ee13),
    (33718, 0x25e5_3301),
    (34451, 0x2807_0b16),
    (35184, 0x2dde_f1b1),
    (35917, 0xf41a_f908),
    (36650, 0x1d39_05ed),
    (37383, 0x2339_fbfa),
    (38116, 0xe3d3_05f7),
    (38849, 0x28f5_f021),
    (39582, 0x44ee_130b),
    (40315, 0xf50d_2506),
    (41048, 0x0c2a_d8f9),
    (41781, 0x07eb_0011),
    (42514, 0x04d8_061f),
    (43247, 0x23e6_ecc5),
    (43980, 0xfc0f_25e7),
    (44713, 0xe8ee_011a),
    (45446, 0xe11e_33fe),
    (46179, 0xf2d6_03dd),
    (46912, 0x2e18_2214),
    (47645, 0xd2e5_0109),
    (48378, 0x4f21_070d),
    (49111, 0x0b0b_070b),
    (49844, 0xe9e5_e1f6),
    (50577, 0x14d2_e5fc),
    (51310, 0x1401_d223),
    (52043, 0x2208_131a),
    (52776, 0x170f_eee3),
    (53509, 0xf3c5_0117),
    (54242, 0xf7f8_faed),
    (54975, 0x3ecd_160a),
    (55708, 0x1af5_2a07),
    (56441, 0xe901_17fd),
    (57174, 0x1410_f806),
    (57907, 0x13cd_f8ed),
    (58640, 0x2afa_f123),
    (59373, 0x04d0_19fe),
    (60106, 0xe6d8_f4f0),
    (60839, 0xfa16_3926),
    (61572, 0xeec7_1f12),
];

#[uzu_test]
fn state_hash_matches_python_oracle() {
    /// PR #800's `11 * pairs + dither - 81` map, as the fixture was dumped
    /// under. Not what uzu decodes with; it exists to read the fixture.
    fn q2dither_word(x: u32) -> u32 {
        const M3: u32 = 0x0303_0303;
        const MF: u32 = 0x0F0F_0F0F;
        let pairs = (x & M3).wrapping_add((x >> 2) & M3).wrapping_add((x >> 4) & M3).wrapping_add((x >> 6) & M3);
        let dither = (((x & MF).wrapping_mul(3).wrapping_add(0x0101_0101)) & MF) << 1;
        pairs.wrapping_mul(11).wrapping_add(dither).wrapping_add(0x2F2F_2F2F) ^ 0x8080_8080
    }

    let (a, b) = hash_params();
    for &(state, want) in STATE_HASH_PYTHON_GOLDEN.iter() {
        assert_eq!(q2dither_word(state_hash(state, a, b)), want, "state {state:#x}");
    }
}

/// The induced level set and the epilogue scale, which are FORMAT rather than
/// implementation: if either drifts the codebook silently stops being unit
/// variance and every downstream number is off by a constant nobody would
/// notice. The counts are E8's, for its k = 3 tier-A map.
#[uzu_test]
fn codebook_is_the_fitted_one() {
    let table = codebook_table();
    assert_eq!(table.len(), 256);
    assert_eq!(*table.iter().min().unwrap(), -54);
    assert_eq!(*table.iter().max().unwrap(), 55);
    let levels: std::collections::BTreeSet<i8> = table.iter().copied().collect();
    assert_eq!(levels.len(), 74, "E8's k = 3 tier A induces 74 distinct levels");

    let scale = f64::from(codebook_scale());
    let variance: f64 = table.iter().map(|&v| (f64::from(v) * scale).powi(2)).sum::<f64>() / table.len() as f64;
    assert!((variance - 1.0).abs() < 1e-6, "decoded variance {variance}");
}
