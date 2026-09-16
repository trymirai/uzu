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
    let tape = TrellisTape::from_packed_rows(config, 1, cols, &bytes);

    // The one-shift window read packing v2 exists for, and the trellis
    // recurrence. Both must reproduce Python, and each other.
    assert_eq!(tape.states(), GOLDEN_STATES, "window read");
    assert_eq!(tape.states_by_recurrence(), GOLDEN_STATES, "recurrence");
}

/// Two rows each of the tapes lalamo fits, written by its `TrellisSpec` at
/// trymirai/lalamo#364 (`_states_to_tape` / `_states_to_levels`, commit
/// 936facd4): `L = 16`, `restart_columns = 64`, 128 columns at `k = 2` and 192
/// at `k = 3`. Headers and codes drawn with `numpy.random.default_rng(808)`,
/// states run through the recurrence, then packed. Per row: the tape bytes,
/// the 32-bit states as eight hex digits each, and the int8 levels those
/// states decode to as one byte each. uzu reads a fitted tape; this pins that
/// read, tape by tape, against the writer.
const LALAMO_K2_TAPE: [&str; 2] = [
    "0b2647642e369042ad404a5443ab519cd8e8bd493b30b15f6a28c84af981f8e2d46d",
    "1a28e62beadcaf6e90774e8da4dd7bfe73258960912dd3212c07f133eb170ee9f05a",
];
const LALAMO_K2_STATES: [&str; 2] = [
    "0000d89c00009c51000051ab0000ab43000043540000544a00004a40000040ad0000ad4200004290000090360000362e00002e6400006447000047260000260b00006dd40000d4e20000e2f80000f881000081f90000f94a00004ac80000c8280000286a00006a5f00005fb10000b1300000303b00003b49000049bd0000bde8",
    "000073fe0000fe7b00007bdd0000dda40000a48d00008d4e00004e77000077900000906e00006eaf0000afdc0000dcea0000ea2b00002be60000e6280000281a00005af00000f0e90000e90e00000e17000017eb0000eb33000033f10000f1070000072c00002c21000021d30000d32d00002d91000091600000608900008925",
];
const LALAMO_K2_LEVELS: [&str; 2] = [
    "00ff0ff80c03fdf5f5f803140212fb06f5fde6f6e0032407f6eaf608fc05052013030fe5fdf5ddf4ed0bedfde51001e9f2050b030e03e5fc04060501f3fe13041f05060f270d1405eeedfc0000f60b19fafaeaf5fc1300f0ec01ee020d020f0bf0e80013f301cafbfce51014f824f62fee0c0b02240cf4f51103eaf8eaf9fd01",
    "0b04200c142816f209ea03fffafe0a14f0fcfbf90ff2fcf5f6ed0af8fcda3702d2eae2091b000fec00190be20a08d20410141ff6e0eae20607ee041000f8faf91706fa1ffe19f606ede8faf207fa0004070f21161005011211f9ee1c1715fc2808051bfeedf9e2e40427f3152ffa02fb272f37ed13f7fc0427ed04f1200b19da",
];
const LALAMO_K3_TAPE: [&str; 2] = [
    "cd28b1507b6cc5158a5adedd7142971c8035203d9d76bbcec8287476d296ff805600287496fb9fba04ec73ba62a01118121ef6d253db5b4ffa45615c9f7635cef42e5202ccdc84c3140a",
    "f039ac1ff50b47b9a1c20e97c964ec930392ae0d30db3ac5316ee82ae2d5da011b665697ac67440bb14af4e39ede7104a000361a7dc29c91ac9b12001846f71354a2b38094e9c4961e0f",
];
const LALAMO_K3_STATES: [&str; 2] = [
    "00008ceb0000bb76000069d300003d20000003580000801c0000c9740000427100001ddd0000de5a0000a8a1000015c5000056c700007b5000000b12000028cd000012180000811a0000a06200002ba7000073ec0000c04b0000ba9f0000ffb90000967400004280000000560000680f0000ff9600006d27000076740000428c0000a14c0000c38400004dcc0000cc020000252200002ef400004ce300003576000069f500005c610000145f0000fa4f0000f5bd0000db5300003d2f0000f61e",
    "00001c5300003adb0000b30000000dae0000e9200000039300003ec6000064c90000997000000ec200002a1b0000b947000070bf0000f51f0000fac3000039f00000a0040000471d0000de9e0000ee3f0000f44a0000ab1000000b440000467a0000ac97000075660000661b0000b01d0000dad500005e2200002ae8000086e30000f1e9000096c400004e990000948000000b3a0000a2540000413f0000f7460000618000000012000029ba0000ac91000019cc0000c27d0000d1a300003600",
];
const LALAMO_K3_LEVELS: [&str; 2] = [
    "05241f0701e5060d0d1be5ea11e5f3f7e00a0800f6f00727ecfb031907f3edebec1b0ff7e4f0ffe5e8f8dafefa1dfefafd03e6140f0407200bdef0ea07f4fdea070b0c0b0ff1ebf105f6fd050c1f062f07f809da100104f800e80bed081105010907edf5fedae2130602d203f52f0e0403ea1006eedd180301140d10e8181ff105d5f3f61313f6f11feef1ee03ea1cfe041905010becea09f1d2e0fd0bf91b1127da08170211f306e5131109f9f81423e80119f801e2f113eaf4eeea1fe62f10",
    "f5f6080c11fef614072cfb10f8fde2e200f61814f319da2027110b24191f03f51beaf9100518f2fbf5f61403140b07f61409140e10fe09ed0806052cfb150e14f31701fa1105feca1dfefbeeeaf2dde6ed0b141d0bf5e8daecfaf1fa130ffe17060d10f513021f1106f509191712eefa0613fe04030a080e0308f40ff6e2f1f6001708f60bf20100e627e8fefc1914112ccaee07001f08080319031b1b030c02ed1bf111ec1c01f2fadd24e81015031006ee1908e501fbf4fbf20b271109f6e6",
];

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect()
}

#[uzu_test]
fn restart_tape_layout_matches_lalamo() {
    fn check(
        config: TrellisConfig,
        tape: &[&str; 2],
        states: &[&str; 2],
        levels: &[&str; 2],
    ) {
        let bytes: Vec<u8> = tape.iter().flat_map(|hex| hex_bytes(hex)).collect();
        let want_levels: Vec<i8> = levels.iter().flat_map(|hex| hex_bytes(hex)).map(|byte| byte as i8).collect();
        let cols = want_levels.len() as u32 / 2;
        let want_states: Vec<u32> = states
            .iter()
            .flat_map(|hex| (0..hex.len() / 8).map(move |i| u32::from_str_radix(&hex[8 * i..8 * i + 8], 16).unwrap()))
            .collect();
        let tape = TrellisTape::from_packed_rows(config, 2, cols, &bytes);
        assert_eq!(tape.states(), want_states, "{config:?} window read");
        assert_eq!(tape.states_by_recurrence(), want_states, "{config:?} recurrence");
        assert_eq!(tape.codes(), want_levels, "{config:?} levels");
    }
    check(TrellisConfig::new(16, 2).with_restart(64), &LALAMO_K2_TAPE, &LALAMO_K2_STATES, &LALAMO_K2_LEVELS);
    check(TrellisConfig::new(16, 3).with_restart(64), &LALAMO_K3_TAPE, &LALAMO_K3_STATES, &LALAMO_K3_LEVELS);
}

/// Every restart width the kernels accept, at row lengths that are one tape,
/// several, and enough for a window at every alignment: the window read and
/// the recurrence must agree, and a row must keep a word of slack.
#[uzu_test]
fn restart_tapes_read_back_by_window_and_by_recurrence() {
    for (config, cols) in [
        (TrellisConfig::new(16, 2).with_restart(16), 64u32),
        (TrellisConfig::new(16, 2).with_restart(32), 512),
        (TrellisConfig::new(16, 2).with_restart(64), 64),
        (TrellisConfig::new(16, 3).with_restart(64), 1152),
        (TrellisConfig::new(32, 3).with_restart(128), 1024),
        (TrellisConfig::new(24, 3).with_restart(256), 1024),
        (TrellisConfig::new(16, 1).with_restart(512), 1024),
        (TrellisConfig::new(16, 1).with_restart(1536), 3072),
        (TrellisConfig::new(16, 4).with_restart(64), 256),
    ] {
        let tape = TrellisTape::random(config, 3, cols, 0xA5A5);
        assert_eq!(tape.states(), tape.states_by_recurrence(), "{config:?} cols={cols}");
        assert!(config.row_stride_words(cols) * 32 >= config.bits_per_row(cols) + 32, "{config:?}");
    }
}

#[uzu_test]
fn trellis_params_carry_the_tape_geometry() {
    let restart = trellis_params(TrellisConfig::new(16, 2).with_restart(64), 128).unwrap();
    assert_eq!((restart.tape_steps, restart.tape_bits), (16, 136));
    let whole_row = trellis_params(TrellisConfig::new(16, 2), 128).unwrap();
    assert_eq!((whole_row.tape_steps, whole_row.tape_bits), (32, 264));
    assert_eq!((restart.row_stride_words, whole_row.row_stride_words), (13, 13));
}

#[uzu_test]
fn restart_widths_the_kernels_cannot_walk_are_rejected() {
    let rejected = |columns: u32, k: u32| trellis_params(TrellisConfig::new(16, 2).with_restart(columns), k).is_none();
    assert!(rejected(0, 4096), "no tape at all");
    assert!(rejected(8, 4096), "narrower than a lane's run");
    assert!(rejected(48, 4800), "not whole runs");
    assert!(rejected(192, 4800), "neither divides the widest K group nor is whole multiples of it");
    assert!(rejected(64, 4096 + 32), "a row that is not whole tapes");
    assert!(!rejected(64, 4096));
    assert!(!rejected(1024, 4096));
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
