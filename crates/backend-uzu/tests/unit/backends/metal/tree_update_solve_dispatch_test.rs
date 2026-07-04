use proc_macros::uzu_test;

use super::tree_update_solve_variant;

fn variant(
    supports_mxu: bool,
    cores: u32,
    batch: u32,
    tree: u32,
) -> (bool, u32) {
    let v = tree_update_solve_variant(supports_mxu, cores, batch, tree);
    (v.use_mxu, v.bv)
}

#[uzu_test]
fn tree_update_solve_dispatch_matches_fleet() {
    // MXU chips (M5+): simdgroup BV16 only at the B=1 small-T corner, MXU BV32 else.
    assert_eq!(variant(true, 40, 1, 64), (false, 16));
    assert_eq!(variant(true, 40, 1, 128), (true, 32));
    assert_eq!(variant(true, 40, 8, 256), (true, 32));

    // Max/Ultra (M3/M4 Max, 40): BV16 across B=1 and B<=8 tiny trees, BV32 beyond.
    assert_eq!(variant(false, 40, 1, 256), (false, 16));
    assert_eq!(variant(false, 40, 8, 64), (false, 16));
    assert_eq!(variant(false, 40, 8, 256), (false, 32));

    // Mid (M2/M4/M4 Pro, 10-20): BV16 through B=1 T<=128.
    assert_eq!(variant(false, 20, 1, 128), (false, 16));
    assert_eq!(variant(false, 10, 1, 256), (false, 32));

    // Narrow (M1, 8): BV16 only at the tiniest tree. Phone-class (A18, 5): never.
    assert_eq!(variant(false, 8, 1, 33), (false, 16));
    assert_eq!(variant(false, 8, 1, 64), (false, 32));
    assert_eq!(variant(false, 5, 1, 33), (false, 32));
}
