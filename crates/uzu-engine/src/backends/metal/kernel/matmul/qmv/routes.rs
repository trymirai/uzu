// Qwen3.6-27B W4/ZP and W8/Symmetric, G32/G64.
// Tuned on M1, M2, M2 Pro, M3 Max, M4, M4 Pro, M5 Max.
use super::{
    super::{
        gemm::{GemmEngine, GemmPlan},
        gemv::GemvTile,
    },
    QmvRoute,
};
use crate::backends::{
    common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
        kernel::matmul::MatmulShape,
    },
    metal::device_profile::{DeviceIdentity, DeviceProfile},
};

const DOWN: u8 = 1 << 0;
const GATE: u8 = 1 << 1;
const GATE_UP: u8 = 1 << 2;
const PROJECTION_IN: u8 = 1 << 3;
const PROJECTION_OUT: u8 = 1 << 4;
const QKV: u8 = 1 << 5;
const READOUT: u8 = 1 << 6;

#[derive(Clone, Copy)]
struct RouteRow {
    identity: DeviceIdentity,
    bits: u32,
    group: u32,
    m: u32,
    shapes: u8,
    route: QmvRoute,
}

const fn shape(
    n: u32,
    k: u32,
) -> u8 {
    match (n, k) {
        (5120, 17408) => DOWN,
        (6144, 5120) => GATE,
        (34816, 5120) => GATE_UP,
        (16480, 5120) => PROJECTION_IN,
        (5120, 6144) => PROJECTION_OUT,
        (8192, 5120) => QKV,
        (248320, 5120) => READOUT,
        _ => 0,
    }
}

#[rustfmt::skip]
macro_rules! tuned { ($input:literal, $output:literal, $lanes:literal, $group_lanes:literal, $simdgroups:literal) => { QmvRoute::Tuned(GemvTile::quantized_output_tile($simdgroups, $output, $input, $lanes, $group_lanes)) }; }
#[rustfmt::skip]
macro_rules! main_gemv { ($input:literal, $results_per_simdgroup:literal, $lanes:literal, $group_lanes:literal, $simdgroups:literal) => { QmvRoute::MainGemv(GemvTile::quantized($simdgroups, $results_per_simdgroup, $input, $lanes, $group_lanes)) }; }
#[rustfmt::skip]
macro_rules! main_gemm { ($engine:ident, $tiling:ident, $split:literal) => { QmvRoute::MainGemm(GemmPlan { engine: GemmEngine::$engine, tiling: GemmTiling::$tiling, split_k: $split }) }; }
#[rustfmt::skip]
macro_rules! qmv_format { (w4_zp_g32) => { (4, 32) }; (w4_zp_g64) => { (4, 64) }; (w8_sym_g32) => { (8, 32) }; (w8_sym_g64) => { (8, 64) }; }
#[rustfmt::skip]
macro_rules! row { ($identity:ident, $format:ident, $m:literal, $shapes:expr, $route:expr) => { RouteRow { identity: DeviceIdentity::$identity, bits: qmv_format!($format).0, group: qmv_format!($format).1, m: $m, shapes: $shapes, route: $route } }; }

#[rustfmt::skip]
const ROWS: &[RouteRow] = &[
    row!(M1, w4_zp_g32, 2, READOUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M1, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(2, 16, 8, 1, 2)),
    row!(M1, w4_zp_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M1, w4_zp_g32, 4, DOWN, main_gemv!(1, 4, 32, 2, 8)),
    row!(M1, w4_zp_g32, 4, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M1, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M1, w4_zp_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 16, 1, 4)),
    row!(M1, w4_zp_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M1, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M1, w4_zp_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M1, w4_zp_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 16, 1, 4)),
    row!(M1, w4_zp_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 16, 1, 4)),
    row!(M1, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 16, 1, 4)),
    row!(M1, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M1, w8_sym_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M1, w8_sym_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M1, w8_sym_g32, 3, PROJECTION_IN | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M1, w8_sym_g32, 4, GATE_UP, main_gemv!(1, 4, 32, 4, 8)),
    row!(M1, w8_sym_g32, 4, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M1, w8_sym_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M1, w8_sym_g32, 6, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M1, w8_sym_g32, 6, DOWN, tuned!(6, 16, 16, 1, 4)),
    row!(M1, w8_sym_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M1, w8_sym_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M1, w8_sym_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M1, w8_sym_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M1, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M1, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M1, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 2, PROJECTION_IN, main_gemv!(1, 4, 32, 2, 8)),
    row!(M2, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 7, READOUT, tuned!(4, 8, 8, 1, 2)),
    row!(M2, w4_zp_g32, 7, GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(7, 16, 8, 1, 2)),
    row!(M2, w4_zp_g32, 7, DOWN | GATE, tuned!(8, 8, 8, 1, 2)),
    row!(M2, w4_zp_g64, 2, PROJECTION_IN, tuned!(2, 8, 8, 1, 2)),
    row!(M2, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M2, w4_zp_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M2, w4_zp_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2, w4_zp_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M2, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M2, w8_sym_g32, 2, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M2, w8_sym_g32, 2, DOWN, tuned!(2, 16, 8, 1, 2)),
    row!(M2, w8_sym_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M2, w8_sym_g32, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2, w8_sym_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M2, w8_sym_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2, w8_sym_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 2, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M2, w8_sym_g64, 2, DOWN, tuned!(2, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 3, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M2, w8_sym_g64, 3, DOWN, tuned!(3, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 2, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M2Pro, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 3, DOWN | GATE_UP | READOUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M2Pro, w4_zp_g32, 3, PROJECTION_IN, tuned!(3, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 3, GATE | PROJECTION_OUT | QKV, tuned!(3, 16, 16, 1, 4)),
    row!(M2Pro, w4_zp_g32, 4, DOWN | GATE_UP | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M2Pro, w4_zp_g32, 4, PROJECTION_IN, tuned!(4, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 4, GATE, tuned!(4, 16, 16, 1, 4)),
    row!(M2Pro, w4_zp_g32, 5, GATE_UP | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w4_zp_g32, 5, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w4_zp_g32, 5, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w4_zp_g32, 5, PROJECTION_IN, tuned!(5, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 6, GATE_UP | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w4_zp_g32, 6, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w4_zp_g32, 6, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w4_zp_g32, 6, PROJECTION_IN, tuned!(6, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g32, 7, GATE_UP | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w4_zp_g32, 7, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w4_zp_g32, 7, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w4_zp_g32, 7, PROJECTION_IN, tuned!(7, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 3, PROJECTION_OUT, tuned!(3, 16, 16, 1, 4)),
    row!(M2Pro, w4_zp_g64, 4, GATE_UP | PROJECTION_IN | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 4, DOWN | GATE | PROJECTION_OUT | QKV, tuned!(4, 16, 16, 1, 4)),
    row!(M2Pro, w4_zp_g64, 5, PROJECTION_IN, tuned!(5, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 16, 1, 4)),
    row!(M2Pro, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2Pro, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M2Pro, w8_sym_g32, 3, DOWN | GATE_UP | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M2Pro, w8_sym_g32, 3, PROJECTION_IN, tuned!(3, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g32, 3, GATE, tuned!(3, 16, 16, 1, 4)),
    row!(M2Pro, w8_sym_g32, 4, DOWN | GATE_UP | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M2Pro, w8_sym_g32, 4, PROJECTION_IN, tuned!(4, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g32, 4, GATE | PROJECTION_OUT, tuned!(4, 16, 16, 1, 4)),
    row!(M2Pro, w8_sym_g32, 5, GATE_UP | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w8_sym_g32, 5, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w8_sym_g32, 5, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w8_sym_g32, 5, PROJECTION_IN, tuned!(5, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g32, 6, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w8_sym_g32, 6, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w8_sym_g32, 6, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w8_sym_g32, 7, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 1)),
    row!(M2Pro, w8_sym_g32, 7, DOWN | GATE | QKV, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M2Pro, w8_sym_g32, 7, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M2Pro, w8_sym_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M2Pro, w8_sym_g64, 3, DOWN | GATE | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 8, 8)),
    row!(M2Pro, w8_sym_g64, 3, GATE_UP | PROJECTION_IN | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g64, 4, DOWN | PROJECTION_OUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M2Pro, w8_sym_g64, 4, GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M2Pro, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 2, 8)),
    row!(M3Max, w4_zp_g32, 2, READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 3, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M3Max, w4_zp_g32, 3, GATE, tuned!(3, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 3, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g32, 4, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M3Max, w4_zp_g32, 4, DOWN | GATE_UP | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 4, GATE | PROJECTION_IN, tuned!(4, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 5, PROJECTION_IN, tuned!(5, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g32, 6, PROJECTION_IN, tuned!(6, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 2, GATE_UP | PROJECTION_IN | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M3Max, w4_zp_g64, 2, DOWN | GATE | PROJECTION_OUT, tuned!(2, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 3, DOWN | GATE | PROJECTION_OUT, tuned!(3, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 3, GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g64, 4, DOWN | GATE | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 4, GATE_UP | PROJECTION_IN, tuned!(4, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g64, 5, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 5, GATE_UP, tuned!(5, 16, 16, 1, 4)),
    row!(M3Max, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M3Max, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M3Max, w8_sym_g32, 2, PROJECTION_OUT | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 3, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M3Max, w8_sym_g32, 3, GATE_UP | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 4, GATE, main_gemv!(1, 4, 32, 4, 8)),
    row!(M3Max, w8_sym_g32, 4, DOWN | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(4, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 4, READOUT, tuned!(4, 16, 16, 1, 4)),
    row!(M3Max, w8_sym_g32, 5, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M3Max, w8_sym_g32, 5, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M3Max, w8_sym_g32, 5, DOWN, tuned!(5, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 5, GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(5, 16, 16, 1, 4)),
    row!(M3Max, w8_sym_g32, 6, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M3Max, w8_sym_g32, 6, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g32, 6, PROJECTION_OUT, tuned!(6, 16, 16, 1, 4)),
    row!(M3Max, w8_sym_g32, 7, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M3Max, w8_sym_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M3Max, w8_sym_g64, 3, GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M3Max, w8_sym_g64, 3, DOWN | PROJECTION_OUT, tuned!(3, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g64, 4, GATE_UP | PROJECTION_IN, main_gemv!(1, 4, 32, 8, 8)),
    row!(M3Max, w8_sym_g64, 4, DOWN | GATE | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M3Max, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4, w4_zp_g32, 2, READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 3, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4, w4_zp_g32, 3, GATE, tuned!(3, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 3, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M4, w4_zp_g32, 4, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4, w4_zp_g32, 4, DOWN | GATE_UP | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 4, GATE | PROJECTION_IN, tuned!(4, 16, 16, 1, 4)),
    row!(M4, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 5, PROJECTION_IN, tuned!(5, 16, 16, 1, 4)),
    row!(M4, w4_zp_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4, w4_zp_g32, 6, PROJECTION_IN, tuned!(6, 16, 16, 1, 4)),
    row!(M4, w4_zp_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4, w4_zp_g64, 3, DOWN, tuned!(3, 16, 8, 1, 2)),
    row!(M4, w4_zp_g64, 3, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M4, w4_zp_g64, 4, DOWN | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M4, w4_zp_g64, 4, GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(4, 16, 16, 1, 4)),
    row!(M4, w4_zp_g64, 5, DOWN | GATE_UP | PROJECTION_IN | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4, w4_zp_g64, 5, GATE | PROJECTION_OUT | QKV, tuned!(5, 16, 16, 1, 4)),
    row!(M4, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4, w8_sym_g32, 2, PROJECTION_OUT | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 3, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4, w8_sym_g32, 3, GATE_UP | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 4, GATE, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4, w8_sym_g32, 4, DOWN | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(4, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 4, READOUT, tuned!(4, 16, 16, 1, 4)),
    row!(M4, w8_sym_g32, 5, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M4, w8_sym_g32, 5, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M4, w8_sym_g32, 5, DOWN, tuned!(5, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 5, GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(5, 16, 16, 1, 4)),
    row!(M4, w8_sym_g32, 6, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M4, w8_sym_g32, 6, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4, w8_sym_g32, 6, PROJECTION_OUT, tuned!(6, 16, 16, 1, 4)),
    row!(M4, w8_sym_g32, 7, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M4, w8_sym_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4, w8_sym_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M4, w8_sym_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M4, w8_sym_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M4, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4Pro, w4_zp_g32, 2, READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 3, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4Pro, w4_zp_g32, 3, GATE, tuned!(3, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 3, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g32, 4, PROJECTION_OUT, main_gemv!(1, 4, 32, 2, 8)),
    row!(M4Pro, w4_zp_g32, 4, DOWN | GATE_UP | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 4, GATE | PROJECTION_IN, tuned!(4, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 5, PROJECTION_IN, tuned!(5, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g32, 6, PROJECTION_IN, tuned!(6, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4Pro, w4_zp_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_OUT | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g64, 4, PROJECTION_IN | QKV, tuned!(4, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_OUT | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g64, 5, PROJECTION_IN | QKV, tuned!(5, 16, 16, 1, 4)),
    row!(M4Pro, w4_zp_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4Pro, w4_zp_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4Pro, w8_sym_g32, 2, PROJECTION_OUT | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 3, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4Pro, w8_sym_g32, 3, GATE_UP | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 4, GATE, main_gemv!(1, 4, 32, 4, 8)),
    row!(M4Pro, w8_sym_g32, 4, DOWN | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(4, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 4, READOUT, tuned!(4, 16, 16, 1, 4)),
    row!(M4Pro, w8_sym_g32, 5, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M4Pro, w8_sym_g32, 5, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M4Pro, w8_sym_g32, 5, DOWN, tuned!(5, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 5, GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(5, 16, 16, 1, 4)),
    row!(M4Pro, w8_sym_g32, 6, GATE, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 2)),
    row!(M4Pro, w8_sym_g32, 6, DOWN | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g32, 6, PROJECTION_OUT, tuned!(6, 16, 16, 1, 4)),
    row!(M4Pro, w8_sym_g32, 7, PROJECTION_OUT, main_gemm!(Simdgroup, Tile8x32x32_Simdgroups1x1, 3)),
    row!(M4Pro, w8_sym_g32, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV, main_gemv!(1, 4, 32, 8, 8)),
    row!(M4Pro, w8_sym_g64, 2, READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, main_gemv!(1, 4, 32, 8, 8)),
    row!(M4Pro, w8_sym_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g64, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M4Pro, w8_sym_g64, 7, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(7, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g32, 2, DOWN | GATE | PROJECTION_IN | PROJECTION_OUT | QKV, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g32, 2, GATE_UP | READOUT, tuned!(2, 16, 16, 1, 4)),
    row!(M5Max, w4_zp_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g32, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g32, 6, DOWN, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w4_zp_g32, 6, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w4_zp_g32, 6, GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 16, 1, 4)),
    row!(M5Max, w4_zp_g32, 7, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 1)),
    row!(M5Max, w4_zp_g32, 7, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w4_zp_g32, 7, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w4_zp_g32, 7, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
    row!(M5Max, w4_zp_g64, 2, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g64, 4, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M5Max, w4_zp_g64, 6, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 1)),
    row!(M5Max, w4_zp_g64, 6, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w4_zp_g64, 6, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w4_zp_g64, 6, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
    row!(M5Max, w4_zp_g64, 7, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 1)),
    row!(M5Max, w4_zp_g64, 7, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w4_zp_g64, 7, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w4_zp_g64, 7, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
    row!(M5Max, w8_sym_g32, 2, GATE_UP | PROJECTION_IN, main_gemv!(1, 4, 32, 4, 8)),
    row!(M5Max, w8_sym_g32, 2, DOWN | GATE | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g32, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g32, 4, PROJECTION_IN, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g32, 4, DOWN | GATE | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g32, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g32, 6, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(6, 16, 16, 1, 4)),
    row!(M5Max, w8_sym_g32, 7, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 1)),
    row!(M5Max, w8_sym_g32, 7, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w8_sym_g32, 7, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w8_sym_g32, 7, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
    row!(M5Max, w8_sym_g64, 2, GATE | GATE_UP | PROJECTION_IN, main_gemv!(1, 4, 32, 8, 8)),
    row!(M5Max, w8_sym_g64, 2, DOWN | PROJECTION_OUT | QKV | READOUT, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 3, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(3, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 4, GATE | PROJECTION_IN, tuned!(2, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 4, DOWN | GATE_UP | PROJECTION_OUT | QKV | READOUT, tuned!(4, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 5, DOWN | GATE | GATE_UP | PROJECTION_IN | PROJECTION_OUT | QKV | READOUT, tuned!(5, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 6, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w8_sym_g64, 6, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w8_sym_g64, 6, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
    row!(M5Max, w8_sym_g64, 6, GATE_UP | PROJECTION_IN | READOUT, tuned!(6, 16, 8, 1, 2)),
    row!(M5Max, w8_sym_g64, 7, GATE_UP | PROJECTION_IN | READOUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 1)),
    row!(M5Max, w8_sym_g64, 7, DOWN | QKV, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 4)),
    row!(M5Max, w8_sym_g64, 7, GATE, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 5)),
    row!(M5Max, w8_sym_g64, 7, PROJECTION_OUT, main_gemm!(Mxu, Tile32x64x256_Simdgroups2x2, 6)),
];

fn qmv_format(shape: &MatmulShape) -> Option<(u32, u32)> {
    match (shape.b_prologue, shape.b_bits, shape.b_group_size) {
        (GemmBPrologueKind::ScaleZeroPointDequant, Some(4), Some(group @ (32 | 64))) => Some((4, group)),
        (GemmBPrologueKind::ScaleSymmetricDequant, Some(8), Some(group @ (32 | 64))) => Some((8, group)),
        _ => None,
    }
}

pub fn route(
    device: DeviceProfile,
    shape: &MatmulShape,
    all_bf16: bool,
) -> Option<QmvRoute> {
    if !all_bf16
        || shape.m < 2
        || shape.m > 7
        || !shape.a_full_precision
        || !shape.b_transpose
        || shape.b_leading_dimension.is_some()
        || shape.gathered
    {
        return None;
    }
    let (bits, group) = qmv_format(shape)?;
    let mask = self::shape(shape.n, shape.k);
    if mask == 0 {
        return None;
    }
    let matches =
        |row: &&RouteRow| row.bits == bits && row.group == group && row.m == shape.m && row.shapes & mask != 0;
    let route =
        ROWS.iter().filter(matches).find(|row| row.identity == device.identity()).map(|row| row.route).or_else(|| {
            let same_family = |row: &&RouteRow| device.gpu_family().contains(row.identity);
            let mut routes = ROWS.iter().filter(matches).filter(same_family).map(|row| row.route);
            let route = routes.next()?;
            routes.all(|candidate| candidate == route).then_some(route)
        });
    route.filter(|route| {
        !matches!(
            route,
            QmvRoute::MainGemm(GemmPlan {
                engine: GemmEngine::Mxu,
                ..
            })
        ) || device.supports_mxu()
    })
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/matmul/qmv/routes_test.rs"]
mod tests;
