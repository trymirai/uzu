use uzu_engine_macros::uzu_test;

use super::GemvTile;
use crate::{
    backends::{common::kernel::matmul::MatmulShape, metal::gpu_types::gemm::GemmBLayout},
    tests::matmul::quantized_matmul_shape as shape,
};

#[uzu_test]
fn gathered_group_major_is_a_gemv_route() {
    // Gathered readout has no GEMM implementation, so group-major gathered rows must stay on GEMV.
    let mut shape: MatmulShape = shape(1, 8192, 4096, 4, 32);
    shape.gathered = true;
    let tile = GemvTile::quantized(8, 4, 1, 32, 2);
    assert!(super::super::kernel::GemvSpecialization::select_tile(&shape, GemmBLayout::RowMajor, tile).is_some());
}
