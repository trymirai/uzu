use proc_macros::uzu_test;

use super::*;
use crate::{
    backends::common::gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
    data_type::DataType,
};

fn shape(
    m: u32,
    n: u32,
    k: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::FullPrecision,
        b_bits: None,
        b_group_size: None,
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        d_transform: GemmDTransform::empty(),
    }
}

#[uzu_test]
fn gemv_override_stays_at_orchestration_boundary() {
    let cases = [
        (shape(4, 4096, 8192), DataType::BF16, true),
        (shape(4, 8192, 4096), DataType::BF16, false),
        (shape(4, 4096, 4096), DataType::F32, false),
        (shape(3, 4096, 8192), DataType::BF16, false),
        (shape(5, 4096, 8192), DataType::BF16, false),
    ];
    for (shape, data_type, expected) in cases {
        let problem = GemmProblem::new(shape, data_type, data_type, true);
        assert_eq!(
            MatmulMetalKernel::prefer_gemm_over_gemv(shape, problem.select_plan(), data_type, data_type, data_type,),
            expected
        );
    }

    let mut gathered = shape(4, 4096, 8192);
    gathered.gathered = true;
    let problem = GemmProblem::new(gathered, DataType::BF16, DataType::BF16, true);
    assert!(!MatmulMetalKernel::prefer_gemm_over_gemv(
        gathered,
        problem.select_plan(),
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
    ));
}
