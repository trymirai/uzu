use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::{
        common::{Context, kernel::matmul::MatmulKernel},
        metal::{context::MetalContext, kernel::matmul::MatmulMetalKernel},
    },
    tests::matmul::{a8_format, matmul_shape as shape, quantized_matmul_shape},
};

fn quant(mut shape: MatmulShape) -> MatmulShape {
    shape.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
    shape.b_bits = Some(4);
    shape.b_group_size = Some(64);
    shape.signed_codes = true;
    shape
}

fn problem(
    shape: MatmulShape,
    output_data_type: DataType,
) -> GemmProblem {
    GemmProblem::new(shape, DataType::BF16, output_data_type, true)
}

#[uzu_test]
fn illegal_layouts_fall_back() {
    use GemmEngine::*;

    let mut invalid_layout = quant(shape(64, 4096, 4096));
    invalid_layout.b_transpose = false;
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Simdgroup);
    invalid_layout.a = a8_format();
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Mxu);
    assert_eq!(problem(quant(shape(64, 4096, 4095)), DataType::BF16).select_plan().engine, Simdgroup);
}

#[uzu_test]
fn forced_engine_errors_are_preserved() {
    let huge = shape(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        GemmProblem::new(huge, DataType::BF16, DataType::BF16, false).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::MxuUnavailable)
    );

    let mut invalid_layout = quant(huge);
    invalid_layout.b_transpose = false;
    assert_eq!(
        problem(invalid_layout, DataType::BF16).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::UnsupportedLayout("quantized weights require transposed contiguous B"))
    );
}

#[uzu_test]
fn int8_activations_require_mxu_tiled_weights() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let kernel =
        MatmulMetalKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).expect("matmul kernel");
    for (n, k, bits, expected) in [
        (96, 256, 4, true),
        (72, 256, 4, false),
        (64, 40960, 4, true),
        (64, 41088, 4, false),
        (64, 20480, 8, true),
        (64, 20608, 8, false),
    ] {
        let shape = quantized_matmul_shape(16, n, k, bits, 64);
        assert_eq!(kernel.select_activation_quantization(&shape, &context).is_some(), expected, "n={n} k={k} W{bits}");
    }
}
