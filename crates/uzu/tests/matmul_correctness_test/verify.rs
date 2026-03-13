use metal::{MTLDeviceExt, MTLResourceOptions};
use uzu::backends::{
    common::{
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
        kernel::matmul::MatmulKernel,
    },
    metal::{MetalContext, kernel::matmul::MatmulMetalKernel},
};

use super::{
    common::matmul::{DtypeCombo, MatmulVariant, TestShape, make_full_precision_arguments},
    output::TestResult,
    reference::{generate_typed_data, ndarray_reference, output_to_f64, tolerance_for},
};

pub fn run_correctness_case(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    dispatch_path_name: &str,
    variant: MatmulVariant,
) -> TestResult {
    let a_bytes = generate_typed_data(combo.a_dtype, shape.batch * shape.input_dim, 13, -6);
    let b_bytes = generate_typed_data(combo.b_dtype, shape.output_dim * shape.input_dim, 17, -8);

    let metal_result = run_metal_matmul(context, combo, &a_bytes, &b_bytes, shape, variant);
    let reference = ndarray_reference(combo, &a_bytes, &b_bytes, shape);

    let tolerance = tolerance_for(combo, shape);
    let max_diff = metal_result
        .iter()
        .zip(reference.iter())
        .map(|(&metal_value, &reference_value)| (metal_value - reference_value).abs())
        .fold(0.0f64, f64::max);

    TestResult {
        combo: format!("{combo}"),
        shape: format!("{shape}"),
        dispatch_path: dispatch_path_name.to_owned(),
        passed: max_diff <= tolerance,
        max_diff,
        tolerance,
    }
}

fn run_metal_matmul(
    context: &MetalContext,
    combo: &DtypeCombo,
    a_bytes: &[u8],
    b_bytes: &[u8],
    shape: &TestShape,
    variant: MatmulVariant,
) -> Vec<f64> {
    let a_buffer = context.device
        .new_buffer_with_data(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create A buffer");
    let b_buffer = context.device
        .new_buffer_with_data(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create B buffer");
    let mut d_buffer = context.device
        .new_buffer(
            shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create D buffer");

    let mut kernel = MatmulMetalKernel::new(context, combo.output_dtype).expect("kernel creation");

    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();
    let arguments = make_full_precision_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
    match variant {
        MatmulVariant::Gemv => kernel.encode_gemv(context, &mut command_buffer, arguments),
        MatmulVariant::GemmMpp => kernel.encode_gemm_mpp(context, &mut command_buffer, arguments),
        MatmulVariant::Gemm => kernel.encode_gemm(context, &mut command_buffer, arguments),
    }
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let element_count = shape.batch * shape.output_dim;
    output_to_f64(combo.output_dtype, &d_buffer, element_count)
}
