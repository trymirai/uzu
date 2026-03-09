use metal::{MTLDeviceExt, MTLResourceOptions};
use uzu::backends::{
    common::{
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
        kernel::matmul::{MatmulDispatchDescriptor, MatmulKernel},
    },
    metal::{Metal, MetalContext},
};

use super::common::matmul::{DtypeCombo, TestShape, make_arguments};
use super::output::TestResult;
use super::reference::{generate_typed_data, ndarray_reference, output_to_f64, tolerance_for};

pub fn run_correctness_case(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    dispatch_path_name: &str,
    dispatch_descriptor: &MatmulDispatchDescriptor,
) -> TestResult {
    let a_bytes = generate_typed_data(combo.a_dtype, shape.batch * shape.input_dim, 13, -6);
    let b_bytes = generate_typed_data(combo.b_dtype, shape.output_dim * shape.input_dim, 17, -8);

    let metal_result = run_metal_matmul(context, combo, &a_bytes, &b_bytes, shape, dispatch_descriptor);
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
    dispatch_descriptor: &MatmulDispatchDescriptor,
) -> Vec<f64> {
    let a_buffer = context
        .device
        .new_buffer_with_data(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create A buffer");
    let b_buffer = context
        .device
        .new_buffer_with_data(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create B buffer");
    let mut d_buffer = context
        .device
        .new_buffer(
            shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create D buffer");

    let mut kernel =
        MatmulKernel::<Metal>::new_mixed(combo.a_dtype, combo.b_dtype, combo.output_dtype).expect("kernel creation");

    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();
    let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
    kernel.encode_with_descriptor(context, arguments, dispatch_descriptor, &mut command_buffer).expect("encode");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let element_count = shape.batch * shape.output_dim;
    output_to_f64(combo.output_dtype, &d_buffer, element_count)
}
