use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::backends::{
    common::kernel::matmul::{MatmulArguments, MatmulDispatchDescriptor, MatmulKernel},
    metal::{Metal, MetalContext},
};

use super::combos::DtypeCombo;
use super::error::BenchError;
use super::output::PerfResult;
use super::shapes::TestShape;

const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;

fn fill_buffer_random(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_count: usize,
) {
    let pointer = buffer.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(pointer, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn encode_and_run(
    context: &MetalContext,
    kernel: &mut MatmulKernel<Metal>,
    arguments: MatmulArguments<Metal>,
    dispatch_descriptor: &MatmulDispatchDescriptor,
) -> Result<f64, BenchError> {
    let command_buffer = context
        .command_queue
        .command_buffer()
        .ok_or(BenchError::CommandBuffer)?
        .to_owned();
    let mut compute_encoder = command_buffer
        .new_compute_command_encoder()
        .ok_or(BenchError::ComputeEncoder)?;
    kernel
        .encode_with_descriptor(context, arguments, dispatch_descriptor, &mut compute_encoder)
        .map_err(|e| BenchError::Encode(e.to_string()))?;
    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    match (command_buffer.gpu_start_time(), command_buffer.gpu_end_time()) {
        (Some(start), Some(end)) => Ok((end - start) * 1000.0),
        _ => Err(BenchError::GpuTimestamps),
    }
}

pub fn make_arguments<'a>(
    a_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    b_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    d_buffer: &'a mut Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: &TestShape,
) -> MatmulArguments<'a, Metal> {
    MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        d: d_buffer,
        bias: None,
        batch: shape.batch as i32,
        input_dim: shape.input_dim as i32,
        output_dim: shape.output_dim as i32,
        lda: shape.input_dim as i32,
        ldb: shape.input_dim as i32,
        ldd: shape.output_dim as i32,
        batch_count: 1,
        transpose_b: true,
    }
}

fn run_benchmark(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    dispatch_descriptor: &MatmulDispatchDescriptor,
) -> Result<f64, BenchError> {
    let mut kernel = MatmulKernel::<Metal>::new_mixed(combo.a_dtype, combo.b_dtype, combo.output_dtype)
        .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let a_byte_count = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
    let b_byte_count = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
    let d_byte_count = shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes();

    let (a_buffer, b_buffer, mut d_buffer) = match (
        context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
    ) {
        (Some(a), Some(b), Some(d)) => (a, b, d),
        _ => return Err(BenchError::BufferAllocation),
    };

    fill_buffer_random(&a_buffer, a_byte_count);
    fill_buffer_random(&b_buffer, b_byte_count);

    for iteration in 0..WARMUP_ITERATIONS {
        let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
        encode_and_run(context, &mut kernel, arguments, dispatch_descriptor)
            .map_err(|source| BenchError::Warmup { iteration, source: Box::new(source) })?;
    }

    let mut gpu_time_total_ms = 0.0;
    for iteration in 0..BENCHMARK_ITERATIONS {
        let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
        let gpu_ms = encode_and_run(context, &mut kernel, arguments, dispatch_descriptor)
            .map_err(|source| BenchError::Benchmark { iteration, source: Box::new(source) })?;
        gpu_time_total_ms += gpu_ms;
    }

    Ok(gpu_time_total_ms / BENCHMARK_ITERATIONS as f64)
}

pub fn benchmark_single(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    path_name: &str,
    dispatch_descriptor: &MatmulDispatchDescriptor,
) -> PerfResult {
    match run_benchmark(context, combo, shape, dispatch_descriptor) {
        Ok(duration_ms) => {
            let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            PerfResult {
                combo: format!("{combo}"),
                shape: format!("{shape}"),
                dispatch_path: path_name.to_owned(),
                duration_ms,
                gflops,
                status: "ok".into(),
                error: None,
            }
        },
        Err(error) => PerfResult {
            combo: format!("{combo}"),
            shape: format!("{shape}"),
            dispatch_path: path_name.to_owned(),
            duration_ms: 0.0,
            gflops: 0.0,
            status: "error".into(),
            error: Some(error.to_string()),
        },
    }
}
