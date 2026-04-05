#![cfg(metal_backend)]

use half::bf16;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::QuantizationMode,
            kernel::quant_matmul::{
                ForceKernel, QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
                QuantizedMatmulType,
            },
        },
        metal::Metal,
    },
};

use super::{error::BenchError, output::PerfResult, shapes::TestShape};

type Ctx = <Metal as Backend>::Context;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

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
    context: &Ctx,
    kernel: &QuantizedMatmulKernelEncodable<Metal>,
    shape: &TestShape,
    x_buffer: &Buf,
    w_buffer: &Buf,
    s_buffer: &Buf,
    b_buffer: &Buf,
    y_buffer: &mut Buf,
) -> Result<f64, BenchError> {
    let mut encoder = Encoder::new(context).map_err(|_| BenchError::CommandBuffer)?;

    kernel
        .encode(
            &mut encoder,
            QuantizedMatmulArguments {
                a_buffer: x_buffer,
                a_offset: 0,
                b_buffer: w_buffer,
                scales_buffer: s_buffer,
                zero_points_or_biases_buffer: b_buffer,
                output_buffer: y_buffer,
                batch: shape.batch,
                input_dim: shape.input_dim,
                output_dim: shape.output_dim,
                quantization_type: QuantizedMatmulType::Mlx,
            },
        )
        .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let completed = encoder.end_encoding().submit().wait_until_completed().map_err(|_| BenchError::CommandBuffer)?;
    completed.gpu_execution_time().map(|duration| duration.as_secs_f64() * 1000.0).ok_or(BenchError::GpuTimestamps)
}

fn run_benchmark(
    context: &Ctx,
    data_type: DataType,
    shape: &TestShape,
    group_size: usize,
    bits: usize,
    mode: QuantizationMode,
    force_kernel: ForceKernel,
) -> Result<f64, BenchError> {
    let kernel = QuantizedMatmulKernelEncodable::<Metal>::new(
        context,
        QuantizedMatmulConfiguration {
            data_type,
            group_size,
            input_dim: shape.input_dim,
            output_dim: shape.output_dim,
            mode,
            quantization_type: QuantizedMatmulType::Mlx,
            weights_transposed: true,
            force_kernel,
        },
    )
    .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let elem_size = data_type.size_in_bytes();
    let pack_divisor = if bits == 4 {
        2
    } else {
        1
    };

    let x_byte_count = shape.batch * shape.input_dim * elem_size;
    let w_byte_count = shape.output_dim * shape.input_dim / pack_divisor;
    let num_groups = (shape.input_dim + group_size - 1) / group_size;
    let s_byte_count = shape.output_dim * num_groups * elem_size;
    let b_byte_count = shape.output_dim * num_groups * elem_size;
    let y_byte_count = shape.batch * shape.output_dim * elem_size;

    let x_buffer = context
        .device
        .new_buffer(x_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(BenchError::BufferAllocation)?;
    let w_buffer = context
        .device
        .new_buffer(w_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(BenchError::BufferAllocation)?;
    let s_buffer = context
        .device
        .new_buffer(s_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(BenchError::BufferAllocation)?;
    let b_buffer = context
        .device
        .new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(BenchError::BufferAllocation)?;
    let mut y_buffer = context
        .device
        .new_buffer(y_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(BenchError::BufferAllocation)?;

    fill_buffer_random(&x_buffer, x_byte_count);
    fill_buffer_random(&w_buffer, w_byte_count);

    {
        let s_ptr = s_buffer.contents().as_ptr() as *mut bf16;
        let s_slice = unsafe { std::slice::from_raw_parts_mut(s_ptr, shape.output_dim * num_groups) };
        for v in s_slice.iter_mut() {
            *v = bf16::from_f32(1.0);
        }
        let b_ptr = b_buffer.contents().as_ptr() as *mut bf16;
        let b_slice = unsafe { std::slice::from_raw_parts_mut(b_ptr, shape.output_dim * num_groups) };
        for v in b_slice.iter_mut() {
            *v = bf16::from_f32(0.0);
        }
    }

    for iteration in 0..WARMUP_ITERATIONS {
        encode_and_run(context, &kernel, shape, &x_buffer, &w_buffer, &s_buffer, &b_buffer, &mut y_buffer).map_err(
            |source| BenchError::Warmup {
                iteration,
                source: Box::new(source),
            },
        )?;
    }

    let mut gpu_time_total_ms = 0.0;
    for iteration in 0..BENCHMARK_ITERATIONS {
        let gpu_ms = encode_and_run(context, &kernel, shape, &x_buffer, &w_buffer, &s_buffer, &b_buffer, &mut y_buffer)
            .map_err(|source| BenchError::Benchmark {
                iteration,
                source: Box::new(source),
            })?;
        gpu_time_total_ms += gpu_ms;
    }

    Ok(gpu_time_total_ms / BENCHMARK_ITERATIONS as f64)
}

pub fn benchmark_single(
    context: &Ctx,
    data_type: DataType,
    shape: &TestShape,
    group_size: usize,
    bits: usize,
    mode: QuantizationMode,
    force_kernel: ForceKernel,
) -> PerfResult {
    let dispatch_name = match force_kernel {
        ForceKernel::Auto => "Auto",
        ForceKernel::QmvFast => "QmvFast",
        ForceKernel::QmmTransposedSmall => "QmmSmall",
        ForceKernel::QmmTransposedSmallSplitK => "QmmSmallSplitK",
        ForceKernel::QmmTransposed64x64 => "Qmm64x64",
    };

    match run_benchmark(context, data_type, shape, group_size, bits, mode, force_kernel) {
        Ok(duration_ms) => {
            let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            PerfResult {
                combo: format!("Q{bits} g{group_size}"),
                shape: format!("{shape}"),
                dispatch_path: dispatch_name.to_owned(),
                duration_ms,
                gflops,
                status: "ok".into(),
                error: None,
            }
        },
        Err(error) => PerfResult {
            combo: format!("Q{bits} g{group_size}"),
            shape: format!("{shape}"),
            dispatch_path: dispatch_name.to_owned(),
            duration_ms: 0.0,
            gflops: 0.0,
            status: "error".into(),
            error: Some(error.to_string()),
        },
    }
}
