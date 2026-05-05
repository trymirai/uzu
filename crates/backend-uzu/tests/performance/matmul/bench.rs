#![cfg(metal_backend)]

use backend_uzu::{
    DataType,
    backends::{
        common::{
            AllocationType, Backend, Buffer, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        metal::Metal,
    },
};

use super::{error::BenchError, output::PerfResult, shapes::TestShape};

type Ctx = <Metal as Backend>::Context;

const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;

fn fill_buffer_random(
    buffer: &<Metal as Backend>::Buffer,
    byte_count: usize,
) {
    let slice = unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn encode_and_run(
    context: &Ctx,
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    shape: &TestShape,
    a_allocation: &backend_uzu::backends::common::Allocation<Metal>,
    b_allocation: &backend_uzu::backends::common::Allocation<Metal>,
    d_allocation: &mut backend_uzu::backends::common::Allocation<Metal>,
) -> Result<f64, BenchError> {
    let mut encoder = Encoder::new(context).map_err(|_| BenchError::CommandBuffer)?;

    kernel.encode(
        MatmulArguments {
            a: a_allocation,
            a_offset: 0,
            b: b_allocation,
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: d_allocation,
            batch_dim: shape.batch as u32,
            input_dim: shape.input_dim as u32,
            output_dim: shape.output_dim as u32,
        },
        &mut encoder,
    );

    let completed = encoder.end_encoding().submit().wait_until_completed().map_err(|_| BenchError::CommandBuffer)?;

    Ok(completed.gpu_execution_time().as_secs_f64() * 1000.0)
}

fn run_benchmark(
    context: &Ctx,
    data_type: DataType,
    shape: &TestShape,
) -> Result<f64, BenchError> {
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)
        .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let elem_size = data_type.size_in_bytes();
    let a_byte_count = shape.batch * shape.input_dim * elem_size;
    let b_byte_count = shape.output_dim * shape.input_dim * elem_size;
    let d_byte_count = shape.batch * shape.output_dim * elem_size;

    let a_allocation =
        context.create_allocation(a_byte_count, AllocationType::Global).map_err(|_| BenchError::BufferAllocation)?;
    let b_allocation =
        context.create_allocation(b_byte_count, AllocationType::Global).map_err(|_| BenchError::BufferAllocation)?;
    let mut d_allocation =
        context.create_allocation(d_byte_count, AllocationType::Global).map_err(|_| BenchError::BufferAllocation)?;

    let (a_raw, _) = a_allocation.as_buffer_range();
    let (b_raw, _) = b_allocation.as_buffer_range();
    fill_buffer_random(a_raw, a_byte_count);
    fill_buffer_random(b_raw, b_byte_count);

    for iteration in 0..WARMUP_ITERATIONS {
        encode_and_run(context, &mut kernel, shape, &a_allocation, &b_allocation, &mut d_allocation).map_err(
            |source| BenchError::Warmup {
                iteration,
                source: Box::new(source),
            },
        )?;
    }

    let mut gpu_time_total_ms = 0.0;
    for iteration in 0..BENCHMARK_ITERATIONS {
        let gpu_ms = encode_and_run(context, &mut kernel, shape, &a_allocation, &b_allocation, &mut d_allocation)
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
) -> PerfResult {
    match run_benchmark(context, data_type, shape) {
        Ok(duration_ms) => {
            let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            PerfResult {
                combo: format!("{data_type:?}"),
                shape: format!("{shape}"),
                dispatch_path: "auto".to_owned(),
                duration_ms,
                gflops,
                status: "ok".into(),
                error: None,
            }
        },
        Err(error) => PerfResult {
            combo: format!("{data_type:?}"),
            shape: format!("{shape}"),
            dispatch_path: "auto".to_owned(),
            duration_ms: 0.0,
            gflops: 0.0,
            status: "error".into(),
            error: Some(error.to_string()),
        },
    }
}
