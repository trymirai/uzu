use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
            CommandBufferPending, Context,
            kernel::matmul::{MatmulArguments, MatmulKernel, MatmulKernels},
        },
        metal::Metal,
    },
};

use super::{error::BenchError, output::PerfResult, shapes::TestShape};

type Ctx = <Metal as Backend>::Context;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

const WARMUP_ITERATIONS: usize = 10;
const BENCHMARK_ITERATIONS: usize = 30;

#[derive(Debug, Clone, Copy)]
pub enum DispatchPath {
    Gemv,
    Gemm,
    GemmMpp,
    GemmMppMxu,
    GemmMppNative,
}

impl DispatchPath {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gemv => "Gemv",
            Self::Gemm => "Gemm",
            Self::GemmMpp => "GemmMpp",
            Self::GemmMppMxu => "GemmMppMxu",
            Self::GemmMppNative => "GemmMppNative",
        }
    }

    pub fn available_paths(context: &Ctx) -> Vec<Self> {
        let mut paths = vec![Self::Gemv, Self::Gemm, Self::GemmMpp];
        if context.device_capabilities().supports_mxu {
            paths.push(Self::GemmMppMxu);
            paths.push(Self::GemmMppNative);
        }
        paths
    }
}

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
    kernel: &mut <<Metal as Backend>::Kernels as MatmulKernels>::MatmulKernel,
    dispatch_path: DispatchPath,
    shape: &TestShape,
    a_buffer: &Buf,
    b_buffer: &Buf,
    d_buffer: &mut Buf,
) -> Result<f64, BenchError> {
    let mut command_buffer = context.create_command_buffer().map_err(|_| BenchError::CommandBuffer)?.start_encoding();

    let arguments = MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        d: d_buffer,
        bias: None,
        batch: shape.batch as i32,
        input_dim: shape.input_dim as i32,
        output_dim: shape.output_dim as i32,
        leading_dimension_a: shape.input_dim as i32,
        leading_dimension_b: shape.input_dim as i32,
        leading_dimension_d: shape.output_dim as i32,
        transpose_b: true,
    };

    let encode_result = match dispatch_path {
        DispatchPath::Gemv => kernel.encode_gemv(context, &mut command_buffer, arguments),
        DispatchPath::Gemm => kernel.encode_gemm(context, &mut command_buffer, arguments),
        DispatchPath::GemmMpp => kernel.encode_gemm_mpp(context, &mut command_buffer, arguments),
        DispatchPath::GemmMppMxu => kernel.encode_gemm_mpp_mxu(context, &mut command_buffer, arguments),
        DispatchPath::GemmMppNative => kernel.encode_gemm_mpp_native(context, &mut command_buffer, arguments),
    };

    encode_result.map_err(|e| BenchError::Kernel(e.to_string()))?;

    let completed =
        command_buffer.end_encoding().submit().wait_until_completed().map_err(|_| BenchError::CommandBuffer)?;

    completed.gpu_execution_time().map(|duration| duration.as_secs_f64() * 1000.0).ok_or(BenchError::GpuTimestamps)
}

fn run_benchmark(
    context: &Ctx,
    data_type: DataType,
    dispatch_path: DispatchPath,
    shape: &TestShape,
) -> Result<f64, BenchError> {
    let mut kernel = <<Metal as Backend>::Kernels as MatmulKernels>::MatmulKernel::new(context, data_type)
        .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let elem_size = data_type.size_in_bytes();
    let a_byte_count = shape.batch * shape.input_dim * elem_size;
    let b_byte_count = shape.output_dim * shape.input_dim * elem_size;
    let d_byte_count = shape.batch * shape.output_dim * elem_size;

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
        encode_and_run(context, &mut kernel, dispatch_path, shape, &a_buffer, &b_buffer, &mut d_buffer).map_err(
            |source| BenchError::Warmup {
                iteration,
                source: Box::new(source),
            },
        )?;
    }

    let mut gpu_time_total_ms = 0.0;
    for iteration in 0..BENCHMARK_ITERATIONS {
        let gpu_ms = encode_and_run(context, &mut kernel, dispatch_path, shape, &a_buffer, &b_buffer, &mut d_buffer)
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
    dispatch_path: DispatchPath,
    shape: &TestShape,
) -> PerfResult {
    match run_benchmark(context, data_type, dispatch_path, shape) {
        Ok(duration_ms) => {
            let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            PerfResult {
                combo: format!("{data_type:?}"),
                shape: format!("{shape}"),
                dispatch_path: dispatch_path.name().to_owned(),
                duration_ms,
                gflops,
                status: "ok".into(),
                error: None,
            }
        },
        Err(error) => PerfResult {
            combo: format!("{data_type:?}"),
            shape: format!("{shape}"),
            dispatch_path: dispatch_path.name().to_owned(),
            duration_ms: 0.0,
            gflops: 0.0,
            status: "error".into(),
            error: Some(error.to_string()),
        },
    }
}
