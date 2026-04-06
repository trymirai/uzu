use half::bf16;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::runtime::ProtocolObject;

use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        metal::Metal,
    },
};

type Ctx = <Metal as Backend>::Context;

const WARMUP_ITERATIONS: usize = 5;
const BENCHMARK_ITERATIONS: usize = 20;

fn dispatch_path(
    context: &Ctx,
    m: u32,
) -> String {
    use crate::backends::metal::metal_extensions::DeviceExt;
    let gemv_threshold = super::max_gemv_batch_threshold();
    if m <= gemv_threshold {
        "gemv".into()
    } else if context.device.supports_mxu() {
        "gemm_mpp".into()
    } else {
        "gemm".into()
    }
}

#[derive(Debug, Clone)]
pub struct MatmulBenchmarkResult {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub dispatch_path: String,
    pub duration_ms: f64,
    pub gflops: f64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MatmulCorrectnessResult {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub dispatch_path: String,
    pub max_diff: f32,
    pub tolerance: f32,
    pub passed: bool,
    pub error: Option<String>,
}

fn fill_buffer_deterministic(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_count: usize,
) {
    let pointer = buffer.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(pointer, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn generate_test_data(
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let a: Vec<bf16> = (0..(m * k)).map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06)).collect();
    let b: Vec<bf16> = (0..(n * k)).map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15)).collect();
    (a, b)
}

fn reference_matmul(
    a: &[bf16],
    b: &[bf16],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<bf16> {
    let mut result = vec![bf16::from_f32(0.0); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i].to_f32() * b[col * k + i].to_f32();
            }
            result[row * n + col] = bf16::from_f32(sum);
        }
    }
    result
}

const BENCHMARK_SHAPES: &[(u32, u32, u32)] = &[
    // Small batch (GEMV path)
    (1, 896, 896),
    (1, 4096, 4096),
    (1, 4096, 14336),
    // Medium batch
    (16, 896, 896),
    (16, 4096, 4096),
    (16, 4096, 14336),
    // Large batch (GEMM/GEMM-MPP path)
    (64, 896, 896),
    (64, 4096, 4096),
    (64, 4096, 14336),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 14336),
];

const CORRECTNESS_SHAPES: &[(u32, u32, u32)] = &[
    (1, 128, 128),
    (1, 896, 896),
    (4, 896, 4864),
    (16, 2048, 8192),
    (32, 4096, 4096),
    (64, 4096, 14336),
    (128, 896, 896),
    (256, 4096, 4096),
    // Non-aligned dimensions
    (3, 127, 255),
    (7, 1000, 1000),
    (33, 4096, 4096),
    (65, 4096, 14336),
];

pub fn run_benchmarks(context: &Ctx) -> Vec<MatmulBenchmarkResult> {
    let mut results = Vec::new();

    for &(m, k, n) in BENCHMARK_SHAPES {
        let result = benchmark_single(context, DataType::BF16, m, k, n);
        results.push(result);
    }

    results
}

pub fn run_correctness_tests(context: &Ctx) -> Vec<MatmulCorrectnessResult> {
    let mut results = Vec::new();

    for &(m, k, n) in CORRECTNESS_SHAPES {
        let result = correctness_single(context, m, k, n);
        results.push(result);
    }

    results
}

fn benchmark_single(
    context: &Ctx,
    data_type: DataType,
    m: u32,
    k: u32,
    n: u32,
) -> MatmulBenchmarkResult {
    let run = || -> Result<f64, String> {
        let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)
            .map_err(|e| e.to_string())?;

        let elem_size = data_type.size_in_bytes();
        let a_bytes = m as usize * k as usize * elem_size;
        let b_bytes = n as usize * k as usize * elem_size;
        let d_bytes = m as usize * n as usize * elem_size;

        let a_buf = context
            .device
            .new_buffer(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to allocate A buffer")?;
        let b_buf = context
            .device
            .new_buffer(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to allocate B buffer")?;
        let mut d_buf = context
            .device
            .new_buffer(d_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to allocate D buffer")?;

        fill_buffer_deterministic(&a_buf, a_bytes);
        fill_buffer_deterministic(&b_buf, b_bytes);

        for _ in 0..WARMUP_ITERATIONS {
            let mut encoder = Encoder::<Metal>::new(context).map_err(|e| e.to_string())?;
            kernel.encode(
                context,
                MatmulArguments {
                    a: &a_buf,
                    a_offset: 0,
                    b: &b_buf,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: &mut d_buf,
                    batch_dim: m,
                    input_dim: k,
                    output_dim: n,
                },
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        }

        let mut total_ms = 0.0;
        for _ in 0..BENCHMARK_ITERATIONS {
            let mut encoder = Encoder::<Metal>::new(context).map_err(|e| e.to_string())?;
            kernel.encode(
                context,
                MatmulArguments {
                    a: &a_buf,
                    a_offset: 0,
                    b: &b_buf,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: &mut d_buf,
                    batch_dim: m,
                    input_dim: k,
                    output_dim: n,
                },
                &mut encoder,
            );
            let completed = encoder.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
            let gpu_ms = completed.gpu_execution_time().map(|d| d.as_secs_f64() * 1000.0).ok_or("no GPU timestamps")?;
            total_ms += gpu_ms;
        }

        Ok(total_ms / BENCHMARK_ITERATIONS as f64)
    };

    let path = dispatch_path(context, m);
    match run() {
        Ok(duration_ms) => {
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            MatmulBenchmarkResult {
                m,
                k,
                n,
                dispatch_path: path,
                duration_ms,
                gflops,
                status: "ok".into(),
                error: None,
            }
        },
        Err(e) => MatmulBenchmarkResult {
            m,
            k,
            n,
            dispatch_path: path,
            duration_ms: 0.0,
            gflops: 0.0,
            status: "error".into(),
            error: Some(e),
        },
    }
}

fn correctness_single(
    context: &Ctx,
    m: u32,
    k: u32,
    n: u32,
) -> MatmulCorrectnessResult {
    let run = || -> Result<(f32, f32), String> {
        let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, DataType::BF16)
            .map_err(|e| e.to_string())?;

        let (a_data, b_data) = generate_test_data(m as usize, k as usize, n as usize);
        let reference = reference_matmul(&a_data, &b_data, m as usize, k as usize, n as usize);

        let a_buf = context
            .device
            .new_buffer_with_data(bytemuck::cast_slice(&a_data), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to create A buffer")?;
        let b_buf = context
            .device
            .new_buffer_with_data(bytemuck::cast_slice(&b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to create B buffer")?;
        let mut d_buf = context
            .device
            .new_buffer(m as usize * n as usize * core::mem::size_of::<bf16>(), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("failed to create D buffer")?;

        let mut encoder = Encoder::<Metal>::new(context).map_err(|e| e.to_string())?;
        kernel.encode(
            context,
            MatmulArguments {
                a: &a_buf,
                a_offset: 0,
                b: &b_buf,
                ab_scale: 1.0,
                c: MatmulArgumentC::None,
                d: &mut d_buf,
                batch_dim: m,
                input_dim: k,
                output_dim: n,
            },
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;

        let metal_result: Vec<bf16> = unsafe {
            let ptr = d_buf.contents().as_ptr() as *const bf16;
            std::slice::from_raw_parts(ptr, m as usize * n as usize).to_vec()
        };

        let tolerance = 0.01 * (k as f32 / 1024.0).sqrt() * (1.0 + (m as f32).log2() * 0.02);

        let mut max_diff: f32 = 0.0;
        for (i, (&metal_val, &ref_val)) in metal_result.iter().zip(reference.iter()).enumerate() {
            let diff = (metal_val.to_f32() - ref_val.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                return Err(format!(
                    "mismatch at index {i}: metal={} ref={} diff={diff} tolerance={tolerance}",
                    metal_val.to_f32(),
                    ref_val.to_f32(),
                ));
            }
        }

        Ok((max_diff, tolerance))
    };

    let path = dispatch_path(context, m);
    match run() {
        Ok((max_diff, tolerance)) => MatmulCorrectnessResult {
            m,
            k,
            n,
            dispatch_path: path,
            max_diff,
            tolerance,
            passed: true,
            error: None,
        },
        Err(e) => MatmulCorrectnessResult {
            m,
            k,
            n,
            dispatch_path: path,
            max_diff: f32::NAN,
            tolerance: 0.0,
            passed: false,
            error: Some(e),
        },
    }
}
