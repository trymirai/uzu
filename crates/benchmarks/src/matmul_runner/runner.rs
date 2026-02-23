use std::time::Instant;

use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use uzu::{
    DataType,
    backends::{
        common::{Context, kernel::matmul::{MatmulArguments, MatmulDispatchDescriptor, MatmulKernel}},
        metal::{Metal, MetalContext, choose_dispatch_descriptor},
    },
};

use super::types::{MatmulBenchmarkResult, MatmulBenchmarkTask, MatmulDtypeCombo, MatmulShape};

fn parse_dtype(s: &str) -> Option<DataType> {
    match s {
        "i8" => Some(DataType::I8),
        "i16" => Some(DataType::I16),
        "i32" => Some(DataType::I32),
        "f16" => Some(DataType::F16),
        "bf16" => Some(DataType::BF16),
        "f32" => Some(DataType::F32),
        _ => None,
    }
}

pub struct MatmulRunner {
    task: MatmulBenchmarkTask,
}

impl MatmulRunner {
    pub fn new(task: MatmulBenchmarkTask) -> Self {
        Self { task }
    }

    pub fn run<F>(&self, mut progress: Option<F>) -> Result<Vec<MatmulBenchmarkResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        eprintln!("[matmul_bench] Creating Metal context...");
        let ctx = MetalContext::new().map_err(|e| format!("Metal context creation failed: {e}"))?;

        eprintln!("[matmul_bench] Metal device: {}", ctx.device.name());
        eprintln!("[matmul_bench] is_mpp_available: {}", ctx.is_mpp_available());

        let total_cases =
            self.task.combos.len() * self.task.shapes.len();
        let mut results = Vec::with_capacity(total_cases);
        let mut completed = 0usize;

        for combo in &self.task.combos {
            for shape in &self.task.shapes {
                let result = self.benchmark_single(&ctx, combo, shape);
                results.push(result);

                completed += 1;
                if let Some(ref mut progress_fn) = progress {
                    progress_fn(completed as f64 / total_cases as f64);
                }
            }
        }

        eprintln!("[matmul_bench] Completed {} benchmark cases", results.len());
        Ok(results)
    }

    fn benchmark_single(
        &self,
        ctx: &MetalContext,
        combo: &MatmulDtypeCombo,
        shape: &MatmulShape,
    ) -> MatmulBenchmarkResult {
        eprintln!("[matmul_bench] Starting {} shape {}", combo, shape);

        let a_dtype = match parse_dtype(&combo.a_dtype) {
            Some(dt) => dt,
            None => {
                let msg = format!("Unknown a_dtype: {}", combo.a_dtype);
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };
        let b_dtype = match parse_dtype(&combo.b_dtype) {
            Some(dt) => dt,
            None => {
                let msg = format!("Unknown b_dtype: {}", combo.b_dtype);
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };
        let output_dtype = match parse_dtype(&combo.output_dtype) {
            Some(dt) => dt,
            None => {
                let msg = format!("Unknown output_dtype: {}", combo.output_dtype);
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };

        let mut kernel = match MatmulKernel::<Metal>::new_mixed(a_dtype, b_dtype, output_dtype) {
            Ok(k) => k,
            Err(e) => {
                let msg = format!("MatmulKernel::new_mixed failed: {e}");
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };
        eprintln!("[matmul_bench] Kernel created for {}", combo);

        let m = shape.m;
        let n = shape.n;
        let k = shape.k;

        let a_bytes = m * k * a_dtype.size_in_bytes();
        let b_bytes = n * k * b_dtype.size_in_bytes();
        let d_bytes = m * n * output_dtype.size_in_bytes();

        let a_buf = ctx.device.new_buffer(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED);
        let b_buf = ctx.device.new_buffer(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED);
        let d_buf = ctx.device.new_buffer(d_bytes, MTLResourceOptions::STORAGE_MODE_SHARED);

        let (a_buf, b_buf, d_buf) = match (a_buf, b_buf, d_buf) {
            (Some(a), Some(b), Some(d)) => (a, b, d),
            _ => {
                let msg = "Failed to allocate Metal buffers".to_string();
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };

        fill_buffer_random(&a_buf, a_bytes);
        fill_buffer_random(&b_buf, b_bytes);

        let dispatch_dtype = a_dtype;
        let mut arguments = MatmulArguments {
            a: &a_buf,
            a_offset: 0,
            b: &b_buf,
            d: &d_buf,
            bias: None,
            batch: m as i32,
            input_dim: k as i32,
            output_dim: n as i32,
            lda: k as i32,
            ldb: k as i32,
            ldd: n as i32,
            batch_count: 1,
            transpose_b: true,
        };
        MatmulKernel::<Metal>::apply_batch_collapse(&mut arguments);

        let descriptor = match choose_dispatch_descriptor(ctx, dispatch_dtype, &arguments) {
            Ok(d) => {
                eprintln!("[matmul_bench] Dispatch descriptor: {:?}", std::mem::discriminant(&d));
                d
            },
            Err(e) => {
                let msg = format!("choose_dispatch_descriptor failed: {e}");
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            },
        };

        // Warmup
        eprintln!("[matmul_bench] Warmup ({} iterations)...", self.task.warmup_iterations);
        for i in 0..self.task.warmup_iterations {
            let warmup_args = MatmulArguments {
                a: &a_buf,
                a_offset: 0,
                b: &b_buf,
                d: &d_buf,
                bias: None,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: k as i32,
                ldd: n as i32,
                batch_count: 1,
                transpose_b: true,
            };
            if let Err(e) = encode_and_run(ctx, &mut kernel, warmup_args, &descriptor) {
                let msg = format!("Warmup iteration {} failed: {e}", i);
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            }
        }
        eprintln!("[matmul_bench] Warmup complete");

        // Benchmark
        eprintln!("[matmul_bench] Benchmarking ({} iterations)...", self.task.benchmark_iterations);
        let start = Instant::now();
        for i in 0..self.task.benchmark_iterations {
            let bench_args = MatmulArguments {
                a: &a_buf,
                a_offset: 0,
                b: &b_buf,
                d: &d_buf,
                bias: None,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: k as i32,
                ldd: n as i32,
                batch_count: 1,
                transpose_b: true,
            };
            if let Err(e) = encode_and_run(ctx, &mut kernel, bench_args, &descriptor) {
                let msg = format!("Benchmark iteration {} failed: {e}", i);
                eprintln!("[matmul_bench] ERROR: {msg}");
                return error_result(combo, shape, msg);
            }
        }
        let elapsed = start.elapsed();

        let duration_ms = elapsed.as_secs_f64() * 1000.0 / self.task.benchmark_iterations as f64;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (duration_ms / 1000.0) / 1e9;

        eprintln!(
            "[matmul_bench] {} {}: {:.1} GFLOPS ({:.2} ms/iter)",
            combo, shape, gflops, duration_ms
        );

        MatmulBenchmarkResult {
            combo: combo.clone(),
            shape: shape.clone(),
            duration_ms,
            gflops,
            status: "ok".into(),
            error_message: None,
        }
    }
}

fn encode_and_run(
    ctx: &MetalContext,
    kernel: &mut MatmulKernel<Metal>,
    arguments: MatmulArguments<Metal>,
    descriptor: &MatmulDispatchDescriptor,
) -> Result<(), String> {
    let cb = ctx
        .command_queue
        .command_buffer()
        .ok_or("Failed to create command buffer")?
        .to_owned();
    let enc = cb
        .new_compute_command_encoder()
        .ok_or("Failed to create compute encoder")?;

    kernel
        .encode_with_descriptor(ctx, arguments, descriptor, &enc)
        .map_err(|e| format!("encode failed: {e}"))?;

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

fn fill_buffer_random(buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>, byte_count: usize) {
    let ptr = buf.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn error_result(combo: &MatmulDtypeCombo, shape: &MatmulShape, msg: String) -> MatmulBenchmarkResult {
    MatmulBenchmarkResult {
        combo: combo.clone(),
        shape: shape.clone(),
        duration_ms: 0.0,
        gflops: 0.0,
        status: "error".into(),
        error_message: Some(msg),
    }
}
