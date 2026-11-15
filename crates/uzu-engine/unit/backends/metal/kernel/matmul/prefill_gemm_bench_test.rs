//! Standalone micro-benchmark for the 8-bit prefill GEMM on this machine.
//!
//! The LFM3 prefill is dominated by the quantized GEMM (~94% of the GPU time),
//! so kernel experiments are measured here first, without loading a model.
//!
//! Run it directly on the built test binary (so the numbers are not measured
//! right after a compile):
//!
//! ```text
//! cargo test -p uzu-engine --release --no-run
//! <test-bin> lfm3_prefill_gemm_microbench --nocapture
//! ```
//!
//! Each run prints one JSON line per shape with the GPU time and TFLOPS.

use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{
                    MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                },
            },
        },
        metal::{Metal, context::MetalContext, kernel::matmul::gemm::GemmEngine},
    },
    data_type::DataTypeRepr,
    tests::{helpers::create_buffer, matmul::{QuantBuffers, QuantInput, quant_arguments}},
};

/// LFM3-1.7B/2B-VL decoder shapes: d_model 2048, mlp hidden 7680.
/// `m` is a prefill batch of 1024 tokens.
fn lfm3_prefill_shapes() -> &'static [(u32, u32, u32)] {
    &[
        // gate + up projections
        (1024, 2048, 7680),
        // down projection
        (1024, 7680, 2048),
        // attention q projection (16 heads x 128)
        (1024, 2048, 2048),
    ]
}

fn time_gemm(
    context: &MetalContext,
    input: &QuantInput<bf16>,
    engine: GemmEngine,
    iterations: u32,
) -> f64 {
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");

    // Warm-up: pipeline/first-use setup is not part of the measurement.
    let mut command_buffer = context.create_command_buffer(Some("gemm_warmup"), None).unwrap();
    matmul
        .encode_with_gemm_engine(quant_arguments(&mut buffers, input), engine, &mut command_buffer)
        .expect("encode warmup");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let mut command_buffer = context.create_command_buffer(Some("gemm_bench"), None).unwrap();
    for _ in 0..iterations {
        matmul
            .encode_with_gemm_engine(quant_arguments(&mut buffers, input), engine, &mut command_buffer)
            .expect("encode gemm");
    }
    let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    let seconds = completed.gpu_execution_time().unwrap().as_secs_f64();
    seconds / f64::from(iterations)
}

#[uzu_test]
fn lfm3_prefill_gemm_microbench() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let iterations: u32 = std::env::var("GEMM_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let engines: &[(&str, GemmEngine)] = if context.supports_mxu {
        &[("simdgroup", GemmEngine::Simdgroup), ("mxu", GemmEngine::Mxu)]
    } else {
        &[("simdgroup", GemmEngine::Simdgroup)]
    };

    for &(m, k, n) in lfm3_prefill_shapes() {
        // 8-bit, group size 64, scale+bias: the LFM3 weight format.
        let input = QuantInput::<bf16>::new(m, k, n, 64, 8, QuantizationMethod::ScaleBias, 42);
        for &(label, engine) in engines {
            let seconds = time_gemm(&context, &input, engine, iterations);
            let tflops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n) / seconds / 1e12;
            println!(
                "{{\"bench\":\"prefill_gemm\",\"engine\":\"{label}\",\"m\":{m},\"k\":{k},\"n\":{n},\"ms\":{:.3},\"tflops\":{:.2}}}",
                seconds * 1e3,
                tflops
            );
        }
    }
}

/// Upper bound for the same shapes with unquantized bf16 weights: no
/// dequantization, just a vectorized copy into threadgroup memory.
fn time_gemm_full_precision(context: &MetalContext, m: u32, k: u32, n: u32, iterations: u32) -> f64 {
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    let a = create_buffer::<Metal, bf16>(context, m as usize * k as usize);
    let b = create_buffer::<Metal, bf16>(context, n as usize * k as usize);
    let mut d = create_buffer::<Metal, bf16>(context, m as usize * n as usize);

    let mut run = |matmul: &mut <<Metal as Backend>::Kernels as Kernels>::MatmulKernel,
                   command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding| {
        for _ in 0..iterations {
            matmul
                .encode_with_gemm_engine(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &a,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: &b,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        output: MatmulOutput::new(&mut d, MatmulDOps::none()),
                        gather_indices: None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        m,
                        n,
                        k,
                    },
                    GemmEngine::Simdgroup,
                    command_buffer,
                )
                .expect("encode full precision gemm");
        }
    };

    let mut command_buffer = context.create_command_buffer(Some("fp_warmup"), None).unwrap();
    run(&mut matmul, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let mut command_buffer = context.create_command_buffer(Some("fp_bench"), None).unwrap();
    run(&mut matmul, &mut command_buffer);
    let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    completed.gpu_execution_time().unwrap().as_secs_f64() / f64::from(iterations)
}

#[uzu_test]
fn lfm3_prefill_gemm_full_precision_microbench() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let iterations: u32 = std::env::var("GEMM_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    for &(m, k, n) in lfm3_prefill_shapes() {
        let seconds = time_gemm_full_precision(&context, m, k, n, iterations);
        let tflops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n) / seconds / 1e12;
        println!(
            "{{\"bench\":\"prefill_gemm_fp\",\"engine\":\"simdgroup\",\"m\":{m},\"k\":{k},\"n\":{n},\"ms\":{:.3},\"tflops\":{:.2}}}",
            seconds * 1e3,
            tflops
        );
    }
}
