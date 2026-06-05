//! Focused GEMV profiling harness for `gpu_trace` (no criterion, no MLX).
//!
//! Hammers a single GEMV specialization in sustained batches so the GPU stays
//! at its maximum performance state for the whole capture window — unlike a
//! criterion run, whose startup/setup leaves the GPU idle (Minimum clock) and
//! produces empty/garbage counter traces.
//!
//! Usage:
//!   gemv_profile <quant|fp> <m> <k> <n> [gs] [bits] [batches]
//! Defaults to the quant decode shape uzu trails MLX on: quant 1 14336 4096 64 4.
//!
//! Run under the tracer:
//!   gpu-trace run --gpu-counters -- target/release/examples/gemv_profile quant 1 14336 4096

#![cfg(all(metal_backend, target_os = "macos"))]

use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            AllocationType, Backend, Context, Encoder,
            gpu_types::QuantizationMode,
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{Metal, MetalContext},
    },
};
use half::bf16;

const DISPATCHES_PER_BUFFER: usize = 64;

fn alloc<T>(
    ctx: &MetalContext,
    elems: usize,
) -> backend_uzu::backends::common::Allocation<Metal> {
    ctx.create_allocation(elems * std::mem::size_of::<T>(), AllocationType::Global).expect("allocation")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(|s| s.as_str()).unwrap_or("quant");
    let m: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(14336);
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let gs: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
    let bits: u32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(4);
    let batches: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(120);

    let ctx = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &ctx,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");

    // Buffers (uninitialized content is fine — GEMV timing is data-independent).
    let x = alloc::<bf16>(&ctx, m * k);
    let mut y = alloc::<bf16>(&ctx, m * n);
    let mode = if bits == 4 {
        QuantizationMode::U4
    } else {
        QuantizationMode::I8
    };
    let num_groups_k = k.div_ceil(gs as usize);
    let w = alloc::<u32>(&ctx, n * k * bits as usize / 32);
    let scales = alloc::<bf16>(&ctx, n * num_groups_k);
    let biases = alloc::<bf16>(&ctx, n * num_groups_k);

    let is_fp = kind == "fp";
    eprintln!("profiling {kind} m={m} k={k} n={n} gs={gs} bits={bits} x {batches}*{DISPATCHES_PER_BUFFER} dispatches");
    for _ in 0..batches {
        let mut encoder = Encoder::<Metal>::new(&ctx).expect("encoder");
        for _ in 0..DISPATCHES_PER_BUFFER {
            let b = if is_fp {
                MatmulB::FullPrecision {
                    b: &w,
                }
            } else {
                MatmulB::ScaleBiasDequant {
                    b: &w,
                    scales: &scales,
                    biases: &biases,
                    mode,
                    group_size: gs,
                }
            };
            let args = MatmulArguments {
                a: &x,
                a_offset: 0,
                b,
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut y,
                d_transform: MatmulDOps::none(),
                m: m as u32,
                n: n as u32,
                k: k as u32,
            };
            kernel.encode(args, &mut encoder).expect("encode");
        }
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    }
    eprintln!("done");
}
