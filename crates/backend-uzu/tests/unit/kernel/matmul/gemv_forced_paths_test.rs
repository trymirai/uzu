#![cfg(metal_backend)]

//! Parity of every forced GEMV dispatch path against the CPU reference.
//!
//! The batch-1 dispatch heuristics (fp `GpuDeviceTier` rules and the
//! quantized tile table) may select any valid `GemvDispatchPath`, so each
//! template specialization the selector can reach must produce correct
//! output — a performance sweep alone would happily tune toward a broken one.
//! Shapes cover tiny k (256), tiny n (256), unaligned k, and a mid-size
//! decode shape, for fp bf16 plus both production quant modes
//! (q4 gs64 scale+bias, q4 gs32 scale+zero-point with and without RHT).

use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{GemvDispatchPath, Metal, MetalContext},
    },
};
use half::bf16;

use crate::{
    common::{
        helpers::{alloc_allocation_with_data, allocation_to_vec},
        matmul::{
            Case, QuantBuffers, QuantInput, Shape, cpu_reference, deterministic_input, quant_arguments,
            run_metal_gemv_path,
        },
    },
    uzu_test,
};

/// The dispatch paths instantiated in the metallib (selector-reachable set;
/// see the CONSTRAINT block in gemv.metal).
/// - fp: 8 simdgroups, R in {1, 4}, any k-split (KS = 1 only on unaligned k).
/// - quant (bf16 IO): full SG x R grid, KS = 1 only.
/// - RHT outputs: the Hadamard rotation covers one 32-row block per
///   threadgroup, so only 32-row tiles (SG8xR4, SG4xR8) are valid.
fn gemv_paths(
    quant_only: bool,
    rht: bool,
) -> Vec<GemvDispatchPath> {
    if rht {
        return vec![
            GemvDispatchPath {
                k_split: 1,
                results_per_simdgroup: 4,
                num_simdgroups: 8,
            },
            GemvDispatchPath {
                k_split: 1,
                results_per_simdgroup: 8,
                num_simdgroups: 4,
            },
        ];
    }
    let mut paths = Vec::new();
    let simdgroups: &[u32] = if quant_only {
        &[2, 4, 8]
    } else {
        &[8]
    };
    for &num_simdgroups in simdgroups {
        let k_splits: &[u32] = if quant_only {
            &[1]
        } else {
            &[1, 2, 4, 8]
        };
        let results: &[u32] = if quant_only {
            &[1, 2, 4, 8]
        } else {
            &[1, 4]
        };
        for &k_split in k_splits {
            if !num_simdgroups.is_multiple_of(k_split) {
                continue;
            }
            for &results_per_simdgroup in results {
                paths.push(GemvDispatchPath {
                    k_split,
                    results_per_simdgroup,
                    num_simdgroups,
                });
            }
        }
    }
    paths
}

fn assert_close(
    label: &str,
    reference: &[bf16],
    actual: &[bf16],
    rel: f32,
    abs: f32,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    let errors = reference
        .iter()
        .zip(actual)
        .filter(|(r, a)| {
            let (r, a) = (r.to_f32(), a.to_f32());
            (r - a).abs() > abs + rel * r.abs()
        })
        .count();
    assert_eq!(errors, 0, "{label}: {errors}/{} mismatches", reference.len());
}

#[uzu_test]
fn fp_forced_paths_match_cpu_reference() {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    // tiny k / tiny n / unaligned k / mid decode shape (m = 1 -> GEMV).
    // The selector only splits k on block-aligned inputs, so the unaligned
    // shape covers the KS = 1 column only.
    for shape in
        [Shape::new(1, 256, 1536), Shape::new(1, 1536, 256), Shape::new(1, 1000, 2048), Shape::new(1, 2560, 3072)]
    {
        let input = deterministic_input::<bf16>(Case::new(shape));
        let reference = cpu_reference::<bf16>(&input);
        let aligned = shape.k % 128 == 0;
        for path in gemv_paths(false, false) {
            if !aligned && path.k_split != 1 {
                continue;
            }
            let actual = run_metal_gemv_path::<bf16>(&context, &mut kernel, &input, path);
            assert_close(&format!("fp {shape:?} {path:?}"), &reference, &actual, 0.05, 0.05);
        }
    }
}

fn quant_case(
    context: &MetalContext,
    k: usize,
    n: usize,
    group_size: u32,
    method: QuantizationMethod,
    rht: bool,
) {
    let input = QuantInput::<bf16>::new(1, k, n, group_size, 4, method, 7);
    let rht_signs: Option<Vec<i32>> = rht.then(|| {
        (0..n as usize)
            .map(|i| {
                if i % 2 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    });

    // CPU reference (applies RHT through the CPU kernel when requested).
    let cpu_context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut cpu_buffers = QuantBuffers::<Cpu, bf16>::allocate(&cpu_context, &input);
    let cpu_rht = rht_signs.as_ref().map(|s| alloc_allocation_with_data::<Cpu, i32>(&cpu_context, s));
    let mut cpu_kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &cpu_context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("Cpu MatmulKernel");
    let mut cpu_encoder = Encoder::<Cpu>::new(&cpu_context).expect("cpu encoder");
    let mut cpu_args = quant_arguments(&mut cpu_buffers, &input);
    if let Some(rht) = cpu_rht.as_ref() {
        cpu_args.d_transform = MatmulDOps {
            ab_scale: 1.0,
            accumulate: false,
            bias: None,
            rht_factors: Some(rht),
        };
    }
    cpu_kernel.encode(cpu_args, &mut cpu_encoder).expect("cpu encode");
    cpu_encoder.end_encoding().submit().wait_until_completed().unwrap();
    let reference = allocation_to_vec::<Cpu, bf16>(&cpu_buffers.y);

    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("Metal MatmulKernel");
    for path in gemv_paths(true, rht) {
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(context, &input);
        let metal_rht = rht_signs.as_ref().map(|s| alloc_allocation_with_data::<Metal, i32>(context, s));
        let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
        let mut args = quant_arguments(&mut buffers, &input);
        if let Some(rht) = metal_rht.as_ref() {
            args.d_transform = MatmulDOps {
                ab_scale: 1.0,
                accumulate: false,
                bias: None,
                rht_factors: Some(rht),
            };
        }
        kernel.encode_gemv_dispatch_path(args, path, &mut encoder).expect("gemv forced path encode");
        encoder.end_encoding().submit().wait_until_completed().unwrap();
        let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);
        assert_close(
            &format!("quant k{k} n{n} gs{group_size} {method:?} rht={rht} {path:?}"),
            &reference,
            &actual,
            0.05,
            0.6,
        );
    }
}

#[uzu_test]
fn quant_forced_paths_match_cpu_reference() {
    let context = MetalContext::new().expect("Metal context");
    // Gemma4-style: q4 gs64 scale+bias.
    quant_case(&context, 256, 1536, 64, QuantizationMethod::ScaleBias, false);
    quant_case(&context, 1536, 256, 64, QuantizationMethod::ScaleBias, false);
    // Qwen3.5-style: q4 gs32 scale+zero-point, with and without output RHT.
    quant_case(&context, 992, 2048, 32, QuantizationMethod::ScaleZeroPoint, true);
    quant_case(&context, 2560, 3072, 32, QuantizationMethod::ScaleZeroPoint, true);
    quant_case(&context, 1024, 2048, 32, QuantizationMethod::ScaleZeroPoint, false);
}
