use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use crate::backends::metal::{Metal, MetalContext};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
        matmul::{QuantBuffers, QuantInput, run_quant_cpu},
    },
};

struct Input<T: ArrayElement + Float> {
    a: Box<[T]>,
    b: Box<[T]>,
    m: usize,
    k: usize,
    n: usize,
    ids: Option<Box<[u32]>>,
    soft_cap: Option<f32>,
}

fn get_test_data<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
) -> (Input<T>, Vec<T>) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();

    let input = Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        m,
        k,
        n,
        ids: None,
        soft_cap: None,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

// Encode one GEMV (dense, or a per-row B-row gather when `gather_indices` is set) and copy out.
fn run_gemv<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    a: impl BufferRef<Backend = B>,
    b: MatmulB<impl BufferRef<Backend = B>>,
    gather_indices: Option<impl BufferRef<Backend = B>>,
    m: usize,
    n_out: usize,
    k: usize,
    soft_cap: Option<f32>,
) -> Vec<T> {
    let mut d = create_buffer::<B, T>(context, m * n_out);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context, T::data_type(), T::data_type(), T::data_type())
            .expect("MatmulKernel");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: a,
                    offset: 0,
                },
                b,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d,
                d_transform: MatmulDOps {
                    soft_cap,
                    ..MatmulDOps::none()
                },
                gather_indices,
                m: m as u32,
                n: n_out as u32,
                k: k as u32,
            },
            &mut command_buffer,
        )
        .expect("encode failed");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, T>(&d)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let a = create_buffer_with_data::<B, T>(&context, &input.a);
    let weights = create_buffer_with_data::<B, T>(&context, &input.b);
    let ids = input.ids.as_ref().map(|ids| create_buffer_with_data::<B, u32>(&context, ids));
    run_gemv::<B, T>(
        &context,
        &a,
        MatmulB::FullPrecision {
            b: &weights,
        },
        ids.as_ref(),
        input.m,
        input.n,
        input.k,
        input.soft_cap,
    )
}

fn test<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    let (input, expected) = get_test_data::<T>(m, k, n);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float(&expected, &output, eps, &format!("backend {}", std::any::type_name::<B>()));
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
#[case::gemm_m16(16, 256, 96)]
#[case::gemm_m70_unaligned(70, 200, 72)]
fn gemv_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<bf16>(m, k, n, 0.1);
}

#[cfg(backend = "metal")]
#[rstest]
#[test_attr(uzu_test)]
#[case::w4_zero_point(4, QuantizationMethod::ScaleZeroPoint)]
#[case::w8_bias(8, QuantizationMethod::ScaleBias)]
fn group_major_gemv_bf16(
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(1, 256, 72, 32, bits, method, 0).with_group_output();
    let reference = run_quant_cpu::<bf16>(&input);

    let buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let actual = run_gemv::<Metal, bf16>(
        &context,
        &buffers.x,
        buffers.matmul_b(&input),
        None::<&<Metal as Backend>::GlobalBuffer>,
        1,
        input.n as usize,
        input.k as usize,
        None,
    );

    assert_eq_float(&reference, &actual, 0.05, &format!("GroupOutput GEMV W{bits} {method:?}"));
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
#[case::gemm_m16(16, 256, 96)]
#[case::gemm_m70_unaligned(70, 200, 72)]
fn gemv_f32(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<f32>(m, k, n, 0.01);
}

fn assert_gather<T: ArrayElement + Float + Debug + Display>(
    dense: &[T],
    gather: &[T],
    ids: &[u32],
    m: usize,
    vocab: usize,
    ids_per_row: usize,
    eps: f32,
    name: &str,
) {
    let mut expected = vec![T::from(0.0).unwrap(); m * ids_per_row];
    for r in 0..m {
        for c in 0..ids_per_row {
            expected[r * ids_per_row + c] = dense[r * vocab + ids[r * ids_per_row + c] as usize];
        }
    }
    assert_eq_float(&expected, gather, eps, &format!("gather vs dense ({name})"));
}

macro_rules! check_gather {
    ($m:expr, $vocab:expr, $ids:expr, $ids_per_row:expr, $eps:expr, |$B:ident| $run:block) => {{
        {
            #[allow(non_camel_case_types)]
            type $B = Cpu;
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, "Cpu");
        }
        for_each_non_cpu_backend!(|$B| {
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, std::any::type_name::<$B>());
        });
    }};
}

fn fp_gather_case<T: ArrayElement + Float + Debug + Display>(
    soft_cap: Option<f32>,
    eps: f32,
) {
    let (m, k, vocab, ids_per_row) = (4usize, 128usize, 256usize, 8usize);
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let weights: Vec<T> = (0..vocab * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();

    // Dense (`n = vocab`, no ids) and gather (`n = ids_per_row`, ids) share `a`/`weights`/soft-cap.
    let make = |n: usize, ids: Option<Box<[u32]>>| Input {
        a: a.clone().into_boxed_slice(),
        b: weights.clone().into_boxed_slice(),
        m,
        k,
        n,
        ids,
        soft_cap,
    };
    let dense_input = make(vocab, None);
    let gather_input = make(ids_per_row, Some(ids.clone().into_boxed_slice()));

    check_gather!(m, vocab, ids, ids_per_row, eps, |B| {
        (get_output::<T, B>(&dense_input), get_output::<T, B>(&gather_input))
    });
}

fn quant_gather_case(
    input: QuantInput<bf16>,
    eps: f32,
) {
    let (m, k, vocab, ids_per_row) = (input.m as usize, input.k as usize, input.n as usize, 8);
    let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();
    // K_SPLIT == 1 keeps dense and gathered accumulation order identical.
    check_gather!(m, vocab, ids, ids_per_row, eps, |B| {
        let context = <B as Backend>::Context::new().expect("context");
        let buffers = QuantBuffers::<B, bf16>::allocate(&context, &input);
        let ids_alloc = create_buffer_with_data::<B, u32>(&context, &ids);
        let b = || buffers.matmul_b(&input);
        (
            run_gemv::<B, bf16>(&context, &buffers.x, b(), None::<&<B as Backend>::GlobalBuffer>, m, vocab, k, None),
            run_gemv::<B, bf16>(&context, &buffers.x, b(), Some(&ids_alloc), m, ids_per_row, k, None),
        )
    });
}

#[uzu_test]
fn gemv_gather() {
    // Full precision: one call per dtype (generic over T, so bf16/f32 can't be a runtime loop).
    for soft_cap in [None, Some(15.0)] {
        fp_gather_case::<bf16>(soft_cap, 0.1);
        fp_gather_case::<f32>(soft_cap, 0.01);
    }
    // Quantized (bf16, per bits/method) — inline, since it isn't type-generic.
    for (bits, method, signed_codes) in [
        (4, QuantizationMethod::ScaleBias, false),
        (4, QuantizationMethod::ScaleZeroPoint, false),
        (4, QuantizationMethod::ScaleZeroPoint, true),
        (4, QuantizationMethod::ScaleSymmetric, false),
        (8, QuantizationMethod::ScaleZeroPoint, false),
    ] {
        let mut input = QuantInput::new(8, 128, 64, 32, bits, method, 0x5EED);
        if signed_codes {
            input = input.with_signed_weight_codes();
        }
        quant_gather_case(input, 0.05);
    }
    quant_gather_case(
        QuantInput::new(8, 96, 66, 32, 4, QuantizationMethod::ScaleZeroPoint, 0x5EED).with_group_output(),
        0.5,
    );
}

// Quantized matmul through each backend's MatmulKernel at decode, verification and small-prefill batch
// sizes: on AMD the single-row GEMV (m = 1, W8) and the WMMA qmv (W4, m >= 2: several 16-row groups, a ragged
// last column tile, 16- and 32-column tiles from n = 8192, every prologue, signed codes, group-major scales).
#[rstest]
#[test_attr(uzu_test)]
#[case::w4_zp_g32_m1(1, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m4(4, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m7(7, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m8(8, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m16(16, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g64_m16(16, 96, 4, 64, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m20_ragged(20, 100, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m40(40, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g64_m5_wide(5, 8200, 4, 64, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::w4_zp_g32_m9_signed(9, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, true, false)]
#[case::w4_zp_g32_m6_group_output(6, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, true)]
#[case::w4_sym_g32_m6(6, 96, 4, 32, QuantizationMethod::ScaleSymmetric, false, false)]
#[case::w8_sym_g32_m5(5, 96, 8, 32, QuantizationMethod::ScaleSymmetric, false, false)]
#[case::w8_sym_g32_m16(16, 96, 8, 32, QuantizationMethod::ScaleSymmetric, false, false)]
#[case::w4_bias_g32_m16(16, 96, 4, 32, QuantizationMethod::ScaleBias, false, false)]
#[case::w4_bias_g64_m3(3, 96, 4, 64, QuantizationMethod::ScaleBias, false, false)]
fn quantized_gemv_bf16(
    #[case] m: u32,
    #[case] n: u32,
    #[case] bits: u32,
    #[case] group: u32,
    #[case] method: QuantizationMethod,
    #[case] signed_codes: bool,
    #[case] group_output: bool,
) {
    let mut input = QuantInput::<bf16>::new(m, 1024, n, group, bits, method, 0);
    if signed_codes {
        input = input.with_signed_weight_codes();
    }
    if group_output {
        input = input.with_group_output();
    }
    let reference = run_quant_cpu::<bf16>(&input);

    for_each_non_cpu_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("context");
        let buffers = QuantBuffers::<B, bf16>::allocate(&context, &input);
        route::reset::<B>();
        let actual = run_gemv::<B, bf16>(
            &context,
            &buffers.x,
            buffers.matmul_b(&input),
            None::<&<B as Backend>::GlobalBuffer>,
            m as usize,
            input.n as usize,
            input.k as usize,
            None,
        );
        let expected_route = if bits == 4 {
            route::qmv(m)
        } else {
            route::MSL_GEMV
        };
        route::check::<B>(expected_route, &format!("W{bits} G{group} {method:?} m={m} n={n}"));
        assert_eq!(reference.len(), actual.len());
        // GEMM stages dequantized weights in bf16, so errors scale with the output magnitude, not the element
        let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
        for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
            let (expected, got) = (f32::from(expected), f32::from(got));
            assert!(
                (expected - got).abs() <= 0.1 + 0.01 * magnitude,
                "{} W{bits} G{group} {method:?} m={m} n={n}: index {index} expected {expected} got {got}",
                std::any::type_name::<B>()
            );
        }
    });
}

// Route checks (AMDGPU). The backend picks a matmul kernel by shape and environment switches, and a native
// kernel that stops being selected would pass the conformance tests through a fallback kernel; these tests
// assert the kernel that ran (`take_routes`: "GemvW4", "GemmW4", "QmvWmma", "QmvWmma int8", an MSL GEMM tiling
// `Tile...`, an MSL GEMV `GemvSpecialization`) against the route the switches select.
mod route {
    pub const NATIVE_GEMV: &str = "GemvW4";
    pub const NATIVE_GEMM: &str = "GemmW4";
    pub const QMV: &str = "\"QmvWmma\"";
    pub const QMV_INT8: &str = "QmvWmma int8";
    pub const MSL_GEMM: &str = "Tile";
    pub const MSL_GEMV: &str = "GemvSpecialization";

    #[cfg(backend = "amdgpu")]
    use crate::backends::amdgpu::kernel::matmul::{gemm_w4, gemv_w4, qmv_wmma};

    /// Native M = 1 GEMV (`UZU_AMDGPU_GEMV_NATIVE`) for W4 with group-major metadata and N a multiple of 4.
    pub fn gemv_group_output(n: u32) -> &'static str {
        #[cfg(backend = "amdgpu")]
        if gemv_w4::enabled() && n % 4 == 0 {
            return NATIVE_GEMV;
        }
        let _ = n;
        MSL_GEMV
    }

    /// Prefill-sized W4 (`UZU_AMDGPU_GEMM_NATIVE`).
    pub fn gemm() -> &'static str {
        #[cfg(backend = "amdgpu")]
        if gemm_w4::enabled() {
            return NATIVE_GEMM;
        }
        MSL_GEMM
    }

    /// W4 with bf16 activations and 2 <= M <= 47 (`UZU_AMDGPU_QMV_WMMA_MIN_M` / `_MAX_M`): WMMA qmv, else MSL GEMV.
    pub fn qmv(m: u32) -> &'static str {
        #[cfg(backend = "amdgpu")]
        {
            let (min_m, max_m) = qmv_wmma::m_range();
            if (min_m..=max_m).contains(&m) {
                return QMV;
            }
        }
        let _ = m;
        MSL_GEMV
    }

    /// int8 activations are only taken with `UZU_AMDGPU_A8` on (default).
    pub fn int8_enabled() -> bool {
        #[cfg(backend = "amdgpu")]
        return qmv_wmma::a8_enabled();
        #[cfg(not(backend = "amdgpu"))]
        true
    }

    /// Clears the routes recorded on this thread.
    pub fn reset<B: crate::backends::common::Backend>() {
        #[cfg(backend = "amdgpu")]
        if B::NAME == "amdgpu" {
            crate::backends::amdgpu::kernel::matmul::take_routes();
        }
    }

    /// Asserts that every matmul encoded on this thread since the last check took `expected`.
    pub fn check<B: crate::backends::common::Backend>(
        expected: &str,
        context: &str,
    ) {
        #[cfg(backend = "amdgpu")]
        if B::NAME == "amdgpu" {
            let routes = crate::backends::amdgpu::kernel::matmul::take_routes();
            assert!(
                !routes.is_empty() && routes.iter().all(|taken| taken.contains(expected)),
                "{context}: expected the {expected} route, took {routes:?}"
            );
        }
        let _ = (expected, context);
    }
}

fn run_quant_with_output_ops<B: Backend>(
    input: &QuantInput<bf16>,
    rht_factors: Option<&[i32]>,
    bias: Option<&[bf16]>,
    ab_scale: f32,
) -> Vec<bf16> {
    let context = <B as Backend>::Context::new().expect("context");
    let buffers = QuantBuffers::<B, bf16>::allocate(&context, input);
    let factors = rht_factors.map(|factors| create_buffer_with_data::<B, i32>(&context, factors));
    let bias = bias.map(|bias| create_buffer_with_data::<B, bf16>(&context, bias));
    let mut d = create_buffer::<B, bf16>(&context, (input.m * input.n) as usize);
    let mut kernel = <B::Kernels as Kernels>::MatmulKernel::new(
        &context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::BF16,
    )
    .expect("MatmulKernel");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel
        .encode(
            MatmulArguments {
                // int8 activations when the input carries them (QuantInput::with_prepared_a)
                a: match &input.prepared_a {
                    Some(prepared) => MatmulA::Int8Symmetric {
                        values: buffers.prepared_a.as_ref().expect("int8 activations"),
                        scales: buffers.prepared_a_scales.as_ref().expect("int8 activation scales"),
                        group_sums: buffers.prepared_a_group_sums.as_ref(),
                        scale_group_size: prepared.quantization.scale_group_size(),
                        code_layout: prepared.quantization.code_layout(),
                    },
                    None => MatmulA::FullPrecision {
                        values: &buffers.x,
                        offset: 0,
                    },
                },
                b: buffers.matmul_b(input),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d,
                d_transform: MatmulDOps {
                    ab_scale,
                    bias: bias.as_ref(),
                    rht_factors: factors.as_ref(),
                    ..MatmulDOps::none()
                },
                gather_indices: None::<&<B as Backend>::GlobalBuffer>,
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut command_buffer,
        )
        .expect("encode");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, bf16>(&d)
}

// Quantized matmul with the output random Hadamard transform (+ bias, ab_scale) of Mirai's RHT models, at the
// batch sizes where AMD fuses it into the GEMV tile (m = 1) or the WMMA qmv epilogue (m >= 2).
#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, 1.0)]
#[case::m1_bias_scale(1, 256, 4, 32, QuantizationMethod::ScaleZeroPoint, true, 0.5)]
#[case::m4(4, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, false, 1.0)]
#[case::m16_bias(16, 256, 4, 64, QuantizationMethod::ScaleZeroPoint, true, 1.0)]
#[case::m20_bias_scale(20, 96, 4, 32, QuantizationMethod::ScaleZeroPoint, true, 0.5)]
#[case::m3_scale_bias_quant(3, 96, 4, 32, QuantizationMethod::ScaleBias, true, 1.0)]
fn quantized_matmul_output_rht_bf16(
    #[case] m: u32,
    #[case] n: u32,
    #[case] bits: u32,
    #[case] group: u32,
    #[case] method: QuantizationMethod,
    #[case] with_bias: bool,
    #[case] ab_scale: f32,
) {
    let input = QuantInput::<bf16>::new(m, 1024, n, group, bits, method, 3);
    let factors: Vec<i32> = (0..n)
        .map(|i| {
            if (i * 7 + i / 3) % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();
    let bias: Option<Vec<bf16>> =
        with_bias.then(|| (0..n).map(|i| bf16::from_f32(((i % 11) as f32) * 0.05 - 0.25)).collect());
    let reference = run_quant_with_output_ops::<Cpu>(&input, Some(&factors), bias.as_deref(), ab_scale);
    // M = 1 here has output-major metadata (the MSL GEMV), larger M the WMMA qmv
    let expected_route = if m == 1 {
        route::MSL_GEMV
    } else {
        route::qmv(m)
    };
    let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
    for_each_non_cpu_backend!(|B| {
        route::reset::<B>();
        let actual = run_quant_with_output_ops::<B>(&input, Some(&factors), bias.as_deref(), ab_scale);
        route::check::<B>(expected_route, &format!("output RHT m={m} n={n} {method:?}"));
        for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
            let (expected, got) = (f32::from(expected), f32::from(got));
            assert!(
                (expected - got).abs() <= 0.05 + 0.01 * magnitude,
                "{} m={m} n={n} {method:?} bias={with_bias} scale={ab_scale}: index {index} expected {expected} got {got}",
                B::NAME
            );
        }
    });
}

// One activation row against W4 weights with group-major scales and zero points ([groups, N], the layout of
// uzu's quantized checkpoints), over K with several 1024-wide steps and a partial last one, N with a partial
// last tile, and the output operations (scale, RHT, bias): on AMD the native GEMV (gemv_w4.clcpp), which also
// falls back to the MSL GEMV for N not a multiple of 4.
#[rstest]
#[test_attr(uzu_test)]
#[case::zp_g32_n64_k1024(64, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::zp_g32_n72_k3104_tails(72, 3104, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::zp_g64_n96_k2048_signed(96, 2048, 64, QuantizationMethod::ScaleZeroPoint, true, false)]
#[case::sym_g32_n128_k2048(128, 2048, 32, QuantizationMethod::ScaleSymmetric, false, false)]
#[case::zp_g32_n100_k1024_tail_tile(100, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::zp_g32_n102_k1024_msl_fallback(102, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, false)]
#[case::zp_g32_n256_k2048_output_ops(256, 2048, 32, QuantizationMethod::ScaleZeroPoint, false, true)]
#[case::sym_g64_n96_k1024_output_ops(96, 1024, 64, QuantizationMethod::ScaleSymmetric, false, true)]
#[case::zp_g32_n65536_k1024_tall_output_ops(65536, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, true)]
fn quantized_gemv_group_output_bf16(
    #[case] n: u32,
    #[case] k: u32,
    #[case] group: u32,
    #[case] method: QuantizationMethod,
    #[case] signed_codes: bool,
    #[case] output_ops: bool,
) {
    let mut input = QuantInput::<bf16>::new(1, k, n, group, 4, method, 21).with_group_output();
    if signed_codes {
        input = input.with_signed_weight_codes();
    }
    let factors: Option<Vec<i32>> = output_ops.then(|| {
        (0..n)
            .map(|i| {
                if (i * 3 + i / 5) % 4 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect()
    });
    let bias: Option<Vec<bf16>> =
        output_ops.then(|| (0..n).map(|i| bf16::from_f32(((i % 11) as f32) * 0.03 - 0.15)).collect());
    let ab_scale = if output_ops {
        0.5
    } else {
        1.0
    };
    let reference = run_quant_with_output_ops::<Cpu>(&input, factors.as_deref(), bias.as_deref(), ab_scale);
    let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
    for_each_non_cpu_backend!(|B| {
        route::reset::<B>();
        let actual = run_quant_with_output_ops::<B>(&input, factors.as_deref(), bias.as_deref(), ab_scale);
        route::check::<B>(route::gemv_group_output(n), &format!("group-output GEMV n={n} k={k}"));
        for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
            let (expected, got) = (f32::from(expected), f32::from(got));
            assert!(
                (expected - got).abs() <= 0.02 + 0.01 * magnitude,
                "{} n={n} k={k} {method:?} G{group} output_ops={output_ops}: index {index} expected {expected} got {got}",
                B::NAME
            );
        }
    });
}

// Prefill-sized quantized matmuls (M >= 48) with W4 weights in both metadata layouts, ragged M and N tiles and
// the output operations: on AMD the native WMMA GEMM (gemm_w4.clcpp).
#[rstest]
#[test_attr(uzu_test)]
#[case::zp_g32_m48_n128(48, 128, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, true, false)]
#[case::zp_g32_m130_n160_ragged(130, 160, 1024, 32, QuantizationMethod::ScaleZeroPoint, false, true, false)]
#[case::zp_g64_m256_n256_output_major(256, 256, 2048, 64, QuantizationMethod::ScaleZeroPoint, false, false, false)]
#[case::sym_g32_m200_n96_signed(200, 96, 1024, 32, QuantizationMethod::ScaleSymmetric, true, true, false)]
#[case::zp_g32_m131_n256_output_ops(131, 256, 2048, 32, QuantizationMethod::ScaleZeroPoint, false, true, true)]
#[case::sym_g64_m64_n192_output_ops(64, 192, 1024, 64, QuantizationMethod::ScaleSymmetric, false, true, true)]
fn quantized_gemm_w4_bf16(
    #[case] m: u32,
    #[case] n: u32,
    #[case] k: u32,
    #[case] group: u32,
    #[case] method: QuantizationMethod,
    #[case] signed_codes: bool,
    #[case] group_output: bool,
    #[case] output_ops: bool,
) {
    let mut input = QuantInput::<bf16>::new(m, k, n, group, 4, method, 33);
    if group_output {
        input = input.with_group_output();
    }
    if signed_codes {
        input = input.with_signed_weight_codes();
    }
    let factors: Option<Vec<i32>> = output_ops.then(|| {
        (0..n)
            .map(|i| {
                if (i * 5 + i / 3) % 4 == 1 {
                    -1
                } else {
                    1
                }
            })
            .collect()
    });
    let bias: Option<Vec<bf16>> =
        output_ops.then(|| (0..n).map(|i| bf16::from_f32(((i % 7) as f32) * 0.05 - 0.15)).collect());
    let ab_scale = if output_ops {
        0.5
    } else {
        1.0
    };
    let reference = run_quant_with_output_ops::<Cpu>(&input, factors.as_deref(), bias.as_deref(), ab_scale);
    let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
    for_each_non_cpu_backend!(|B| {
        route::reset::<B>();
        let actual = run_quant_with_output_ops::<B>(&input, factors.as_deref(), bias.as_deref(), ab_scale);
        route::check::<B>(route::gemm(), &format!("GEMM m={m} n={n} k={k}"));
        for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
            let (expected, got) = (f32::from(expected), f32::from(got));
            assert!(
                (expected - got).abs() <= 0.02 + 0.01 * magnitude,
                "{} m={m} n={n} k={k} {method:?} G{group} output_ops={output_ops}: index {index} expected {expected} got {got}",
                B::NAME
            );
        }
    });
}

// Qwen3.5-9B production shapes against the CPU reference (slow: tens of GMAC on the CPU, hence ignored): the
// layout of the checkpoint (W4, group-major scales and zero points), output RHT where the model has it, and the
// batch sizes of decode (M = 1), speculation trees (M = 15-16, bf16 and int8 activations) and prefill chunks
// (M = 331, 200, 64); K up to 32768 (the DFlash feature projection) and N up to the 248k-row readout. The route
// is asserted, so each case runs the kernel it is meant for.
// `cargo test ... quantized_matmul_production_shapes -- --ignored` (one test over all shapes: `#[ignore]` does
// not reach rstest cases generated through `test_attr(uzu_test)`).
#[uzu_test]
#[ignore]
fn quantized_matmul_production_shapes() {
    // (name, m, n, k, group, output RHT, int8 activations)
    const CASES: [(&str, u32, u32, u32, u32, bool, bool); 9] = [
        ("readout_m1", 1, 248320, 4096, 32, false, false),
        ("down_m1", 1, 4096, 12288, 32, true, false),
        ("dflash_projection_m1", 1, 4096, 32768, 64, true, false),
        ("gate_up_m16", 16, 24576, 4096, 32, true, false),
        ("gate_up_m16_int8", 16, 24576, 4096, 32, true, true),
        ("readout_m15", 15, 248320, 4096, 32, false, false),
        ("down_m331", 331, 4096, 12288, 32, true, false),
        ("qkv_m200", 200, 12352, 4096, 32, true, false),
        ("dflash_projection_m64", 64, 4096, 32768, 64, true, false),
    ];
    for (name, m, n, k, group, rht, int8) in CASES {
        production_shape_case(name, m, n, k, group, rht, int8);
    }
}

fn production_shape_case(
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    group: u32,
    rht: bool,
    int8: bool,
) {
    use crate::backends::common::kernel::activation_transform::ACTIVATION_SCALE_GROUP_SIZE;

    if int8 && !route::int8_enabled() {
        return;
    }
    let mut input =
        QuantInput::<bf16>::new(m, k, n, group, 4, QuantizationMethod::ScaleZeroPoint, 47).with_group_output();
    if int8 {
        input = input.with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);
    }
    let factors: Option<Vec<i32>> = rht.then(|| {
        (0..n)
            .map(|i| {
                if (i * 7 + i / 11) % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect()
    });
    let reference = run_quant_with_output_ops::<Cpu>(&input, factors.as_deref(), None, 1.0);
    let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
    let expected_route = if int8 {
        route::QMV_INT8
    } else if m == 1 {
        route::gemv_group_output(n)
    } else if m >= 48 {
        route::gemm()
    } else {
        route::qmv(m)
    };
    for_each_non_cpu_backend!(|B| {
        route::reset::<B>();
        let actual = run_quant_with_output_ops::<B>(&input, factors.as_deref(), None, 1.0);
        route::check::<B>(expected_route, &format!("production {name}"));
        let mut worst = 0.0f32;
        for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
            let (expected, got) = (f32::from(expected), f32::from(got));
            worst = worst.max((expected - got).abs());
            assert!(
                (expected - got).abs() <= 0.02 + 0.01 * magnitude,
                "{} production {name} (m={m} n={n} k={k} G{group} rht={rht} int8={int8}): index {index} expected {expected} got {got}",
                B::NAME
            );
        }
        eprintln!("{} {name}: max |error| {worst:.4} of max |value| {magnitude:.3}", B::NAME);
    });
}

// Quantized matmul on int8 activations (uzu's A8 path: one scale per 128 values of a row, GroupedByNibble codes),
// as the engine quantizes them for the batch sizes where a backend asks for it: on AMD the WMMA iu8 qmv (Metal
// takes int8 activations only on its MXU GEMM path).
#[cfg(backend = "amdgpu")]
#[rstest]
#[test_attr(uzu_test)]
#[case::zp_g32_m2(2, 96, 32, QuantizationMethod::ScaleZeroPoint, false)]
#[case::zp_g32_m4(4, 96, 32, QuantizationMethod::ScaleZeroPoint, false)]
#[case::zp_g64_m16(16, 96, 64, QuantizationMethod::ScaleZeroPoint, false)]
#[case::zp_g32_m20_ragged(20, 100, 32, QuantizationMethod::ScaleZeroPoint, false)]
#[case::zp_g32_m7_wide(7, 8200, 32, QuantizationMethod::ScaleZeroPoint, false)]
#[case::sym_g32_m5(5, 96, 32, QuantizationMethod::ScaleSymmetric, false)]
#[case::zp_g32_m6_rht(6, 256, 32, QuantizationMethod::ScaleZeroPoint, true)]
#[case::zp_g64_m16_rht(16, 96, 64, QuantizationMethod::ScaleZeroPoint, true)]
fn quantized_matmul_a8_bf16(
    #[case] m: u32,
    #[case] n: u32,
    #[case] group: u32,
    #[case] method: QuantizationMethod,
    #[case] output_ops: bool,
) {
    use crate::backends::common::kernel::activation_transform::ACTIVATION_SCALE_GROUP_SIZE;

    // the int8 kernel is out of the route with UZU_AMDGPU_A8=0
    if !route::int8_enabled() {
        return;
    }
    let input =
        QuantInput::<bf16>::new(m, 1024, n, group, 4, method, 11).with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);
    let factors: Option<Vec<i32>> = output_ops.then(|| {
        (0..n)
            .map(|i| {
                if (i * 5 + i / 7) % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect()
    });
    let bias: Option<Vec<bf16>> =
        output_ops.then(|| (0..n).map(|i| bf16::from_f32(((i % 9) as f32) * 0.04 - 0.2)).collect());
    let ab_scale = if output_ops {
        0.75
    } else {
        1.0
    };
    let reference = run_quant_with_output_ops::<Cpu>(&input, factors.as_deref(), bias.as_deref(), ab_scale);
    let magnitude = reference.iter().map(|value| f32::from(*value).abs()).fold(0.0f32, f32::max);
    route::reset::<crate::backends::amdgpu::Amdgpu>();
    let actual = run_quant_with_output_ops::<crate::backends::amdgpu::Amdgpu>(
        &input,
        factors.as_deref(),
        bias.as_deref(),
        ab_scale,
    );
    route::check::<crate::backends::amdgpu::Amdgpu>(route::QMV_INT8, &format!("int8 m={m} n={n}"));
    for (index, (&expected, &got)) in reference.iter().zip(actual.iter()).enumerate() {
        let (expected, got) = (f32::from(expected), f32::from(got));
        assert!(
            (expected - got).abs() <= 0.05 + 0.01 * magnitude,
            "a8 m={m} n={n} {method:?} G{group} output_ops={output_ops}: index {index} expected {expected} got {got}"
        );
    }
}

// GPU time of the quantized matmul at Qwen3.5-9B shapes (W4 zero-point, group 32) for decode and
// tree-verification batch sizes. `cargo test ... perf_quantized_gemv_qwen9b -- --ignored --nocapture`.
#[uzu_test]
#[ignore]
fn perf_quantized_gemv_qwen9b() {
    use crate::{backends::common::CommandBufferCompleted, data_type::DataType, tests::matmul::quant_arguments};

    for_each_non_cpu_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("context");
        for (label, n, k) in [("gate_up", 24576u32, 4096u32), ("down", 4096, 12288)] {
            for m in [1u32, 2, 4, 8, 16, 32, 64, 256] {
                let input = QuantInput::<bf16>::new(m, k, n, 32, 4, QuantizationMethod::ScaleZeroPoint, 7);
                let mut buffers = QuantBuffers::<B, bf16>::allocate(&context, &input);
                let mut kernel = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
                    &context,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                )
                .expect("MatmulKernel");
                let mut times = Vec::new();
                for _ in 0..12 {
                    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                    kernel.encode(quant_arguments(&mut buffers, &input), &mut command_buffer).expect("encode");
                    let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                    times.push(completed.gpu_execution_time().as_secs_f64() * 1e3);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_ms = times[times.len() / 2];
                // codes + bf16 scales + 4-bit zero points
                let weight_bytes = (n as f64) * (k as f64) * (0.5 + 2.0 / 32.0 + 0.5 / 32.0);
                eprintln!(
                    "{} {label:8} m={m:2}: {median_ms:7.3} ms, weights {:6.1} GB/s",
                    std::any::type_name::<B>(),
                    weight_bytes / (median_ms * 1e-3) / 1e9
                );
            }
        }
    });
}

// GPU time of all quantized matmuls of one Qwen3.5-9B token (24 DeltaNet and 8 attention layers, readout),
// encoded in one command buffer as the engine does, with the output RHT of the RHT layers; m = 1 is a decode
// step, m = 16 the verification of a 16-token speculation tree. Weights are shared between layers of the same
// shape (each matrix is far larger than L2). `... perf_token_matmuls_qwen9b -- --ignored`.
// Format knobs (to price the quantization format): UZU_PERF_QUANT=zp|sym, UZU_PERF_RHT=1|0,
// UZU_PERF_LAYOUT=output_group|group_output (Mirai-M checkpoints are group_output), UZU_PERF_M=1,4,16.
#[uzu_test]
#[ignore]
fn perf_token_matmuls_qwen9b() {
    use crate::{backends::common::CommandBufferCompleted, data_type::DataType};

    let knob = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_string());
    let method = match knob("UZU_PERF_QUANT", "zp").as_str() {
        "sym" => QuantizationMethod::ScaleSymmetric,
        _ => QuantizationMethod::ScaleZeroPoint,
    };
    let rht = knob("UZU_PERF_RHT", "1") != "0";
    let group_output = knob("UZU_PERF_LAYOUT", "output_group") == "group_output";
    let m_filter: Option<Vec<u32>> = std::env::var("UZU_PERF_M")
        .ok()
        .map(|value| value.split(',').map(|m| m.parse().expect("UZU_PERF_M")).collect());
    let zero_point_bytes = if matches!(method, QuantizationMethod::ScaleZeroPoint) {
        0.5 / 32.0
    } else {
        0.0
    };

    const DELTA_NET: [(u32, u32); 4] = [(12352, 4096), (4096, 4096), (24576, 4096), (4096, 12288)];
    const ATTENTION: [(u32, u32); 4] = [(10240, 4096), (4096, 4096), (24576, 4096), (4096, 12288)];
    const READOUT: (u32, u32) = (248320, 4096);
    for_each_non_cpu_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("context");
        for (m, a8) in [(1u32, false), (4, false), (4, true), (16, false), (16, true)] {
            // int8 activations (A8) only where the backend asks for them
            if a8 && !cfg!(backend = "amdgpu") || m_filter.as_ref().is_some_and(|ms| !ms.contains(&m)) {
                continue;
            }
            let mut shapes: Vec<(u32, u32)> = DELTA_NET.iter().chain(ATTENTION.iter()).copied().collect();
            shapes.push(READOUT);
            shapes.sort_unstable();
            shapes.dedup();
            let inputs: Vec<_> = shapes
                .iter()
                .map(|&(n, k)| {
                    let input = QuantInput::<bf16>::new(m, k, n, 32, 4, method, 7);
                    let input = if group_output {
                        input.with_group_output()
                    } else {
                        input
                    };
                    if a8 {
                        input.with_prepared_a(
                            crate::backends::common::kernel::activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                            None,
                        )
                    } else {
                        input
                    }
                })
                .collect();
            let buffers: Vec<_> =
                inputs.iter().map(|input| QuantBuffers::<B, bf16>::allocate(&context, input)).collect();
            let factors: Vec<_> = shapes
                .iter()
                .map(|&(n, _)| create_buffer_with_data::<B, i32>(&context, &vec![1i32; n as usize]))
                .collect();
            let mut outputs: Vec<_> =
                shapes.iter().map(|&(n, _)| create_buffer::<B, bf16>(&context, (m * n) as usize)).collect();
            let mut kernel = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
            )
            .expect("MatmulKernel");
            let sequence: Vec<(u32, u32)> = std::iter::repeat_n(DELTA_NET, 24)
                .flatten()
                .chain(std::iter::repeat_n(ATTENTION, 8).flatten())
                .chain(std::iter::once(READOUT))
                .collect();
            let weight_bytes: f64 =
                sequence.iter().map(|&(n, k)| (n as f64) * (k as f64) * (0.5 + 2.0 / 32.0 + zero_point_bytes)).sum();
            // UZU_PERF_AB_NATIVE=1 (AMDGPU): each iteration encodes the token with the native GEMV route and with
            // the MSL GEMV, so both see the same thermal state
            let routes: Vec<Option<bool>> = if cfg!(backend = "amdgpu") && knob("UZU_PERF_AB_NATIVE", "0") == "1" {
                vec![Some(true), Some(false)]
            } else {
                vec![None]
            };
            let mut route_times = vec![Vec::new(); routes.len()];
            for _ in 0..5 {
                for (route_index, route) in routes.iter().enumerate() {
                    #[cfg(backend = "amdgpu")]
                    if let Some(enabled) = route {
                        crate::backends::amdgpu::kernel::matmul::gemv_w4::set_enabled(*enabled);
                    }
                    let _ = route;
                    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                    for &(n, k) in &sequence {
                        let index = shapes.binary_search(&(n, k)).expect("shape");
                        let input = &inputs[index];
                        kernel
                            .encode(
                                MatmulArguments {
                                    a: match &input.prepared_a {
                                        Some(prepared) => MatmulA::Int8Symmetric {
                                            values: buffers[index].prepared_a.as_ref().expect("int8 activations"),
                                            scales: buffers[index].prepared_a_scales.as_ref().expect("scales"),
                                            group_sums: None,
                                            scale_group_size: prepared.quantization.scale_group_size(),
                                            code_layout: prepared.quantization.code_layout(),
                                        },
                                        None => MatmulA::FullPrecision {
                                            values: &buffers[index].x,
                                            offset: 0,
                                        },
                                    },
                                    b: buffers[index].matmul_b(input),
                                    b_leading_dimension: None,
                                    b_transpose: true,
                                    d: &mut outputs[index],
                                    d_transform: MatmulDOps {
                                        rht_factors: (rht && (n, k) != READOUT).then_some(&factors[index]),
                                        ..MatmulDOps::none()
                                    },
                                    gather_indices: None::<&<B as Backend>::GlobalBuffer>,
                                    m,
                                    n,
                                    k,
                                },
                                &mut command_buffer,
                            )
                            .expect("encode");
                    }
                    let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                    route_times[route_index].push(completed.gpu_execution_time().as_secs_f64() * 1e3);
                }
            }
            for (route, mut times) in routes.iter().zip(route_times) {
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_ms = times[times.len() / 2];
                eprintln!(
                    "{} token matmuls {method:?} rht={rht} group_output={group_output}{} m={m:2}{}: {} matmuls, {:.2} GB, {median_ms:7.2} ms, {:5.1} GB/s",
                    B::NAME,
                    match route {
                        Some(true) => " native-gemv",
                        Some(false) => " msl-gemv",
                        None => "",
                    },
                    if a8 {
                        " a8"
                    } else {
                        ""
                    },
                    sequence.len(),
                    weight_bytes / 1e9,
                    weight_bytes / (median_ms * 1e-3) / 1e9
                );
            }
        }
    });
}

// GPU time of the full-precision bf16 matmul (no dequantization) at the 9B gate_up shape, to separate the
// GEMM schedule from the quantized prologue. `... perf_fp_gemm_bf16 -- --ignored`.
#[uzu_test]
#[ignore]
fn perf_fp_gemm_bf16() {
    use crate::backends::common::CommandBufferCompleted;

    for_each_non_cpu_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("context");
        let (n, k) = (24576usize, 4096usize);
        let weights = create_buffer_with_data::<B, bf16>(
            &context,
            &(0..n * k).map(|i| bf16::from_f32(((i % 17) as f32) * 0.01 - 0.08)).collect::<Vec<_>>(),
        );
        for m in [1usize, 16, 64, 256] {
            let a = create_buffer_with_data::<B, bf16>(
                &context,
                &(0..m * k).map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06)).collect::<Vec<_>>(),
            );
            let mut d = create_buffer::<B, bf16>(&context, m * n);
            let mut kernel = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                bf16::data_type(),
                bf16::data_type(),
                bf16::data_type(),
            )
            .expect("MatmulKernel");
            let mut times = Vec::new();
            for _ in 0..8 {
                let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                kernel
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &a,
                                offset: 0,
                            },
                            b: MatmulB::FullPrecision {
                                b: &weights,
                            },
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut d,
                            d_transform: MatmulDOps::none(),
                            gather_indices: None::<&<B as Backend>::GlobalBuffer>,
                            m: m as u32,
                            n: n as u32,
                            k: k as u32,
                        },
                        &mut command_buffer,
                    )
                    .expect("encode");
                let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                times.push(completed.gpu_execution_time().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_ms = times[times.len() / 2];
            eprintln!(
                "{} fp gate_up m={m:3}: {median_ms:8.3} ms, {:6.1} GB/s weights, {:6.2} TFLOP/s",
                std::any::type_name::<B>(),
                (n * k * 2) as f64 / (median_ms * 1e-3) / 1e9,
                (2 * m * n * k) as f64 / (median_ms * 1e-3) / 1e12
            );
        }
    });
}
