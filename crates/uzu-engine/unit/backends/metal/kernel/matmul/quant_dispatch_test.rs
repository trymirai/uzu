use std::fmt::{Debug, Display};

use QuantParamsLayout::{GroupOutput, OutputGroup};
use QuantizationMethod::{ScaleBias, ScaleZeroPoint};
use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::{
                activation_type::ActivationType, quantization::QuantizationMode,
                quantization_method::QuantizationMethod,
            },
            kernel::{
                ActivationQuantization, ActivationTransform, GatedActMul, GatedActMulSettings, Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    ActivationFormat, Int8CodeLayout, MatmulA, MatmulArguments, MatmulKernel, MatmulShape, QuantParams,
                    QuantParamsLayout, QuantParamsStrides, packed_half_bytes,
                },
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::matmul::{
                GemmKernel, MatmulMetalKernel, MatmulOutputWork,
                gemm::{GemmEngine, GemmPlan},
            },
        },
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        matmul::{QuantBuffers, QuantInput, quant::run_quant_cpu, quant_arguments},
    },
};

fn tile_weights<T: DataTypeRepr + Float>(
    matmul: &MatmulMetalKernel,
    buffers: &mut QuantBuffers<Metal, T>,
    input: &QuantInput<T>,
    context: &MetalContext,
) {
    let shape = MatmulShape {
        a: ActivationFormat::Int8(
            ActivationQuantization::new(
                ACTIVATION_SCALE_GROUP_SIZE,
                input.group_size,
                false,
                Int8CodeLayout::for_right_bits(DataType::from(input.mode).size_in_bits() as u32).unwrap(),
            )
            .unwrap(),
        ),
        ..MatmulShape::from_arguments(&quant_arguments(buffers, input))
    };
    buffers.weights_layout =
        matmul.optimize_weights_layout(&shape, &mut buffers.w, context).expect("optimize weights layout").0;
    assert_eq!(buffers.weights_layout, GemmBLayout::MxuTiled);
}

#[uzu_test]
fn optimize_weights_layout_prepares_unsigned_codes() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let kernel = MatmulMetalKernel::new(&context, bf16::data_type(), bf16::data_type(), bf16::data_type()).unwrap();
    let signed = QuantInput::<bf16>::new(16, 512, 64, 64, 8, QuantizationMethod::ScaleSymmetric, 0)
        .with_group_output()
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);
    let mut unsigned = signed.clone();
    unsigned.signed_codes = false;
    let mut expected = QuantBuffers::<Metal, bf16>::allocate(&context, &signed);
    let mut actual = QuantBuffers::<Metal, bf16>::allocate(&context, &unsigned);
    let signed_shape = MatmulShape::from_arguments(&quant_arguments(&mut expected, &signed));
    let unsigned_shape = MatmulShape::from_arguments(&quant_arguments(&mut actual, &unsigned));

    assert_eq!(
        kernel.optimize_weights_layout(&signed_shape, &mut expected.w, &context).unwrap(),
        (GemmBLayout::MxuTiled, true),
    );
    assert_eq!(
        kernel.optimize_weights_layout(&unsigned_shape, &mut actual.w, &context).unwrap(),
        (GemmBLayout::MxuTiled, true),
    );
    assert_eq!(buffer_to_vec::<Metal, u32>(&context, &actual.w), buffer_to_vec::<Metal, u32>(&context, &expected.w));
}

fn run_quant_metal<T: DataTypeRepr + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    gemm_b_layout: GemmBLayout,
    dispatch: Option<GemmEngine>,
) -> Vec<T> {
    let mut buffers = QuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul =
        MatmulMetalKernel::new(context, T::data_type(), T::data_type(), T::data_type()).expect("MatmulMetalKernel");
    if gemm_b_layout == GemmBLayout::MxuTiled {
        tile_weights(&matmul, &mut buffers, input, context);
    }
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let args = quant_arguments(&mut buffers, input);
    if let Some(engine) = dispatch {
        matmul.encode_with_gemm_engine(args, engine, &mut command_buffer).expect("forced GEMM engine encode failed");
    } else {
        matmul.encode(args, &mut command_buffer).expect("matmul encode failed");
    }
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<Metal, T>(context, &buffers.y)
}

#[uzu_test]
fn quant_params_layout_table() {
    let cases = [
        (QuantParamsLayout::OutputGroup, [5, 3], [3, 1], [5, 2], [4, 1]),
        (QuantParamsLayout::GroupOutput, [3, 8], [1, 8], [3, 4], [1, 8]),
    ];
    for (layout, scale_shape, scale_strides, u4_shape, u4_strides) in cases {
        let params = QuantParams::new(layout, 5, 3);
        let strides = |strides: QuantParamsStrides| [strides.output_stride, strides.group_stride];

        assert_eq!(params.scale_shape(), scale_shape);
        assert_eq!(strides(params.scale_strides()), scale_strides);
        assert_eq!(params.zero_point_shape(QuantizationMode::U8), scale_shape);
        assert_eq!(strides(params.zero_point_strides(QuantizationMode::U8)), scale_strides);
        assert_eq!(params.zero_point_shape(QuantizationMode::U4), u4_shape);
        assert_eq!(strides(params.zero_point_strides(QuantizationMode::U4)), u4_strides);
    }
}

fn assert_parity<T: DataTypeRepr + Float + Debug + Display>(
    label: &str,
    reference: &[T],
    actual: &[T],
    rel_tol: f64,
    abs_tol: f64,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    for (i, (&expected, &actual)) in reference.iter().zip(actual).enumerate() {
        let expected = expected.to_f32().unwrap();
        let actual = actual.to_f32().unwrap();
        let diff = f64::from((expected - actual).abs());
        let tolerance = abs_tol.max(f64::from(expected.abs()) * rel_tol);
        assert!(diff <= tolerance, "{label}[{i}]: expected {expected}, got {actual}, diff {diff} > {tolerance}");
    }
}

fn run_parity<T: DataTypeRepr + Float + Debug + Display>(
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    dispatch: Option<GemmEngine>,
    label: &str,
    rel_tol: f64,
    abs_tol: f64,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 0);
    let unified = run_quant_metal::<T>(&context, &input, GemmBLayout::RowMajor, dispatch);
    let reference = run_quant_cpu::<T>(&input);

    assert_parity::<T>(
        &format!(
            "{label} m={m} k={k} n={n} gs={group_size} bits={bits} method={quant_method:?} dtype={}",
            std::any::type_name::<T>()
        ),
        &reference,
        &unified,
        rel_tol,
        abs_tol,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_4bit_mlx_prefill(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx_prefill(64, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_4bit_mlx_decode(8, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_zp_decode(8, 256, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_unaligned_n(64, 256, 96, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_narrow_n_tail_zp(64, 256, 72, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_narrow_n_tail_mlx(64, 256, 72, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_narrow_n_tail_sym(64, 256, 40, 32, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, Some(GemmEngine::Simdgroup), "gemm", 0.05, 0.4);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_8bit_symmetric_prefill(64, 256, 64, 32, 8, QuantizationMethod::ScaleSymmetric)]
#[case::gs64_8bit_symmetric_prefill(64, 256, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_bf16_8bit_splitk(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, Some(GemmEngine::Simdgroup), "gemm", 0.05, 1.0);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1_gs32_4bit_mlx(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::m1_gs64_4bit_mlx(1, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::m1_gs32_4bit_zp(1, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_gs64_8bit_sym(1, 256, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
#[case::m2_gs32_4bit_mlx(2, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::m4_gs32_4bit_zp(4, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m8_gs32_4bit_zp(8, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m8_gs32_4bit_mlx(8, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::m3_w4_g32_zp_k1024(3, 1024, 72, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m3_w8_g32_sym_k1024(3, 1024, 72, 32, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_gemv_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.4);
}

#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_with_bias() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    input.output_bias = Some((0..input.n).map(|j| bf16::from_f32(0.5 + 0.1 * (j % 5) as f32)).collect());

    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, Some(GemmEngine::Simdgroup));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>("with_bias", &reference, &actual, 0.05, 0.4);
}

#[uzu_test]
fn parity_bf16_gemv_qmv_fused_scale_bias() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut input = QuantInput::<bf16>::new(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    input.ab_scale = 2.0;
    input.output_bias = Some((0..input.n).map(|j| bf16::from_f32(0.25 + 0.1 * (j % 7) as f32)).collect());
    let reference = run_quant_cpu::<bf16>(&input);
    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, None);

    assert_parity::<bf16>("gemv_qmv_scale_bias", &reference, &actual, 0.05, 0.4);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs64_4bit(1, 96, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_zp(2, 96, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
// Whole quantization groups with an unaligned K block.
#[case::gs64_4bit_k1152_zp(1, 1152, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_8bit_k1152_sym(1, 1152, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
// Same, plus a final partial quantization group (1056 = 16 * 64 + 32).
#[case::gs64_4bit_k1056_partial_group(1, 1056, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_4bit_k514_unaligned_rows(2, 514, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_8bit_k257_unaligned_rows(2, 257, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_gemv_partial_group_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // k is not a multiple of group_size, so the per-row scale/zp stride must use
    // ceil(k / group_size) groups.
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.6);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::n12_gs32_4bit(1, 256, 12, 32, 4, QuantizationMethod::ScaleBias)]
#[case::n20_gs64_4bit_zp(2, 256, 20, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::n36_gs32_8bit_sym(1, 256, 36, 32, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_gemv_unaligned_width_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // n is not a multiple of 8; the output-tail clamp must cover the partial block.
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.6);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::decode_fused(1, 256, 64, 32, QuantizationMethod::ScaleBias, true)]
#[case::gemv(1, 256, 64, 32, QuantizationMethod::ScaleBias, false)]
#[case::short_prefill_deferred_rht_then_bias(2, 5120, 6144, 64, QuantizationMethod::ScaleZeroPoint, true)]
#[case::qmv_short_prefill_m2_g64(2, 5120, 6144, 64, QuantizationMethod::ScaleZeroPoint, false)]
#[case::short_prefill_m5_g64(5, 5120, 6144, 64, QuantizationMethod::ScaleZeroPoint, false)]
#[case::short_prefill_m7_g64(7, 5120, 6144, 64, QuantizationMethod::ScaleZeroPoint, false)]
#[case::gemm_nonsplit_bias(9, 64, 64, 64, QuantizationMethod::ScaleZeroPoint, true)]
fn parity_bf16_quant_rht(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] group_size: u32,
    #[case] method: QuantizationMethod,
    #[case] with_bias: bool,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    // W4/G64 matches the Qwen short-prefill route; keep the decode anchor at G32.
    let input = QuantInput::<bf16>::new(m, k, n, group_size, 4, method, 0);
    let mut input = if group_size == 64 {
        input.with_group_output()
    } else {
        input
    };
    input.output_bias = with_bias.then(|| (0..input.n).map(|j| bf16::from_f32(0.3 + 0.05 * (j % 4) as f32)).collect());
    input.output_rht_factors = Some((0..input.n).map(|i| 1 - 2 * (i % 2) as i32).collect());

    // CPU reference applies the RHT through the CPU matmul kernel.
    let reference = run_quant_cpu::<bf16>(&input);
    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, None);

    assert_parity::<bf16>(
        if with_bias {
            "quant_rht_bias"
        } else {
            "quant_rht"
        },
        &reference,
        &actual,
        0.05,
        0.6,
    );
}

#[uzu_test]
fn cpu_group_major_quantized_gemm_matches_row_major() {
    for bits in [4, 8] {
        for method in
            [QuantizationMethod::ScaleBias, QuantizationMethod::ScaleZeroPoint, QuantizationMethod::ScaleSymmetric]
        {
            let input = QuantInput::<bf16>::new(4, 128, 12, 32, bits, method, 0);
            let expected = run_quant_cpu(&input);
            let actual = run_quant_cpu(&input.with_group_output());
            assert_parity("CPU GroupOutput", &expected, &actual, 0.05, 0.5);
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::output_group_single_group_gs64(OutputGroup, 64, 64, 64, 4, ScaleZeroPoint, 64)]
#[case::output_group_odd_groups(OutputGroup, 192, 64, 64, 4, ScaleZeroPoint, 64)]
#[case::group_output_padded_n(GroupOutput, 192, 66, 64, 4, ScaleZeroPoint, 66)]
#[case::output_group_bias_prefix(OutputGroup, 128, 12, 32, 4, ScaleBias, 4)]
#[case::group_output_bias_prefix(GroupOutput, 128, 12, 32, 4, ScaleBias, 4)]
fn quant_gemm_parameter_layout_prefix_matches_cpu(
    #[case] params_layout: QuantParamsLayout,
    #[case] k: u32,
    #[case] n: u32,
    #[case] group_size: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[case] logical_n: u32,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");

    let input = QuantInput::<bf16>::new(64, k, n, group_size, bits, method, 0);
    let input = if params_layout == GroupOutput {
        input.with_group_output()
    } else {
        input
    };
    let reference = run_quant_cpu(&input);
    for engine in [GemmEngine::Simdgroup, GemmEngine::Mxu] {
        if engine == GemmEngine::Mxu && !context.supports_mxu {
            continue;
        }
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
        let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
        let mut args = quant_arguments(&mut buffers, &input);
        args.n = logical_n;
        matmul.encode_with_gemm_engine(args, engine, &mut command_buffer).unwrap();
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        let actual = buffer_to_vec::<Metal, bf16>(&context, &buffers.y);
        let expected: Vec<_> = reference
            .chunks_exact(input.n as usize)
            .flat_map(|row| row[..logical_n as usize].iter().copied())
            .collect();

        assert_parity("quant params layout", &expected, &actual[..expected.len()], 0.05, 0.5);
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_4bit_mlx(128, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx(128, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::small_m_4bit_mlx(24, 256, 512, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_4bit_zp(128, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_4bit_zp_g1(128, 64, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
fn mxu_quant_parity_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let input = QuantInput::<bf16>::new(m, k, n, gs, bits, method, 0);
    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(
        &format!("MXU m={m} k={k} n={n} gs={gs} bits={bits} method={method:?}"),
        &reference,
        &actual,
        0.05,
        0.5,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::sym_w8_gs64(16u32, 256u32, 128u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::bias_w4_gs32(16u32, 256u32, 128u32, 32u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::zp_w4_gs32(16u32, 256u32, 128u32, 32u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_bias_w4_gs64(1u32, 256u32, 128u32, 64u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::w8_m1(1u32, 256u32, 96u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::w8_m16_split_k(16u32, 4096u32, 1024u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::w8_m64(64u32, 512u32, 256u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::w4_m1(1u32, 256u32, 96u32, 64u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::w4_m16_split_k(16u32, 4096u32, 1024u32, 64u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::w4_m24(24u32, 512u32, 96u32, 64u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::w4_m128_bias(128u32, 512u32, 256u32, 64u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::w4_m16_k17408(16u32, 17408u32, 64u32, 64u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::w8_m16_k17408(16u32, 17408u32, 64u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
fn a8w_mxu_parity_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] weight_gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[values(GemmBLayout::RowMajor, GemmBLayout::MxuTiled)] gemm_b_layout: GemmBLayout,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let (input, reference_input) = QuantInput::<bf16>::new(m, k, n, weight_gs, bits, method, 0)
        .with_prepared_a_and_reference(
            ACTIVATION_SCALE_GROUP_SIZE,
            (method != QuantizationMethod::ScaleSymmetric).then_some(weight_gs),
        );
    let actual = run_quant_metal(&context, &input.with_group_output(), gemm_b_layout, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&reference_input);
    assert_parity::<bf16>(
        &format!(
            "A8W{bits} MXU m={m} k={k} n={n} weight_gs={weight_gs} method={method:?} gemm_b_layout={gemm_b_layout:?}"
        ),
        &reference,
        &actual,
        0.08,
        0.8,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1u32)]
#[case::m16(16u32)]
#[case::m33(33u32)]
fn a8w_independent_activation_group_parity_bf16(#[case] m: u32) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }

    for (activation_group_size, weight_group_size) in
        [(ACTIVATION_SCALE_GROUP_SIZE, 32u32), (ACTIVATION_SCALE_GROUP_SIZE, 64)]
    {
        for (bits, method) in [(8, QuantizationMethod::ScaleSymmetric), (4, QuantizationMethod::ScaleZeroPoint)] {
            let (input, reference_input) = QuantInput::<bf16>::new(m, 256, 12, weight_group_size, bits, method, 0)
                .with_prepared_a_and_reference(
                    activation_group_size,
                    (method != QuantizationMethod::ScaleSymmetric)
                        .then_some(activation_group_size.min(weight_group_size)),
                );
            let actual =
                run_quant_metal(&context, &input.with_group_output(), GemmBLayout::RowMajor, Some(GemmEngine::Mxu));
            let reference = run_quant_cpu::<bf16>(&reference_input);
            assert_parity::<bf16>(
                &format!("A8W{bits} act{activation_group_size} weight{weight_group_size} m={m} method={method:?}"),
                &reference,
                &actual,
                0.08,
                0.8,
            );
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1)]
#[case::m16(16)]
#[case::m33(33)]
fn a8w4_zero_point_tail_parity(#[case] m: u32) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let input = QuantInput::<bf16>::new(m, 256, 72, 32, 4, QuantizationMethod::ScaleZeroPoint, 0)
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, Some(32));
    let reference = run_quant_cpu::<bf16>(&input);
    let actual = run_quant_metal(&context, &input.with_group_output(), GemmBLayout::RowMajor, Some(GemmEngine::Mxu));
    assert_parity::<bf16>(&format!("A8W4 ZP N-tail m={m}"), &reference, &actual, 0.08, 0.8);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w4_bias(4, QuantizationMethod::ScaleBias, 256, 2, 32)]
#[case::w4_zp(4, QuantizationMethod::ScaleZeroPoint, 256, 2, 32)]
#[case::w8_sym(8, QuantizationMethod::ScaleSymmetric, 256, 2, 32)]
#[case::direct_w8_m5(8, QuantizationMethod::ScaleSymmetric, 1024, 5, 64)]
fn signed_weights_full_precision_activations_parity_bf16(
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[case] k: u32,
    #[case] m: u32,
    #[case] group_size: u32,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let n = 128;
    let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0).with_signed_weight_codes();
    let reference_input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0);
    let reference = run_quant_cpu::<bf16>(&reference_input);

    for (label, dispatch) in [("auto", None), ("simdgroup", Some(GemmEngine::Simdgroup))] {
        let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, dispatch);
        assert_parity::<bf16>(
            &format!("signed weights FP-A {label} bits={bits} method={method:?}"),
            &reference,
            &actual,
            0.05,
            0.5,
        );
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w8_bias(8u32, false)]
#[case::w8_rht_bias(8u32, true)]
fn a8w_mxu_output_bias_parity_bf16(
    #[case] bits: u32,
    #[case] with_output_hadamard: bool,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }

    let (m, k, n) = (
        16u32,
        256u32,
        if with_output_hadamard {
            64u32
        } else {
            72u32
        },
    );
    let (mut input, mut reference_input) =
        QuantInput::<bf16>::new(m, k, n, 32, bits, QuantizationMethod::ScaleSymmetric, 0)
            .with_group_output()
            .with_prepared_a_and_reference(ACTIVATION_SCALE_GROUP_SIZE, None);
    let output_bias: Vec<bf16> = (0..n).map(|column| bf16::from_f32(0.25 + 0.05 * (column % 7) as f32)).collect();
    let output_hadamard_factors: Option<Vec<i32>> = with_output_hadamard.then(|| {
        (0..n)
            .map(|column| {
                if column.is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect()
    });
    input.output_bias = Some(output_bias.clone());
    reference_input.output_bias = Some(output_bias);
    input.output_rht_factors = output_hadamard_factors.clone();
    reference_input.output_rht_factors = output_hadamard_factors;

    let reference = run_quant_cpu::<bf16>(&reference_input);

    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::RowMajor, Some(GemmEngine::Mxu));

    assert_parity::<bf16>(
        &format!("A8W{bits} MXU output bias, output Hadamard={with_output_hadamard}"),
        &reference,
        &actual,
        0.08,
        0.8,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w4_zp_gs64(1024u32, 1024u32, 64u32, 4u32, ScaleZeroPoint)]
#[case::w4_zp_gs32_wide(1024u32, 3584u32, 32u32, 4u32, ScaleZeroPoint)]
#[case::w4_zp_gs64_deep(3584u32, 1024u32, 64u32, 4u32, ScaleZeroPoint)]
#[case::w4_bias_gs64(2048u32, 2048u32, 64u32, 4u32, ScaleBias)]
#[case::w8_sym_gs64(1024u32, 2048u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::w8_sym_gs64_blocks(2048u32, 4096u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::w8_sym_gs32_blocks(2048u32, 4096u32, 32u32, 8u32, QuantizationMethod::ScaleSymmetric)]
fn gemv_reads_mxu_tiled_layout_bf16(
    #[case] k: u32,
    #[case] n: u32,
    #[case] group_size: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[values(false, true)] output_hadamard: bool,
    #[values(1, 2, 3, 4)] m: u32,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0).with_group_output();
    let mut input = if bits == 8 {
        input.with_signed_weight_codes()
    } else {
        input
    };
    let factors: Vec<i32> = (0..n)
        .map(|row| {
            if row.is_multiple_of(3) {
                -1
            } else {
                1
            }
        })
        .collect();
    input.output_rht_factors = output_hadamard.then_some(factors);

    let reference = run_quant_cpu::<bf16>(&input);
    let actual = run_quant_metal::<bf16>(&context, &input, GemmBLayout::MxuTiled, None);

    assert_parity::<bf16>(
        &format!(
            "MxuTiled GEMV m={m} k={k} n={n} gs={group_size} bits={bits} method={method:?} hadamard={output_hadamard}"
        ),
        &reference,
        &actual,
        0.05,
        0.5,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w4_zp_gs32(4u32, 32u32, ScaleZeroPoint)]
#[case::w8_sym_gs64(8u32, 64u32, QuantizationMethod::ScaleSymmetric)]
fn gathered_gemv_reads_mxu_tiled_layout_bf16(
    #[case] bits: u32,
    #[case] group_size: u32,
    #[case] method: QuantizationMethod,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let (k, vocab, gathered_rows) = (256u32, 256u32, 10u32);
    let input = QuantInput::<bf16>::new(1, k, vocab, group_size, bits, method, 0).with_group_output();
    let input = if bits == 8 {
        input.with_signed_weight_codes()
    } else {
        input
    };
    let ids: Vec<u32> = (0..gathered_rows).map(|row| (row * 37 + 11) % vocab).collect();
    let dense = run_quant_cpu::<bf16>(&input);
    let expected: Vec<bf16> = ids.iter().map(|&id| dense[id as usize]).collect();

    let ids_buffer = crate::tests::helpers::create_buffer_with_data::<Metal, u32>(&context, &ids);
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("Metal matmul kernel");
    tile_weights(&matmul, &mut buffers, &input, &context);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut arguments = quant_arguments(&mut buffers, &input);
    arguments.gather_indices = Some(&ids_buffer);
    arguments.n = gathered_rows;
    matmul.encode(arguments, &mut command_buffer).expect("gathered GEMV");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    let actual = buffer_to_vec::<Metal, bf16>(&context, &buffers.y)[..gathered_rows as usize].to_vec();

    assert_parity::<bf16>(
        &format!("gathered MxuTiled GEMV bits={bits} gs={group_size} method={method:?}"),
        &expected,
        &actual,
        0.05,
        0.5,
    );
}

fn run_packed_half_pipeline(
    group_size: u32,
    gated: bool,
    (batch, n): (u32, u32),
    with_bias: bool,
    with_rht: bool,
    soft_cap: Option<f32>,
    method: QuantizationMethod,
    split_k: u32,
) {
    let k = 512;
    let source_width = k * (1 + gated as u32);
    let source: Vec<bf16> = (0..batch * source_width).map(|i| bf16::from_f32((i as f32 * 0.013).sin() * 0.2)).collect();
    let signs = |len: u32| (0..len).map(|i| 1 - 2 * (i % 3 == 0) as i32).collect::<Vec<_>>();
    let factors = signs(k);
    let format = ActivationFormat::PackedHalf {
        group_size,
    };
    let mut input = QuantInput::<bf16>::new(batch, k, n, group_size, 4, method, 19).with_group_output();
    input.ab_scale = 0.75;
    input.output_bias = with_bias.then(|| (0..n).map(|i| bf16::from_f32(1.0 + 0.5 * (i % 3) as f32)).collect());
    input.output_rht_factors = with_rht.then(|| signs(n));
    input.soft_cap = soft_cap;
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let source_buffer = create_buffer_with_data::<Metal, bf16>(&context, &source);
    let factors_buffer = create_buffer_with_data::<Metal, i32>(&context, &factors);
    let mut reference_a = create_buffer::<Metal, bf16>(&context, (batch * k) as usize);
    let mut gemm = GemmKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut packed = create_buffer::<Metal, u32>(&context, packed_half_bytes(k, group_size) / size_of::<u32>());
    let mut command_buffer = context.create_command_buffer(None, None).expect("Metal command buffer");
    if !gated {
        ActivationTransform::<Metal>::input_rht(&context, DataType::BF16, false).expect("RHT reference").encode_fp(
            &source_buffer,
            &mut reference_a,
            &factors_buffer,
            batch,
            k,
            &mut command_buffer,
        );
        ActivationTransform::<Metal>::quantize(&context, DataType::BF16, format)
            .expect("Metal RHT PackedHalf pack")
            .encode_pack_half(&source_buffer, &mut packed, &factors_buffer, batch, k, &mut command_buffer);
    } else {
        GatedActMul::<Metal>::full_precision(&context, DataType::BF16, true, true, GatedActMulSettings::default())
            .expect("gate reference")
            .encode_fp(
                &source_buffer,
                None::<&<Metal as Backend>::GlobalGpuBuffer>,
                &mut reference_a,
                Some(&factors_buffer),
                k,
                batch,
                0,
                0,
                ActivationType::SILU,
                &mut command_buffer,
            );
        GatedActMul::<Metal>::quantized(&context, DataType::BF16, format, GatedActMulSettings::default())
            .expect("Metal PackedHalf gated activation")
            .encode_pack_half(
                &source_buffer,
                &mut packed,
                &factors_buffer,
                k,
                batch,
                ActivationType::SILU,
                &mut command_buffer,
            );
    }
    let arguments = quant_arguments(&mut buffers, &input);
    let arguments = MatmulArguments {
        a: MatmulA::PackedHalf {
            values: &packed,
            group_size,
        },
        b: arguments.b,
        b_leading_dimension: arguments.b_leading_dimension,
        b_transpose: arguments.b_transpose,
        output: arguments.output,
        gather_indices: arguments.gather_indices,
        m: arguments.m,
        n: arguments.n,
        k: arguments.k,
    };
    let plan = GemmPlan {
        engine: GemmEngine::Simdgroup,
        tiling: GemmTiling::Tile8x32x64_Simdgroups1x2x4,
        split_k,
    };
    gemm.encode_plan(arguments, plan, &output_work, &mut command_buffer).expect("prepared W4 GEMM");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    input.x = buffer_to_vec::<Metal, bf16>(&context, &reference_a);
    let reference = run_quant_cpu::<bf16>(&input);
    let actual = buffer_to_vec::<Metal, bf16>(&context, &buffers.y);
    assert_parity::<bf16>(&format!("gated={gated} G{group_size} M{batch} K{k}"), &reference, &actual, 0.05, 0.4);
}

#[uzu_test]
fn packed_half_producers_match_bf16_reference() {
    // Fused RHT rounds directly to half; the standalone reference stores BF16 first.
    for (group, gated, n, bias, rht, cap, method, splits) in [
        (32, false, (4, 72), false, false, None, ScaleZeroPoint, 2),
        (64, false, (5, 64), true, false, None, ScaleBias, 1),
        (32, true, (8, 64), true, true, None, ScaleZeroPoint, 1),
        (64, true, (5, 64), true, false, Some(0.5), ScaleBias, 2),
    ] {
        run_packed_half_pipeline(group, gated, n, bias, rht, cap, method, splits);
    }
}
