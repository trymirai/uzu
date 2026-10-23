use half::bf16;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::matmul::{ActivationFormat, MatmulKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_parity,
        helpers::{create_context, for_each_non_cpu_backend},
        matmul::{
            QuantInput, production_quant_buffers, production_quant_input, quant_tolerance, run_quant, run_quant_with,
        },
    },
};

const SCHEMES: [(u32, QuantizationMethod); 5] = [
    (4, QuantizationMethod::ScaleBias),
    (4, QuantizationMethod::ScaleZeroPoint),
    (4, QuantizationMethod::ScaleSymmetric),
    (8, QuantizationMethod::ScaleBias),
    (8, QuantizationMethod::ScaleSymmetric),
];

#[rstest]
#[test_attr(uzu_test)]
fn quant_matmul_bf16(
    #[values(1, 8, 16, 40, 64, 128)] m: u32,
    #[values((256, 64), (256, 72), (1024, 200))] k_n: (u32, u32),
) {
    let (k, n) = k_n;
    for group_size in [32, 64, 128] {
        for (bits, method) in SCHEMES {
            let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0);
            let expected = run_quant::<Cpu, bf16>(&input);
            let (rel_tol, abs_tol) = quant_tolerance(ActivationFormat::Bf16, bits, &expected);
            for_each_non_cpu_backend!(|B| {
                let actual = run_quant::<B, bf16>(&input);
                let backend = std::any::type_name::<B>();
                let label = format!("{backend} m={m} k={k} n={n} gs={group_size} bits={bits} {method:?}");
                assert_parity(&label, &expected, &actual, rel_tol, abs_tol);
            });
        }
    }
}

/// What a linear loaded with A8 storage runs: weights laid out by `optimize_weights_layout`, then the activation format
/// `select_activation_format` picks for the batch, against the CPU's product of the same codes. Weights the backend
/// gives no A8 storage stay row-major with BF16 activations, which `quant_matmul_bf16` covers. Where n allows, the
/// output also takes RHT-wrapped linears' output transform and bias.
#[rstest]
#[test_attr(uzu_test)]
fn quant_matmul_a8_bf16(
    #[values(1, 3, 5, 8, 16, 40, 64, 128)] m: u32,
    // 6144 is long enough for GEMMs to split K.
    #[values((256, 64), (256, 72), (1024, 200), (6144, 64))] k_n: (u32, u32),
) {
    let (k, n) = k_n;
    for_each_non_cpu_backend!(|B| quant_matmul_a8_bf16_impl::<B>(m, k, n));
}

fn quant_matmul_a8_bf16_impl<B: Backend>(
    m: u32,
    k: u32,
    n: u32,
) {
    let context = create_context::<B>();
    let matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
    )
    .expect("MatmulKernel");
    for group_size in [32, 64, 128] {
        for (bits, method) in SCHEMES {
            let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0).with_group_output();
            // Symmetric weights keep the plain epilogue covered.
            let input = match n.is_multiple_of(32) && method != QuantizationMethod::ScaleSymmetric {
                true => input.with_output_rht(method == QuantizationMethod::ScaleZeroPoint),
                false => input,
            };
            let Some((input, format)) = production_quant_input::<B, bf16>(&context, &matmul, input) else {
                continue;
            };
            let expected = run_quant::<Cpu, bf16>(&input);
            let (rel_tol, abs_tol) = quant_tolerance(format, bits, &expected);
            let mut buffers = production_quant_buffers::<B, bf16>(&context, &matmul, &input);
            let actual = run_quant_with::<B, bf16>(&context, &mut buffers, &input);
            let backend = std::any::type_name::<B>();
            let label = format!("{format:?} {backend} m={m} k={k} n={n} gs={group_size} bits={bits} {method:?}");
            assert_parity(&label, &expected, &actual, rel_tol, abs_tol);
        }
    }
}
