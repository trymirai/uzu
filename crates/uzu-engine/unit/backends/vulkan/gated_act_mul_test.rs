use std::{
    fmt::Debug,
    mem::size_of,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    assert_quantized, check_bounds, gpu_transformed, kernel_fixture::KernelFixture, label, oracle, quantizations,
    quantized, raw, round32, signs, silu_oracle, to, transform_cpu_outputs, transform_oracle, values,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{ActivationTransformOp, ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
            kernel::{ActivationQuantization, GatedActMulKernel, matmul::Int8CodeLayout},
        },
        cpu::Cpu,
        vulkan::{ActivationTransformVulkanKernel, Error, GatedActMulVulkanKernel, VkBuffer},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

const ACTIVATIONS: [ActivationType; 5] = [
    ActivationType::SILU,
    ActivationType::GELUApprox,
    ActivationType::GELUExact,
    ActivationType::IDENTITY,
    ActivationType::SOFTPLUS,
];
const CODE_SENTINEL: i8 = 0x5a;
const SCALE_SENTINEL: f32 = -7.0;
const SUM_SENTINEL: i32 = 0x5a5a_5a5a;
/// Value-operand elements outside the gated span of a row.
const PADDING: f32 = -3.25;

/// Kernel settings `(quantization, interleaved, hadamard, alpha, gate clip, value clip)` and the model's
/// specialization: full precision specializes groups of 32, quantization without sums its scale group as sum group.
fn vulkan_kernel<T: ArrayElement>(
    fixture: &KernelFixture,
    (quantization, interleaved, hadamard, alpha, gate_clip, value_clip): (
        Option<(usize, Option<usize>, Int8CodeLayout)>,
        bool,
        bool,
        Option<f32>,
        Option<(f32, f32)>,
        Option<(f32, f32)>,
    ),
) -> GatedActMulVulkanKernel {
    let (ops, grouped, scale_group, sum_group) = specialization(quantization);
    GatedActMulVulkanKernel::new(
        &fixture.context,
        T::data_type(),
        ops,
        grouped,
        interleaved,
        hadamard,
        scale_group,
        sum_group,
        alpha.is_some(),
        gate_clip.is_some(),
        value_clip.is_some(),
    )
    .expect("Vulkan GatedActMul")
}

fn specialization(quantization: Option<(usize, Option<usize>, Int8CodeLayout)>) -> (GatedActMulOp, bool, u32, u32) {
    match quantization {
        None => (GatedActMulOp::FullPrecision, false, HADAMARD_TRANSFORM_BLOCK_SIZE, HADAMARD_TRANSFORM_BLOCK_SIZE),
        Some((scale_group, sum_group, layout)) => (
            if sum_group.is_some() {
                GatedActMulOp::QuantizeWithGroupSums
            } else {
                GatedActMulOp::Quantize
            },
            layout.is_grouped_by_nibble(),
            scale_group as u32,
            sum_group.unwrap_or(scale_group) as u32,
        ),
    }
}

/// The kernel operands of `batch` rows of `dim` gates and values: interleaved rows [values, gates], or gates alone with
/// values at `offset` in rows of `stride`, the rest padding.
fn operands<T: Float>(
    (gates, values): (&[T], &[T]),
    (dim, batch, interleaved, offset, stride): (usize, usize, bool, usize, usize),
) -> (Vec<T>, Option<Vec<T>>) {
    assert_eq!((gates.len(), values.len()), (dim * batch, dim * batch), "gates and values per row");
    let rows = |row: usize| (&gates[row * dim..][..dim], &values[row * dim..][..dim]);
    if interleaved {
        return ((0..batch).flat_map(|row| [rows(row).1, rows(row).0].concat()).collect(), None);
    }
    assert!(offset + dim <= stride, "values fit their rows");
    let padding = T::from(PADDING).unwrap();
    let padded = (0..batch)
        .flat_map(|row| [vec![padding; offset], rows(row).1.to_vec(), vec![padding; stride - offset - dim]].concat());
    (gates.to_vec(), Some(padded.collect()))
}

/// The CPU kernel through the shared trait, with outputs as `gpu_outputs`; CPU buffers cannot be empty.
fn cpu_outputs<T: ArrayElement + Float>(
    settings: (
        Option<(usize, Option<usize>, Int8CodeLayout)>,
        bool,
        bool,
        Option<f32>,
        Option<(f32, f32)>,
        Option<(f32, f32)>,
    ),
    act_type: ActivationType,
    data: (&[T], &[T], &[i32]),
    shape: (usize, usize, usize, usize),
) -> (Vec<T>, Vec<i8>, Vec<f32>, Vec<i32>) {
    let (quantization, interleaved, hadamard, alpha, gate_clip, value_clip) = settings;
    let (dim, batch, offset, stride) = shape;
    let mut result = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if dim * batch == 0 {
        return result;
    }
    let (act, value) = operands((data.0, data.1), (dim, batch, interleaved, offset, stride));
    let context = create_context::<Cpu>();
    let (ops, grouped, scale_group, sum_group) = specialization(quantization);
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        T::data_type(),
        ops,
        grouped,
        interleaved,
        hadamard,
        scale_group,
        sum_group,
        alpha.is_some(),
        gate_clip.is_some(),
        value_clip.is_some(),
    )
    .expect("CPU GatedActMul");
    let n = dim * batch;
    let act = create_buffer_with_data::<Cpu, T>(&context, &act);
    let value = value.map(|value| create_buffer_with_data::<Cpu, T>(&context, &value));
    let mut fp = quantization.is_none().then(|| create_buffer_with_data::<Cpu, T>(&context, &vec![T::zero(); n]));
    let mut codes = quantization.map(|_| create_buffer_with_data::<Cpu, i8>(&context, &vec![0; n]));
    let mut scales =
        quantization.map(|(group, ..)| create_buffer_with_data::<Cpu, f32>(&context, &vec![0.0; n / group]));
    let sums = quantization.and_then(|(_, sum_group, _)| sum_group).map(|group| n / group);
    let mut sums = sums.map(|sums| create_buffer_with_data::<Cpu, i32>(&context, &vec![0; sums]));
    let factors = hadamard.then(|| create_buffer_with_data::<Cpu, i32>(&context, data.2));
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(
        &act,
        value.as_ref(),
        fp.as_mut(),
        codes.as_mut(),
        scales.as_mut(),
        sums.as_mut(),
        factors.as_ref(),
        dim as u32,
        batch as u32,
        offset as u32,
        stride as u32,
        act_type,
        alpha,
        gate_clip.map(|clip| clip.0),
        gate_clip.map(|clip| clip.1),
        value_clip.map(|clip| clip.0),
        value_clip.map(|clip| clip.1),
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    if let Some(fp) = &fp {
        result.0 = buffer_to_vec::<Cpu, T>(fp);
    }
    if let (Some(codes), Some(scales)) = (&codes, &scales) {
        (result.1, result.2) = (buffer_to_vec::<Cpu, i8>(codes), buffer_to_vec::<Cpu, f32>(scales));
    }
    if let Some(sums) = &sums {
        result.3 = buffer_to_vec::<Cpu, i32>(sums);
    }
    result
}

fn range(guarded: &Option<(Arc<VkBuffer>, Range<u64>)>) -> Option<(&Arc<VkBuffer>, Range<u64>)> {
    guarded.as_ref().map(|(buffer, range)| (buffer, range.clone()))
}

/// Records every case `(settings, act_type, (gates, values, factors), (dim, batch, offset, stride))` into one command
/// buffer over guarded ranges, then returns full precision, codes, scales and sums, each empty when absent, after
/// checking every guard and that the read-only operands are unchanged.
#[allow(clippy::type_complexity)]
fn gpu_outputs<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    cases: &[(
        (
            Option<(usize, Option<usize>, Int8CodeLayout)>,
            bool,
            bool,
            Option<f32>,
            Option<(f32, f32)>,
            Option<(f32, f32)>,
        ),
        ActivationType,
        (&[T], &[T], &[i32]),
        (usize, usize, usize, usize),
    )],
) -> Vec<(Vec<T>, Vec<i8>, Vec<f32>, Vec<i32>)> {
    let sentinel = T::from(-7.0).unwrap();
    let buffers = cases
        .iter()
        .map(|&(settings, _, (gates, values, factors), (dim, batch, offset, stride))| {
            let (act, value) = operands((gates, values), (dim, batch, settings.1, offset, stride));
            let n = dim * batch;
            let quantization = settings.0;
            (
                vulkan_kernel::<T>(fixture, settings),
                fixture.guarded(&act, sentinel),
                value.as_ref().map(|value| fixture.guarded(value, sentinel)),
                quantization.is_none().then(|| fixture.guarded(&vec![sentinel; n], sentinel)),
                quantization.map(|_| fixture.guarded(&vec![CODE_SENTINEL; n], CODE_SENTINEL)),
                quantization.map(|(group, ..)| fixture.guarded(&vec![SCALE_SENTINEL; n / group], SCALE_SENTINEL)),
                quantization
                    .and_then(|(_, sum_group, _)| sum_group)
                    .map(|g| fixture.guarded(&vec![SUM_SENTINEL; n / g], SUM_SENTINEL)),
                settings.2.then(|| fixture.guarded(factors, SUM_SENTINEL)),
                (act, value),
            )
        })
        .collect::<Vec<_>>();
    let mut encoding = fixture.encoding();
    for (
        &(settings, act_type, _, (dim, batch, offset, stride)),
        (kernel, act, value, fp, codes, scales, sums, factors, _),
    ) in cases.iter().zip(&buffers)
    {
        let (_, _, _, alpha, gate_clip, value_clip) = settings;
        // SAFETY: each range holds exactly the elements its argument covers for `batch` rows of `dim`, values rows of
        // `stride`, and outputs alias nothing.
        unsafe {
            kernel.encode(
                (&act.0, act.1.clone()),
                range(value),
                range(fp),
                range(codes),
                range(scales),
                range(sums),
                range(factors),
                dim as u32,
                batch as u32,
                offset as u32,
                stride as u32,
                act_type,
                alpha,
                gate_clip.map(|clip| clip.0),
                gate_clip.map(|clip| clip.1),
                value_clip.map(|clip| clip.0),
                value_clip.map(|clip| clip.1),
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    cases
        .iter()
        .zip(&buffers)
        .map(
            |(
                &(_, _, (_, _, factor_values), _),
                (_, act, value, fp, codes, scales, sums, factors, (act_values, values)),
            )| {
                // SAFETY: the only command buffer using these buffers has completed.
                unsafe {
                    KernelFixture::assert_unchanged(act, sentinel, act_values, "act_operand");
                    if let (Some(value), Some(values)) = (value, values) {
                        KernelFixture::assert_unchanged(value, sentinel, values, "value_operand");
                    }
                    if let Some(factors) = factors {
                        KernelFixture::assert_unchanged(factors, SUM_SENTINEL, factor_values, "factors");
                    }
                    (
                        fp.as_ref().map_or(Vec::new(), |fp| KernelFixture::read_guarded(fp, sentinel)),
                        codes.as_ref().map_or(Vec::new(), |codes| KernelFixture::read_guarded(codes, CODE_SENTINEL)),
                        scales
                            .as_ref()
                            .map_or(Vec::new(), |scales| KernelFixture::read_guarded(scales, SCALE_SENTINEL)),
                        sums.as_ref().map_or(Vec::new(), |sums| KernelFixture::read_guarded(sums, SUM_SENTINEL)),
                    )
                }
            },
        )
        .collect()
}

/// `x` clamped like f32::clamp, which keeps NaN.
fn clip(
    x: f64,
    bounds: Option<(f32, f32)>,
) -> f64 {
    bounds.map_or(x, |(lo, hi)| match (x < f64::from(lo), x > f64::from(hi)) {
        (true, _) => f64::from(lo),
        (_, true) => f64::from(hi),
        _ => x,
    })
}

/// Bounds of the products the CPU stages, each rounded to `T`: the clipped gate, its activation (the activation
/// oracle's bounds, custom alpha only for SiLU), the clipped value, and their FP32 product. The product is monotonic in
/// the activation for a fixed value, so the activation's bounds give the product's.
fn product_bounds<T: Float>(
    (gates, values): (&[T], &[T]),
    act_type: ActivationType,
    (alpha, gate_clip, value_clip): (Option<f32>, Option<(f32, f32)>, Option<(f32, f32)>),
) -> Vec<((f64, f64), f64)> {
    assert_eq!(gates.len(), values.len(), "gates and values");
    gates
        .iter()
        .zip(values)
        .map(|(gate, value)| {
            let gate = to::<T>(clip(gate.to_f64().unwrap(), gate_clip));
            let ((lo, hi), center) = match alpha {
                Some(alpha) if act_type == ActivationType::SILU => silu_oracle(gate, alpha),
                _ => oracle(gate, act_type),
            };
            let value = to::<T>(clip(value.to_f64().unwrap(), value_clip));
            let product = |activated: f64| to::<T>(round32(value * to::<T>(activated)));
            let (a, b) = (product(lo), product(hi));
            ((a.min(b), a.max(b)), product(center))
        })
        .collect()
}

/// Bounds of the full-precision output: the products, or with the transform their FP32 transform rounded to `T`.
fn output_bounds<T: Float>(
    products: Vec<((f64, f64), f64)>,
    factors: &[i32],
    hadamard: bool,
) -> Vec<((f64, f64), f64)> {
    if !hadamard {
        return products;
    }
    let transformed = transform_oracle(&products, factors, true, 32);
    transformed.into_iter().map(|((lo, hi), center)| ((to::<T>(lo), to::<T>(hi)), to::<T>(center))).collect()
}

#[uzu_test]
fn f32_gated_transform() {
    let fixture = KernelFixture::new();
    let (gates, values, factors) = (values::<f32>(3 * 64, 1), values::<f32>(3 * 64, 2), signs(64, 1));
    let settings = (None, true, true, None, None, None);
    let case = (settings, ActivationType::SILU, (&gates[..], &values[..], &factors[..]), (64, 3, 0, 0));
    let bounds = output_bounds::<f32>(
        product_bounds((&gates, &values), ActivationType::SILU, (None, None, None)),
        &factors,
        true,
    );
    let gpu = gpu_outputs(&fixture, &[case]);
    check_bounds(&bounds, &cpu_outputs(case.0, case.1, case.2, case.3).0, &gpu[0].0, "F32 SiLU RHT");
    fixture.assert_clean();
}

/// Every activation with four setting mixes (interleaved or separate values at an offset in longer rows, the transform
/// where the row allows it, clips and custom alphas) for every shape, a shape's cases interleaved in one command buffer:
/// guarded spans, unchanged operands, CPU and Vulkan within the staged oracle's bounds. Prints Vulkan's differences
/// from the CPU per setting.
fn full_precision_matches_oracle<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let mut differences = std::collections::BTreeMap::<String, [i64; 2]>::new();
    for (batch, dim) in [(0, 32), (1, 0), (1, 1), (3, 31), (1, 33), (3, 64), (1, 96), (128, 128), (3, 160), (1, 4096)] {
        let (gates, values) = (values::<T>(batch * dim, dim), values::<T>(batch * dim, dim + 7));
        let factors = signs(dim, batch);
        let rht = dim % 32 == 0;
        let mixes = [
            (true, false, None, None, None),
            (false, rht, None, Some((-6.5, 7.0)), None),
            (true, rht, Some(1.702), None, Some((-3.0, 2.5))),
            (false, false, Some(-0.7), Some((-1.0, 1.0)), Some((-4.0, 4.0))),
        ];
        let cases = ACTIVATIONS
            .iter()
            .flat_map(|&act| {
                mixes.map(|(interleaved, hadamard, alpha, gate_clip, value_clip)| {
                    (act, (None, interleaved, hadamard, alpha, gate_clip, value_clip))
                })
            })
            .map(|(act, settings)| (settings, act, (&gates[..], &values[..], &factors[..]), (dim, batch, 3, dim + 5)))
            .collect::<Vec<_>>();
        let outputs = gpu_outputs(&fixture, &cases);
        assert_eq!(outputs.len(), cases.len(), "one output per case");
        for (&(settings, act, data, shape), gpu) in cases.iter().zip(outputs) {
            let (_, interleaved, hadamard, alpha, gate_clip, value_clip) = settings;
            let label =
                format!("{:?} {act:?} interleaved {interleaved} hadamard {hadamard} alpha {alpha:?}", T::data_type());
            let bounds = output_bounds::<T>(
                product_bounds((data.0, data.1), act, (alpha, gate_clip, value_clip)),
                data.2,
                hadamard,
            );
            let [count, steps] = check_bounds(
                &bounds,
                &cpu_outputs(settings, act, data, shape).0,
                &gpu.0,
                &format!("{label} {batch}x{dim}"),
            );
            let entry = differences.entry(label).or_default();
            *entry = [entry[0] + count, entry[1].max(steps)];
        }
    }
    for (label, [count, steps]) in differences.iter().filter(|(_, [count, _])| *count > 0) {
        eprintln!("GatedActMul {label}: {count} results differ from the CPU by up to {steps} storage steps");
    }
    fixture.assert_clean();
}

#[uzu_test]
fn full_precision_matches_oracle_all_types() {
    full_precision_matches_oracle::<f32>();
    full_precision_matches_oracle::<bf16>();
}

/// Gates at the custom-alpha SiLU thresholds, 8 FP32 steps on each side: where -alpha x reaches 2^-26 (the exact half),
/// where 1 + e^(-alpha x) passes 2^126 (quarter scaling) and where the exponential overflows; plus clip bounds,
/// subnormals, the largest finite values, infinities, NaN and signed zeros. Alphas 0, negative, positive, huge and
/// subnormal, all FP32 values that BF16 cannot hold, with and without the transform, both types: CPU and Vulkan within
/// the staged oracle's bounds, so no result flushes. Values include 1e30, so a subnormal activation makes a normal output.
fn thresholds_match_oracle<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let near = |x: f32| {
        let (sign, magnitude) = (x.to_bits() & 0x8000_0000, x.to_bits() & 0x7fff_ffff);
        (-8..=8).map(move |k| f32::from_bits(sign | magnitude.saturating_add_signed(k)))
    };
    let alphas = [0.0, -0.7, 1.702, 1e30, f32::from_bits(0x0010_0000), 1e37, 3e38];
    let gate_clip = (-6.5f32, 7.000_000_5);
    for alpha in alphas {
        let mut gates = vec![1e38, -1e38, 1e-40, -1e-40, -2e-38, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        if alpha != 0.0 {
            for t in [2f32.powi(-26), -87.33655, -88.72284] {
                let x = t / alpha;
                if x.is_finite() && x != 0.0 {
                    gates.extend(near(x).chain(near(-x)));
                }
            }
        }
        gates.extend(near(gate_clip.0).chain(near(gate_clip.1)));
        gates.resize(gates.len().div_ceil(32) * 32, 0.5);
        let dim = gates.len();
        let gates = gates.into_iter().map(|gate| T::from(gate).unwrap()).collect::<Vec<_>>();
        let values =
            [1.0f32, 1e30, -1e20, 0.5].iter().cycle().take(dim).map(|&v| T::from(v).unwrap()).collect::<Vec<_>>();
        let factors = signs(dim, 3);
        let cases = [false, true]
            .into_iter()
            .flat_map(|hadamard| [None, Some(gate_clip)].map(|clip| (None, false, hadamard, Some(alpha), clip, None)))
            .map(|settings| {
                (settings, ActivationType::SILU, (&gates[..], &values[..], &factors[..]), (dim, 1, 1, dim + 2))
            })
            .collect::<Vec<_>>();
        let outputs = gpu_outputs(&fixture, &cases);
        assert_eq!(outputs.len(), cases.len(), "one output per case");
        for (&(settings, act, data, shape), gpu) in cases.iter().zip(outputs) {
            let (_, _, hadamard, alpha, gate_clip, value_clip) = settings;
            let label = format!("{:?} alpha {alpha:?} clip {gate_clip:?} hadamard {hadamard}", T::data_type());
            // Transforms of infinite products are NaN or infinite by class; their bounds come from the CPU.
            let products = product_bounds((data.0, data.1), act, (alpha, gate_clip, value_clip));
            let cpu = cpu_outputs(settings, act, data, shape).0;
            if hadamard && products.iter().any(|(_, center)| !center.is_finite()) {
                let classes = |values: &[T]| {
                    values
                        .iter()
                        .map(|v| {
                            (
                                v.is_nan(),
                                v.is_infinite() && v.is_sign_positive(),
                                v.is_infinite() && v.is_sign_negative(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(classes(&gpu.0), classes(&cpu), "{label}: special classes");
                continue;
            }
            check_bounds(&output_bounds::<T>(products, data.2, hadamard), &cpu, &gpu.0, &label);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn thresholds_match_oracle_all_types() {
    thresholds_match_oracle::<f32>();
    thresholds_match_oracle::<bf16>();
}

/// The products Vulkan and the CPU flush-free arithmetic agree on where FP32 hardware flushes: subnormal gates under
/// IDENTITY times 1e20 (normal products 0x1c2d78ec and BF16 0x20ad), a subnormal alpha with gates near 1e38 (a normal
/// exponent -alpha x), and a subnormal gate under alpha 1e37 times 1e30 (normal 7.4e-10). Exactly the CPU for IDENTITY;
/// within the staged oracle for SiLU, and nonzero.
#[uzu_test]
fn subnormal_operands_keep_normal_outputs() {
    let fixture = KernelFixture::new();
    let cases = [
        (
            ActivationType::IDENTITY,
            None,
            vec![0x0000_1000, 0x8000_1000, 0x0000_0001, 0x0020_0000],
            vec![1e20, 1e20, 1e30, 1e20],
        ),
        (
            ActivationType::SILU,
            Some(f32::from_bits(0x0010_0000)),
            vec![0x7e96_7699, 0xfe96_7699, 0x7db4_8e52, 0x7b40_97ce],
            vec![1.0; 4],
        ),
        (ActivationType::SILU, Some(1e37), vec![0x0010_0000, 0x8010_0000, 0x0000_0400, 0x8000_0001], vec![1e30; 4]),
        (ActivationType::SILU, Some(3e38), vec![0x80d9_c3ea, 0x80e0_0000, 0x8080_0000, 0x8100_0000], vec![1e30; 4]),
    ];
    for (act, alpha, gates, values) in cases {
        let gates = gates.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let factors = [1; 4];
        let case =
            ((None, false, false, alpha, None, None), act, (&gates[..], &values[..], &factors[..]), (4, 1, 0, 4));
        let gpu = gpu_outputs(&fixture, &[case]).remove(0).0;
        let cpu = cpu_outputs(case.0, act, case.2, case.3).0;
        let label = format!("{act:?} alpha {alpha:?}");
        check_bounds(&product_bounds((&gates, &values), act, (alpha, None, None)), &cpu, &gpu, &label);
        assert!(gpu.iter().zip(&cpu).all(|(g, c)| (*g != 0.0) == (*c != 0.0)), "{label}: Vulkan {gpu:?}, CPU {cpu:?}");
        if act == ActivationType::IDENTITY {
            assert_eq!(
                gpu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                cpu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{label}"
            );
        }
    }
    let bf16_gates = [0x0020, 0x8020, 0x0001].map(bf16::from_bits);
    let bf16_values = [bf16::from_f32(1e20); 3];
    let case = (
        (None, false, false, None, None, None),
        ActivationType::IDENTITY,
        (&bf16_gates[..], &bf16_values[..], &[1; 3][..]),
        (3, 1, 0, 3),
    );
    let bits = |values: Vec<bf16>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(gpu_outputs(&fixture, &[case]).remove(0).0), [0x20ad, 0xa0ad, 0x1e2d], "BF16 gates times 1e20");
    assert_eq!(bits(cpu_outputs(case.0, case.1, case.2, case.3).0), [0x20ad, 0xa0ad, 0x1e2d], "CPU");
    fixture.assert_clean();
}

/// The BF16 rounding of the activation and the product before the transform is observable: for the corpus, results
/// from the unrounded FP32 activation and product fall outside the staged bounds, which Vulkan meets.
#[uzu_test]
fn bf16_rounding_stages_are_observable() {
    let fixture = KernelFixture::new();
    let (dim, batch) = (128, 8);
    let (gates, values, factors) = (values::<bf16>(batch * dim, 11), values::<bf16>(batch * dim, 12), signs(dim, 5));
    for hadamard in [false, true] {
        let settings = (None, true, hadamard, None, None, None);
        let case = (settings, ActivationType::GELUApprox, (&gates[..], &values[..], &factors[..]), (dim, batch, 0, 0));
        let gpu = gpu_outputs(&fixture, &[case]).remove(0).0;
        let staged =
            output_bounds::<bf16>(product_bounds((&gates, &values), case.1, (None, None, None)), &factors, hadamard);
        check_bounds(&staged, &cpu_outputs(settings, case.1, case.2, case.3).0, &gpu, &format!("hadamard {hadamard}"));
        let unstaged = gates.iter().zip(&values).map(|(g, v)| {
            let product = round32(f64::from(*v) * oracle(f64::from(*g), case.1).1);
            ((product, product), product)
        });
        let unstaged = unstaged.collect::<Vec<_>>();
        let unstaged = match hadamard {
            true => transform_oracle(&unstaged, &factors, true, 32),
            false => unstaged,
        };
        let outside = staged.iter().zip(&unstaged).filter(|(((lo, hi), _), (_, value))| {
            let value = to::<bf16>(*value);
            value < *lo || value > *hi
        });
        let outside = outside.count();
        eprintln!("GatedActMul BF16 hadamard {hadamard}: {outside} unstaged results outside the staged bounds");
        assert!(outside > 0, "BF16 stages unobservable");
    }
    fixture.assert_clean();
}

/// Every model-admitted quantization (interleaved rows, the transform) and focused smaller raw power-of-two groups, of
/// every shape the groups divide, both types, the five activations in turn; and in ordinary shapes custom SiLU alphas
/// of both signs with gate and value clips on separate operands at an offset in longer rows, quantized with and without
/// sums in both layouts. A shape's cases share one command buffer. Each side's full-precision products without the
/// transform, which keep the BF16 activation and product roundings, lie within the staged product bounds; their FP32
/// transforms, each side's prequantization stage, within the propagated bounds. Vulkan's codes, scales and sums are
/// exactly the canonical quantization of Vulkan's stage, and the CPU kernel's of the CPU's stage; codes that differ
/// between the kernels, from faithfully differing activations, are counted and the first printed with their stages.
fn quantization_matches<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let mut differing = 0;
    let (sequential, grouped) = (Int8CodeLayout::Sequential, Int8CodeLayout::GroupedByNibble);
    let small = [(4, Some(2), grouped), (8, None, sequential), (32, Some(128), grouped)];
    let settings = quantizations().into_iter().map(raw).chain(small).collect::<Vec<_>>();
    for (batch, dim) in [(1, 0), (0, 128), (1, 128), (3, 384), (128, 4096)] {
        let (gates, values, factors) =
            (values::<T>(batch * dim, dim + 3), values::<T>(batch * dim, dim + 4), signs(dim, batch));
        let data = (&gates[..], &values[..], &factors[..]);
        let divides = |(scale_group, sum_group, _): &(usize, Option<usize>, Int8CodeLayout)| {
            dim % scale_group == 0 && dim % sum_group.unwrap_or(1) == 0
        };
        let mut cases = settings
            .iter()
            .filter(|setting| divides(setting))
            .enumerate()
            .map(|(i, &q)| {
                ((Some(q), true, true, None, None, None), ACTIVATIONS[i % ACTIVATIONS.len()], data, (dim, batch, 0, 0))
            })
            .collect::<Vec<_>>();
        if dim >= 128 && batch <= 3 {
            let separate = (dim, batch, 3, dim + 5);
            let clipped = [(64, None, sequential, 1.702, (-6.5, 7.0)), (128, Some(32), grouped, -0.7, (-1.0, 1.0))];
            cases.extend(clipped.map(|(scale_group, sum_group, layout, alpha, gate_clip)| {
                let settings = (
                    Some((scale_group, sum_group, layout)),
                    false,
                    true,
                    Some(alpha),
                    Some(gate_clip),
                    Some((-2.5, 2.5)),
                );
                (settings, ActivationType::SILU, data, separate)
            }));
        }
        let outputs = gpu_outputs(&fixture, &cases);
        assert_eq!(outputs.len(), cases.len(), "one output per case");
        if batch * dim == 0 {
            assert!(outputs.iter().all(|output| output.1.is_empty() && output.2.is_empty() && output.3.is_empty()));
            continue;
        }
        // The same settings at full precision without the transform give each side's products.
        let product_cases = cases
            .iter()
            .map(|&((_, interleaved, _, alpha, gate_clip, value_clip), act, data, shape)| {
                ((None, interleaved, false, alpha, gate_clip, value_clip), act, data, shape)
            })
            .collect::<Vec<_>>();
        let gpu_products = gpu_outputs(&fixture, &product_cases);
        assert_eq!(gpu_products.len(), cases.len(), "one product per case");
        let widen = |products: &[T]| products.iter().map(|p| p.to_f32().unwrap()).collect::<Vec<_>>();
        for ((&(settings, act, data, shape), gpu), (product_case, gpu_products)) in
            cases.iter().zip(outputs).zip(product_cases.iter().zip(gpu_products))
        {
            let (setting, alpha, gate_clip, value_clip) = (settings.0.unwrap(), settings.3, settings.4, settings.5);
            let label = format!("{:?} {act:?} {batch}x{dim} {} alpha {alpha:?}", T::data_type(), label(setting));
            let cpu_products = cpu_outputs(product_case.0, act, data, shape).0;
            let products = product_bounds((data.0, data.1), act, (alpha, gate_clip, value_clip));
            check_bounds(&products, &cpu_products, &gpu_products.0, &format!("{label} products"));
            let (gpu_products, cpu_products) = (widen(&gpu_products.0), widen(&cpu_products));
            let cpu_stage =
                (ActivationTransformOp::InputRht, false, None::<&[f32]>, None, &cpu_products[..], data.2, batch);
            let (gpu_stage, cpu_stage) =
                (gpu_transformed(&fixture, &gpu_products, data.2, batch), transform_cpu_outputs(cpu_stage).0);
            let stage_bounds = transform_oracle(&products, data.2, true, 32);
            check_bounds(&stage_bounds, &cpu_stage, &gpu_stage, &format!("{label} stage"));
            let gpu = (gpu.1, gpu.2, gpu.3);
            assert_quantized(&gpu, &quantized(&gpu_stage, dim, setting), &gpu_stage, &label);
            let cpu = cpu_outputs(settings, act, data, shape);
            let cpu = (cpu.1, cpu.2, cpu.3);
            assert_quantized(&cpu, &quantized(&cpu_stage, dim, setting), &cpu_stage, &format!("{label} CPU"));
            for logical in 0..gpu.0.len() {
                let (stored, group) = (logical - logical % 8 + setting.2.index(logical % 8), logical / setting.0);
                if gpu.0[stored] != cpu.0[stored] {
                    differing += 1;
                    if differing <= 8 {
                        eprintln!(
                            "GatedActMul {label}: code {logical}: Vulkan {} from {:e} at scale {:e}, CPU {} from {:e} at scale {:e}",
                            gpu.0[stored],
                            gpu_stage[logical],
                            gpu.1[group],
                            cpu.0[stored],
                            cpu_stage[logical],
                            cpu.1[group]
                        );
                    }
                }
            }
        }
    }
    eprintln!("GatedActMul {:?}: {differing} codes differ between the kernels", T::data_type());
    fixture.assert_clean();
}

#[uzu_test]
fn quantization_matches_all_types() {
    quantization_matches::<f32>();
    quantization_matches::<bf16>();
}

/// Construction rejects F16 and I32. `encode` rejects every optional argument missing where required or present where
/// not, also for empty rows, before recording anything; the same command buffer then completes valid work and leaves
/// the buffers of the rejected calls untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for data_type in [DataType::F16, DataType::I32] {
        let kernel = GatedActMulVulkanKernel::new(
            &fixture.context,
            data_type,
            GatedActMulOp::FullPrecision,
            false,
            true,
            false,
            32,
            32,
            false,
            false,
            false,
        );
        assert!(
            matches!(
                kernel,
                Err(Error::KernelVariant {
                    kernel: "GatedActMul",
                    ..
                })
            ),
            "{data_type:?}"
        );
    }
    let quantization = ActivationQuantization::new(128, 64, true, Int8CodeLayout::GroupedByNibble).map(raw);
    let without_sums = ActivationQuantization::new(64, 64, false, Int8CodeLayout::Sequential).map(raw);
    let settings = [
        (None, true, false, None, None, None),
        (None, false, true, Some(1.5f32), Some((-1.0f32, 1.0f32)), Some((-2.0f32, 2.0f32))),
        (quantization, true, true, None, None, Some((-2.0, 2.0))),
        (without_sums, true, true, Some(0.5), None, None),
    ];
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    for settings in settings {
        let kernel = vulkan_kernel::<f32>(&fixture, settings);
        let (quantization, interleaved, hadamard, alpha, gate_clip, value_clip) = settings;
        let quantized = quantization.is_some();
        let sums = quantization.is_some_and(|(_, sum_group, _)| sum_group.is_some());
        let buffers = [!interleaved, !quantized, quantized, quantized, sums, hadamard];
        let scalars =
            [alpha.is_some(), gate_clip.is_some(), gate_clip.is_some(), value_clip.is_some(), value_clip.is_some()];
        for (flip, dim) in (0..buffers.len() + scalars.len()).flat_map(|flip| [0, 128].map(|dim| (flip, dim))) {
            let mut present = [&buffers[..], &scalars[..]].concat();
            present[flip] = !present[flip];
            let buffer = |index: usize| present[index].then_some((&untouched, 0..512));
            let scalar = |index: usize| present[buffers.len() + index].then_some(1.0f32);
            let encode = AssertUnwindSafe(|| unsafe {
                // SAFETY: never dispatched: the optional-argument assertion fails before recording.
                kernel.encode(
                    (&untouched, 0..512),
                    buffer(0),
                    buffer(1),
                    buffer(2),
                    buffer(3),
                    buffer(4),
                    buffer(5),
                    dim,
                    1,
                    0,
                    dim,
                    ActivationType::SILU,
                    scalar(0),
                    scalar(1),
                    scalar(2),
                    scalar(3),
                    scalar(4),
                    &mut encoding,
                );
            });
            assert!(
                catch_unwind(encode).is_err(),
                "argument {flip} dim {dim} accepted, quantized {quantized} interleaved {interleaved}"
            );
        }
    }
    let (gates, values) = (values::<f32>(64, 5), values::<f32>(64, 6));
    let (act, _) = operands((&gates, &values), (64, 1, true, 0, 0));
    let (act_buffer, output) = (fixture.buffer(&act), fixture.buffer(&[0.0f32; 64]));
    let kernel = vulkan_kernel::<f32>(&fixture, (None, true, false, None, None, None));
    // SAFETY: the act operand holds one interleaved row of 64 values and gates, the output 64 floats.
    unsafe {
        kernel.encode(
            (&act_buffer, 0..512),
            None,
            Some((&output, 0..256)),
            None,
            None,
            None,
            None,
            64,
            1,
            0,
            0,
            ActivationType::SILU,
            None,
            None,
            None,
            None,
            None,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    let (output, untouched) = unsafe { (KernelFixture::read::<f32>(&output), KernelFixture::read::<u8>(&untouched)) };
    let bounds = product_bounds((&gates, &values), ActivationType::SILU, (None, None, None));
    let case = (
        (None, true, false, None, None, None),
        ActivationType::SILU,
        (&gates[..], &values[..], &[][..]),
        (64, 1, 0, 0),
    );
    check_bounds(&bounds, &cpu_outputs(case.0, case.1, case.2, case.3).0, &output, "valid work");
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// The binding's preconditions. Construction rejects quantization without the transform and quantization groups that
/// are not powers of two up to 128 (0, 3, 256; sum groups only with sums) before creating anything, ignores the groups
/// of full precision, and admits smaller raw powers of two. `encode` rejects transformed rows not a multiple of 32,
/// quantized rows not a multiple of the groups, and clip bounds out of order or NaN, also for empty batches, before
/// recording anything; the same command buffer then completes valid work, and rejected outputs stay untouched.
#[uzu_test]
fn rejects_violated_preconditions() {
    let fixture = KernelFixture::new();
    let (full, quantize, with_sums) =
        (GatedActMulOp::FullPrecision, GatedActMulOp::Quantize, GatedActMulOp::QuantizeWithGroupSums);
    let new = |ops, hadamard, scale_group, sum_group| {
        GatedActMulVulkanKernel::new(
            &fixture.context,
            DataType::F32,
            ops,
            false,
            true,
            hadamard,
            scale_group,
            sum_group,
            false,
            true,
            true,
        )
    };
    let invalid = [
        (quantize, false, 32, 32),
        (with_sums, false, 32, 32),
        (quantize, true, 0, 32),
        (quantize, true, 3, 32),
        (quantize, true, 256, 32),
        (with_sums, true, 64, 0),
        (with_sums, true, 64, 3),
        (with_sums, true, 64, 256),
    ];
    for (ops, hadamard, scale_group, sum_group) in invalid {
        let rejected = matches!(
            new(ops, hadamard, scale_group, sum_group),
            Err(Error::KernelPrecondition {
                kernel: "GatedActMul",
                ..
            })
        );
        assert!(rejected, "{ops:?} hadamard {hadamard} groups {scale_group}/{sum_group} accepted");
    }
    for (ops, hadamard, scale_group, sum_group) in
        [(full, false, 0, 3), (quantize, true, 1, 0), (with_sums, true, 2, 128)]
    {
        new(ops, hadamard, scale_group, sum_group).expect("an admitted setting");
    }
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    let kernels = [(full, true, 32, 32), (quantize, true, 64, 64), (with_sums, true, 32, 64), (full, false, 32, 32)]
        .map(|(ops, hadamard, scale_group, sum_group)| {
            (ops, new(ops, hadamard, scale_group, sum_group).expect("GatedActMul"))
        });
    let ordered = (Some(-1.0f32), Some(1.0f32));
    let cases = [
        (0, 48, 1, ordered, ordered),
        (1, 96, 1, ordered, ordered),
        (2, 96, 0, ordered, ordered),
        (3, 64, 1, (Some(1.0), Some(-1.0)), ordered),
        (3, 64, 0, ordered, (Some(f32::NAN), Some(1.0))),
        (3, 64, 1, (Some(-1.0), Some(f32::NAN)), ordered),
        (3, 0, 1, ordered, (Some(2.0), Some(1.0))),
    ];
    for (index, dim, batch, gate_clip, value_clip) in cases {
        let (ops, kernel) = &kernels[index];
        let quantized = *ops != full;
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the precondition fails before recording.
            kernel.encode(
                (&untouched, 0..512),
                None,
                (!quantized).then_some((&untouched, 0..512)),
                quantized.then_some((&untouched, 0..512)),
                quantized.then_some((&untouched, 0..512)),
                (*ops == with_sums).then_some((&untouched, 0..512)),
                (index < 3).then_some((&untouched, 0..512)),
                dim,
                batch,
                0,
                dim,
                ActivationType::SILU,
                None,
                gate_clip.0,
                gate_clip.1,
                value_clip.0,
                value_clip.1,
                &mut encoding,
            );
        });
        assert!(catch_unwind(encode).is_err(), "case {index} dim {dim} batch {batch} accepted");
    }
    let (gates, values) = (values::<f32>(64, 9), values::<f32>(64, 10));
    let (act, _) = operands((&gates, &values), (64, 1, true, 0, 0));
    let (act_buffer, output) = (fixture.buffer(&act), fixture.buffer(&[0.0f32; 64]));
    let clip = (-1.0f32, 1.0f32);
    // SAFETY: the act operand holds one interleaved row of 64 values and gates, the output 64 floats.
    unsafe {
        kernels[3].1.encode(
            (&act_buffer, 0..512),
            None,
            Some((&output, 0..256)),
            None,
            None,
            None,
            None,
            64,
            1,
            0,
            0,
            ActivationType::SILU,
            None,
            Some(clip.0),
            Some(clip.1),
            Some(clip.0),
            Some(clip.1),
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    let (output, untouched) = unsafe { (KernelFixture::read::<f32>(&output), KernelFixture::read::<u8>(&untouched)) };
    let settings = (None, true, false, None, Some(clip), Some(clip));
    let case = (settings, ActivationType::SILU, (&gates[..], &values[..], &[][..]), (64, 1, 0, 0));
    let bounds = product_bounds((&gates, &values), ActivationType::SILU, (None, Some(clip), Some(clip)));
    check_bounds(&bounds, &cpu_outputs(case.0, case.1, case.2, case.3).0, &output, "valid work");
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// Run alone: `cargo test ... gated_act_mul_test::throughput -- --ignored --nocapture`. Construction of the five
/// activation pipelines, then model-shaped rows with fresh unchanging inputs: the fused kernel against the real
/// two-dispatch composition (the product without the transform, then the transform in place or quantizing).
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(fixture: &KernelFixture) {
        let quantization = ActivationQuantization::new(128, 64, true, Int8CodeLayout::GroupedByNibble).map(raw);
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                vulkan_kernel::<T>(fixture, (quantization, true, true, None, None, None));
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!(
            "GatedActMul {:?} construction of five pipelines: first {first:?}, median of 11 {:?}",
            T::data_type(),
            construction[5]
        );
        let plain = vulkan_kernel::<T>(fixture, (None, true, false, None, None, None));
        let fused = [None, quantization].map(|q| vulkan_kernel::<T>(fixture, (q, true, true, None, None, None)));
        let transforms = [None, quantization].map(|q| {
            let (op, grouped, scale, sum, in_place) = match q {
                None => (ActivationTransformOp::InputRht, false, 32, 32, true),
                Some((scale, sum, layout)) => (
                    ActivationTransformOp::QuantizeWithGroupSums,
                    layout.is_grouped_by_nibble(),
                    scale as u32,
                    sum.unwrap() as u32,
                    false,
                ),
            };
            ActivationTransformVulkanKernel::new(
                &fixture.context,
                T::data_type(),
                T::data_type(),
                op,
                grouped,
                in_place,
                scale,
                sum,
                false,
            )
            .expect("ActivationTransform")
        });
        for rows in [1, 128, 1024] {
            for dim in [4096, 14336] {
                let n = rows * dim;
                let (act, factors) = (fixture.buffer(&values::<T>(2 * n, 1)), fixture.buffer(&signs(dim, 1)));
                let (fp, codes) = (fixture.buffer(&vec![T::zero(); n]), fixture.buffer(&vec![0i8; n]));
                let (scales, sums) = (fixture.buffer(&vec![0f32; n / 128]), fixture.buffer(&vec![0i32; n / 64]));
                let bytes = |size: usize, divisor: usize| 0..(n / divisor * size) as u64;
                let factor_range = 0..(dim * 4) as u64;
                let t = size_of::<T>();
                for (index, label) in ["full precision", "quantized (128, 64)"].into_iter().enumerate() {
                    let quantized = index == 1;
                    let encode_fused = |encoding: &mut _| unsafe {
                        // SAFETY: every range holds the elements of `rows` interleaved rows; outputs do not alias.
                        fused[index].encode(
                            (&act, bytes(t, 1).start..bytes(t, 1).end * 2),
                            None,
                            (!quantized).then(|| (&fp, bytes(t, 1))),
                            quantized.then(|| (&codes, bytes(1, 1))),
                            quantized.then(|| (&scales, bytes(4, 128))),
                            quantized.then(|| (&sums, bytes(4, 64))),
                            Some((&factors, factor_range.clone())),
                            dim as u32,
                            rows as u32,
                            0,
                            0,
                            ActivationType::SILU,
                            None,
                            None,
                            None,
                            None,
                            None,
                            encoding,
                        );
                    };
                    let (gpu, wall) = fixture.median_times(encode_fused);
                    let (composed_gpu, composed_wall) = fixture.median_times(|encoding| unsafe {
                        // SAFETY: as above; the transform reads the products the first dispatch wrote.
                        plain.encode(
                            (&act, 0..(2 * n * t) as u64),
                            None,
                            Some((&fp, bytes(t, 1))),
                            None,
                            None,
                            None,
                            None,
                            dim as u32,
                            rows as u32,
                            0,
                            0,
                            ActivationType::SILU,
                            None,
                            None,
                            None,
                            None,
                            None,
                            encoding,
                        );
                        transforms[index].encode(
                            quantized.then(|| (&fp, bytes(t, 1))),
                            (!quantized).then(|| (&fp, bytes(t, 1))),
                            None,
                            quantized.then(|| (&codes, bytes(1, 1))),
                            quantized.then(|| (&scales, bytes(4, 128))),
                            quantized.then(|| (&sums, bytes(4, 64))),
                            (&factors, factor_range.clone()),
                            rows as u32,
                            dim as u32,
                            encoding,
                        );
                    });
                    eprintln!(
                        "MEASURE GatedActMul {:?} {rows}x{dim} {label} SiLU: median of 10 after 3 warm-up: fused GPU {gpu:?} wall {wall:?}; product then transform GPU {composed_gpu:?} wall {composed_wall:?}",
                        T::data_type()
                    );
                }
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
