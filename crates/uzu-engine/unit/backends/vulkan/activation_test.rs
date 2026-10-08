use std::{
    f64::consts::{FRAC_1_SQRT_2, PI},
    fmt::Debug,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, gpu_types::ActivationType, kernel::ActivationKernel},
        cpu::Cpu,
        vulkan::{ActivationVulkanKernel, Error},
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
/// Output elements past `n`, which every dispatch must leave untouched.
const TAIL: usize = 7;

/// The CPU kernel through the shared trait; CPU buffers cannot be empty.
fn cpu_output<T: ArrayElement + Float>(
    input: &[T],
    activation: ActivationType,
) -> Vec<T> {
    if input.is_empty() {
        return Vec::new();
    }
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, T::data_type(), false)
        .expect("CPU Activation");
    let input_buffer = create_buffer_with_data::<Cpu, T>(&context, input);
    let mut output = create_buffer_with_data::<Cpu, T>(&context, input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(Some(&input_buffer), &mut output, input.len() as u32, activation, &mut command_buffer);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&output)
}

/// The out-of-place and in-place Vulkan kernels.
fn kernels<T: ArrayElement>(fixture: &KernelFixture) -> [ActivationVulkanKernel; 2] {
    [false, true]
        .map(|in_place| ActivationVulkanKernel::new(&fixture.context, T::data_type(), in_place).expect("Activation"))
}

/// Records every `(input, activation, in_place)` case into one command buffer, over guarded ranges whose outputs
/// extend `TAIL` elements past `n`. Returns each case's `n` outputs after checking the guards, the untouched tail and
/// the unchanged read-only input.
fn gpu_outputs<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernels: &[ActivationVulkanKernel; 2],
    cases: &[(&[T], ActivationType, bool)],
) -> Vec<Vec<T>> {
    let sentinel = T::from(-7.0).unwrap();
    let buffers = cases
        .iter()
        .map(|&(input, _, in_place)| {
            let initial = if in_place {
                input.to_vec()
            } else {
                vec![sentinel; input.len()]
            };
            let output = fixture.guarded(&[initial, vec![sentinel; TAIL]].concat(), sentinel);
            (output, (!in_place).then(|| fixture.guarded(input, sentinel)))
        })
        .collect::<Vec<_>>();
    let mut encoding = fixture.encoding();
    for (&(input, activation, in_place), (output, input_buffer)) in cases.iter().zip(&buffers) {
        // SAFETY: the input range holds `n` aligned `T`s and the output range `n + TAIL`, so every index `< n` is
        // inside both; the output aliases only itself.
        unsafe {
            kernels[usize::from(in_place)].encode(
                input_buffer.as_ref().map(|(buffer, range)| (buffer, range.clone())),
                (&output.0, output.1.clone()),
                input.len() as u32,
                activation,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    cases
        .iter()
        .zip(&buffers)
        .map(|(&(input, ..), (output, input_buffer))| {
            // SAFETY: the only command buffer using these buffers has completed.
            let values = unsafe {
                if let Some(input_buffer) = input_buffer {
                    KernelFixture::assert_unchanged(input_buffer, sentinel, input, "input");
                }
                KernelFixture::read_guarded(output, sentinel)
            };
            let tail = bytemuck::cast_slice::<T, u8>(&values[input.len()..]);
            assert_eq!(tail, bytemuck::cast_slice::<T, u8>(&[sentinel; TAIL]), "an element past n was written");
            values[..input.len()].to_vec()
        })
        .collect()
}

/// FP32 ULP at `value`, at least the smallest normal's, so a correctly rounded result is within half of it.
fn ulp32(value: f64) -> f64 {
    match value.is_finite() {
        true => 2f64.powi(value.abs().max(f32::MIN_POSITIVE as f64).log2().floor() as i32) * f32::EPSILON as f64,
        false => 0.0,
    }
}

fn round32(value: f64) -> f64 {
    value as f32 as f64
}

/// The FP32 values an operation with exact result in `[a, b]` and an error of `units` ULPs may return: the outward
/// endpoints rounded to FP32, which is monotonic; infinite endpoints stay.
fn interval(
    a: f64,
    b: f64,
    units: f64,
) -> (f64, f64) {
    let (lo, hi) = (a.min(b), a.max(b));
    let widen = |value: f64| {
        if value.is_finite() {
            units * ulp32(value)
        } else {
            0.0
        }
    };
    (round32(lo - widen(lo)), round32(hi + widen(hi)))
}

/// Bounds of the shader's tanh at an FP32 argument, always within [-1, 1] up to rounding. Below 0.1 the polynomial,
/// within its 5.5e-8 relative truncation and four FP32 roundings; below log(3) / 2 the builtin, whose precision Vulkan
/// inherits from (e^a - e^-a) / (e^a + e^-a): exponentials within 3 + 2|a| ULPs, the sum and difference rounded and the
/// quotient within 2.5 ULPs; beyond, 1 - 2 / (e^2|a| + 1) with the exponential within 3 + 4|a| ULPs, the sum rounded,
/// the quotient within 2.5 ULPs and the difference rounded, which saturates exactly once the exponential overflows.
/// LogitTransform's soft cap shares the tanh.
pub fn tanh_interval(a: f64) -> (f64, f64) {
    let magnitude = a.abs();
    let t = magnitude.tanh();
    let (lo, hi) = if magnitude < 0.1 {
        let error = 5.5e-8 * t + 2.0 * ulp32(t);
        interval(t - error, t + error, 0.0)
    } else if magnitude < round32(0.54930615) {
        let error = (4.0 + 2.0 * magnitude) * f32::EPSILON as f64 * (1.0 + t) + 2.5 * ulp32(t);
        interval(t - error, t + error, 0.0)
    } else {
        let e = (2.0 * magnitude).exp();
        let e = interval(e, e, 3.0 + 4.0 * magnitude);
        let d = interval(e.0 + 1.0, e.1 + 1.0, 0.0);
        let q = interval(2.0 / d.1, 2.0 / d.0, 2.5);
        interval(1.0 - q.1, 1.0 - q.0, 0.0)
    };
    match a.is_sign_negative() {
        true => (-hi, -lo),
        false => (lo, hi),
    }
}

/// Bounds of the shader's log of an FP32 sum s >= 1: below 2 the atanh series in w = (s - 1) / (2 + s - 1), within
/// 5 FP32 epsilons relative (3 from w's sum and quotient, under 1 from the series' roundings and squared w, 1/2 from
/// the final product, 1/4 from the truncation); from 2 on the builtin, within 3 ULPs.
fn log_interval(s: f64) -> (f64, f64) {
    let y = s.ln();
    match s < 2.0 {
        true => interval(y * (1.0 - 5.0 * f32::EPSILON as f64), y * (1.0 + 5.0 * f32::EPSILON as f64), 0.0),
        false => interval(y, y, 3.0),
    }
}

/// Inputs up to 2^-26 in magnitude, where SiLU and both GELUs return x / 2.
const TINY: u32 = 0x3280_0000;

/// The CPU's staged FP32 arithmetic for one FP32 input, as the bounds of every value Vulkan may compute and the value
/// with correctly rounded transcendentals. FP32 additions and products are correctly rounded on both sides; Vulkan
/// allows exp 3 + 2|x| ULPs, division 2.5 ULPs (also kept past 2^126, where Vulkan requires nothing, to hold the CPU's
/// tail), and the contraction of a product into the following addition. GELUExact's erf is the shader's faithfully
/// rounded polynomial (2 ULPs with one more rounding), with its large-input exponential's error. Tiny inputs halve
/// exactly; no FP32 subnormal may flush.
fn oracle(
    x: f64,
    activation: ActivationType,
) -> ((f64, f64), f64) {
    // Exponentials of infinities are exact.
    let exp = |argument: f64| {
        let (value, units) = (
            argument.exp(),
            if argument.is_finite() {
                3.0 + 2.0 * argument.abs()
            } else {
                0.0
            },
        );
        (interval(value, value, units), round32(value))
    };
    let halves = matches!(activation, ActivationType::SILU | ActivationType::GELUApprox | ActivationType::GELUExact);
    if halves && (x as f32).to_bits() & 0x7fff_ffff <= TINY {
        let half = round32(x / 2.0);
        return ((half, half), half);
    }
    match activation {
        ActivationType::SILU => {
            let (e, e_center) = exp(-x);
            let d = interval(1.0 + e.0, 1.0 + e.1, 0.0);
            (interval(x / d.0, x / d.1, 2.5), round32(x / round32(1.0 + e_center)))
        },
        ActivationType::GELUApprox => {
            let (k0, k1) = (round32(0.044715), round32((2.0 / PI).sqrt()));
            let square = round32(round32(k0 * x) * x);
            let (unfused, fused) = (round32(round32(square * x) + x), round32(square.mul_add(x, x)));
            let (fused_t, unfused_t) = (tanh_interval(round32(k1 * fused)), tanh_interval(round32(k1 * unfused)));
            let t = (fused_t.0.min(unfused_t.0), fused_t.1.max(unfused_t.1));
            let (half, u) = (round32(0.5 * x), interval(1.0 + t.0, 1.0 + t.1, 0.0));
            let center = round32(half * round32(1.0 + round32(round32(k1 * unfused).tanh())));
            (interval(half * u.0, half * u.1, 0.0), center)
        },
        ActivationType::GELUExact => {
            let z = round32(x * round32(FRAC_1_SQRT_2));
            let (mut erf, tail) = (libm::erf(z), libm::erfc(z.abs()));
            let mut error = 2.0 * ulp32(erf);
            if z.abs() >= 4.0 {
                (erf, error) = (1f64.copysign(z), 0.0);
            } else if z.abs() > 0.927734375 {
                error += (3.0 + 2.0 * tail.ln().abs()) * ulp32(tail);
            }
            let e = interval(erf - error, erf + error, 0.0);
            let (half, u) = (round32(0.5 * x), interval(1.0 + e.0, 1.0 + e.1, 0.0));
            (interval(half * u.0, half * u.1, 0.0), round32(half * round32(1.0 + round32(erf))))
        },
        ActivationType::SOFTPLUS if x <= 20.0 || x.is_nan() => {
            let (e, e_center) = exp(x);
            let s = interval(1.0 + e.0, 1.0 + e.1, 0.0);
            ((log_interval(s.0).0, log_interval(s.1).1), round32(round32(1.0 + e_center).ln()))
        },
        ActivationType::IDENTITY | ActivationType::SOFTPLUS => ((x, x), x),
    }
}

/// Checks the CPU and Vulkan outputs against the oracle, after rounding its bounds to `T`: NaN where it is NaN, its
/// exact infinities, the oracle's zero sign wherever both are zero, and otherwise within its bounds, which must be finite
/// for every finite result. Returns the largest `[Vulkan, CPU]` error relative to the bound.
fn check<T: ArrayElement + Float + Debug>(
    input: &[T],
    activation: ActivationType,
    cpu: &[T],
    gpu: &[T],
    case: &str,
) -> [f64; 2] {
    assert_eq!((cpu.len(), gpu.len()), (input.len(), input.len()), "{case}: output lengths");
    let to_t = |value: f64| T::from(value).unwrap().to_f64().unwrap();
    let mut worst = [0.0f64; 2];
    let mut violations = 0;
    for (index, ((&x, &cpu), &gpu)) in input.iter().zip(cpu).zip(gpu).enumerate() {
        let x = x.to_f64().unwrap();
        let ((lo, hi), center) = oracle(x, activation);
        let (lo, hi, center) = (to_t(lo), to_t(hi), to_t(center));
        let bound = (center - lo).max(hi - center);
        assert!(!center.is_finite() || bound.is_finite(), "{case}: element {index}: x {x:e}: unbounded oracle");
        for (slot, value) in [gpu, cpu].into_iter().enumerate() {
            let value = value.to_f64().unwrap();
            let valid = match center.is_finite() {
                false => center.is_nan() && value.is_nan() || center == value,
                true if center == 0.0 && value == 0.0 => center.is_sign_negative() == value.is_sign_negative(),
                true => lo <= value && value <= hi,
            };
            if valid && center.is_finite() {
                let error = (value - center).abs();
                worst[slot] = worst[slot].max(if bound > 0.0 {
                    error / bound
                } else {
                    error
                });
            }
            if !valid {
                violations += 1;
                if violations <= 5 {
                    let side = ["Vulkan", "CPU"][slot];
                    eprintln!(
                        "{case}: element {index}: x {x:e}: {side} {value:e}, oracle {center:e} in [{lo:e}, {hi:e}]"
                    );
                }
            }
        }
    }
    assert_eq!(violations, 0, "{case}: {violations} results outside the oracle bounds");
    worst
}

/// The ordinals GELUExact's 16-bit Vulkan result may take in the negative cancellation tail -4 < z < -0.927734375,
/// z = round32(x / sqrt 2): the CPU's erf and the shader's polynomial, both faithful, may round erf to adjacent FP32
/// values, which 1 + erf keeps, so y may move by e = |round32(x / 2)| spacing32(erf). The range is
/// [floor_T(cpu - e) - 2, ceil_T(cpu + e) + 2] in storage steps; outside the tail it is empty.
fn gelu_exact_allowance<T: ArrayElement + Float>(
    x: T,
    cpu: T,
) -> Option<(i64, i64)> {
    let x = x.to_f64().unwrap();
    let z = round32(x * round32(FRAC_1_SQRT_2));
    if !(-4.0 < z && z < -0.927734375) {
        return None;
    }
    let (cpu, e) = (cpu.to_f64().unwrap(), round32(0.5 * x).abs() * ulp32(libm::erf(z)));
    let rounded = |value: f64, outward: i64| {
        let nearest = T::from(value).unwrap();
        let inside = (nearest.to_f64().unwrap() - value) * (outward as f64) < 0.0;
        KernelFixture::ordinal(nearest) + outward * i64::from(inside)
    };
    Some((rounded(cpu - e, -1) - 2, rounded(cpu + e, 1) + 2))
}

/// Counts GELUExact's 16-bit results over two storage steps from the CPU, as `[within the conditioned allowance,
/// violations]`.
fn gelu_exact_conditioned<T: ArrayElement + Float>(
    input: &[T],
    cpu: &[T],
    gpu: &[T],
) -> [usize; 2] {
    assert_eq!((cpu.len(), gpu.len()), (input.len(), input.len()), "GELUExact output lengths");
    let mut counts = [0; 2];
    for ((&x, &cpu), &gpu) in input.iter().zip(cpu).zip(gpu) {
        let ordinal = KernelFixture::ordinal(gpu);
        // NaN placement is checked by `compare`; payloads are not ordered.
        if !cpu.is_nan() && (ordinal - KernelFixture::ordinal(cpu)).abs() > 2 {
            let allowed = gelu_exact_allowance(x, cpu).is_some_and(|(lo, hi)| (lo..=hi).contains(&ordinal));
            counts[usize::from(!allowed)] += 1;
        }
    }
    counts
}

/// Checks every activation of `input` on both layouts in one command buffer, returning per-activation labels with
/// their storage-step differences from the CPU and the oracle maxima. Identity must copy the raw input bits; 16-bit
/// GELUExact results over two steps count as violations only outside their conditioned allowance.
fn check_all<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernels: &[ActivationVulkanKernel; 2],
    input: &[T],
    case: &str,
) -> Vec<(String, [f64; 4], [f64; 2])> {
    let cases = ACTIVATIONS.iter().flat_map(|&activation| [false, true].map(|in_place| (input, activation, in_place)));
    let cases = cases.collect::<Vec<_>>();
    let outputs = gpu_outputs(fixture, kernels, &cases);
    assert_eq!(outputs.len(), cases.len(), "{case}: one output per case");
    let mut results = Vec::new();
    for (&(_, activation, in_place), gpu) in cases.iter().zip(&outputs) {
        let case = format!("{case} {:?} {activation:?} in_place {in_place}", T::data_type());
        if activation == ActivationType::IDENTITY {
            assert!(bytemuck::cast_slice::<T, u8>(gpu) == bytemuck::cast_slice::<T, u8>(input), "{case}: bits differ");
            continue;
        }
        let cpu = cpu_output(input, activation);
        let label = format!("{:?} {activation:?}", T::data_type());
        // Storage steps for 16-bit types, at most two outside GELUExact's conditioned allowance; FP32 differences are
        // only reported.
        let mut steps = KernelFixture::compare(&cpu, gpu, &case, f64::INFINITY, 0.0);
        if activation == ActivationType::GELUExact && size_of::<T>() == 2 {
            let [conditioned, violations] = gelu_exact_conditioned(input, &cpu, gpu);
            eprintln!("{case}: {conditioned} results over two steps within the conditioned erf allowance");
            steps[3] = violations as f64;
        }
        results.push((label, steps, check(input, activation, &cpu, gpu, &case)));
    }
    results
}

/// Prints the oracle maxima per label, then reports the storage steps, failing on 16-bit results over two steps
/// outside GELUExact's conditioned allowance; the printed maxima include conditioned results.
fn report(results: Vec<(String, [f64; 4], [f64; 2])>) {
    let mut oracle = std::collections::BTreeMap::<String, [f64; 2]>::new();
    for (label, _, [gpu, cpu]) in &results {
        let worst = oracle.entry(label.clone()).or_default();
        *worst = [worst[0].max(*gpu), worst[1].max(*cpu)];
    }
    for (label, [gpu, cpu]) in &oracle {
        eprintln!("Activation {label}: max error / oracle bound: Vulkan {gpu:.3e}, CPU {cpu:.3e}");
    }
    KernelFixture::report("Activation", results.into_iter().map(|(label, steps, _)| (label, steps)));
}

/// Mixed-sign inputs from -25 to 25 in steps of 1/40, so every activation leaves the identity and both tails.
fn spread<T: Float>(n: usize) -> Vec<T> {
    (0..n).map(|i| T::from((i * 2_654_435_761 % 2001) as f32 / 40.0 - 25.0).unwrap()).collect()
}

/// Every 16-bit pattern, or for FP32 every 4099th pattern (each exponent of both signs plus NaN payloads), all of
/// [-89, -87] for SiLU's exponential overflow and its division past 2^126, the smallest subnormals and the subnormal
/// boundary, and 64 ULPs around the tiny-input threshold, -20, -17, -16 and -10, the erf branch at
/// |x| / sqrt 2 = 0.927734375, both GELUApprox tanh branch thresholds and Softplus's 20.
fn corpus<T: ArrayElement + Float>() -> Vec<T> {
    if size_of::<T>() == 2 {
        return bytemuck::pod_collect_to_vec(&(0..=u16::MAX).collect::<Vec<_>>());
    }
    let (k0, k1) = (0.044715f32 as f64, (2.0 / PI).sqrt() as f32 as f64);
    let gelu_input = |a: f64| {
        let mut x = a / k1;
        for _ in 0..4 {
            x -= (x + k0 * x.powi(3) - a / k1) / (1.0 + 3.0 * k0 * x.powi(2));
        }
        x as f32
    };
    let mut bits = (0..=u32::MAX).step_by(4099).collect::<Vec<_>>();
    bits.extend((-87.0f32).to_bits()..=(-89.0f32).to_bits());
    bits.extend((0..64).chain(0x007f_ffc0..0x0080_0040));
    let erf_branch = (0.927734375 / FRAC_1_SQRT_2) as f32;
    let tiny = f32::from_bits(TINY);
    for center in [tiny, -20.0, -17.0, -16.0, -10.0, erf_branch, gelu_input(0.1), gelu_input(0.54930615), 20.0] {
        for center in [center, -center] {
            bits.extend(center.to_bits() - 64..=center.to_bits() + 64);
        }
    }
    bytemuck::pod_collect_to_vec(&bits)
}

/// Every length from empty to a partial last workgroup past 65536, every activation on both layouts interleaved in one
/// command buffer per length: exact spans, guards, tails and inputs, and results within the oracle bounds.
fn layouts_and_spans<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let kernels = kernels::<T>(&fixture);
    let results = [0, 1, 31, 33, 257, 65537]
        .into_iter()
        .flat_map(|n| check_all(&fixture, &kernels, &spread::<T>(n), &format!("n {n}")))
        .collect();
    report(results);
    fixture.assert_clean();
}

#[uzu_test]
fn f32_silu_and_identity() {
    let fixture = KernelFixture::new();
    let kernels = kernels::<f32>(&fixture);
    let input = spread::<f32>(257);
    let cases = [ActivationType::SILU, ActivationType::IDENTITY].map(|activation| (&input[..], activation, false));
    let outputs = gpu_outputs(&fixture, &kernels, &cases);
    check(&input, ActivationType::SILU, &cpu_output(&input, ActivationType::SILU), &outputs[0], "F32 SILU");
    assert!(outputs[1] == input, "F32 IDENTITY");
    fixture.assert_clean();
}

#[uzu_test]
fn layouts_and_spans_all_types() {
    layouts_and_spans::<f32>();
    layouts_and_spans::<f16>();
    layouts_and_spans::<bf16>();
}

/// The corpus through every activation on both layouts.
fn matches_oracle<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    report(check_all(&fixture, &kernels::<T>(&fixture), &corpus::<T>(), "corpus"));
    fixture.assert_clean();
}

#[uzu_test]
fn matches_oracle_all_types() {
    matches_oracle::<f32>();
    matches_oracle::<f16>();
    matches_oracle::<bf16>();
}

fn identity_preserves<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    input: &[T],
) {
    let kernels = kernels::<T>(fixture);
    for in_place in [false, true] {
        let outputs = gpu_outputs(fixture, &kernels, &[(input, ActivationType::IDENTITY, in_place)]);
        let same = bytemuck::cast_slice::<T, u8>(&outputs[0]) == bytemuck::cast_slice::<T, u8>(input);
        assert!(same, "{:?} identity in_place {in_place} changed bits", T::data_type());
    }
}

/// Identity copies raw bits on both layouts: all 16-bit patterns, and FP32 signed zeros, subnormals, infinities and
/// quiet and signaling NaNs with payloads.
#[uzu_test]
fn identity_preserves_bits() {
    let fixture = KernelFixture::new();
    let f32_bits = [0, 0x8000_0000, 1, 0x807f_ffff, 0x3f80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001];
    let f32_bits = [&f32_bits[..], &[0x7fa5_a5a5, 0xffc0_0001, 0xff80_0001, 0x7f7f_ffff]].concat();
    identity_preserves(&fixture, &bytemuck::pod_collect_to_vec::<u32, f32>(&f32_bits));
    identity_preserves(&fixture, &corpus::<f16>());
    identity_preserves(&fixture, &corpus::<bf16>());
    fixture.assert_clean();
}

/// Construction rejects a non-float type; `encode` rejects an input that contradicts `in_place`, also when empty,
/// before recording anything, and the same command buffer then completes valid work while rejected outputs stay
/// untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    assert!(matches!(
        ActivationVulkanKernel::new(&fixture.context, DataType::I32, false),
        Err(Error::KernelVariant {
            kernel: "Activation",
            ..
        })
    ));
    let kernels = kernels::<f32>(&fixture);
    let values = fixture.buffer(&[3.0f32; 33]);
    let mut encoding = fixture.encoding();
    for (in_place, n) in [(false, 33), (false, 0), (true, 33), (true, 0)] {
        let wrong = in_place.then_some((&values, 0..132));
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the optional-argument assertion fails before recording.
            kernels[usize::from(in_place)].encode(wrong, (&values, 0..132), n, ActivationType::SILU, &mut encoding);
        });
        assert!(catch_unwind(encode).is_err(), "in_place {in_place} n {n} accepted a wrong optional input");
    }
    let output = fixture.buffer(&[0.0f32; 33]);
    // SAFETY: both ranges hold 33 floats and do not alias.
    unsafe {
        kernels[0].encode(Some((&values, 0..132)), (&output, 0..132), 33, ActivationType::IDENTITY, &mut encoding)
    };
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    unsafe {
        assert_eq!(KernelFixture::read::<f32>(&values), [3.0; 33]);
        assert_eq!(KernelFixture::read::<f32>(&output), [3.0; 33]);
    }
    fixture.assert_clean();
}

/// The shader's erf returns ±1 from |z| = 4 on, as the CPU's libm erff does for 4096 values on each side of 4, every
/// 4099th value above, the largest finite and infinity; NaN stays NaN. There GELUExact's oracle collapses to exactly the
/// CPU result: x for positive and -0 for negative finite inputs, and the staged -inf * 0 = NaN at -inf.
#[uzu_test]
fn erf_saturation_is_exact() {
    let four = 4.0f32.to_bits();
    let saturated = (four..four + 4096).chain((four..0x7f80_0000).step_by(4099)).chain([0x7f7f_ffff, 0x7f80_0000]);
    for z in saturated.map(f32::from_bits) {
        assert_eq!(libm::erff(z).to_bits(), 1f32.to_bits(), "erff({z:e})");
        assert_eq!(libm::erff(-z).to_bits(), (-1f32).to_bits(), "erff({:e})", -z);
    }
    assert!(libm::erff(f32::NAN).is_nan());
    let threshold = (4.0 / FRAC_1_SQRT_2) as f32;
    let inputs = (threshold.to_bits() - 64..threshold.to_bits() + 64).chain([0x7e96_7699, 0x7f7f_ffff, 0x7f80_0000]);
    for x in inputs.map(f32::from_bits).flat_map(|x| [x, -x]) {
        if round32(f64::from(x) * round32(FRAC_1_SQRT_2)).abs() < 4.0 {
            continue;
        }
        let cpu = ActivationType::GELUExact.activate(x);
        let ((lo, hi), center) = oracle(f64::from(x), ActivationType::GELUExact);
        if x == f32::NEG_INFINITY {
            assert!(cpu.is_nan() && center.is_nan(), "GELUExact at -inf: CPU {cpu:e}, oracle {center:e}");
            continue;
        }
        let expected = if x > 0.0 {
            x
        } else {
            -0.0
        };
        assert_eq!(cpu.to_bits(), expected.to_bits(), "CPU GELUExact x {x:e}");
        assert!(lo == hi && hi == center && center as f32 == expected, "oracle at x {x:e}: [{lo:e}, {hi:e}]");
        assert_eq!((center as f32).is_sign_negative(), expected.is_sign_negative(), "oracle zero sign at x {x:e}");
    }
}

/// GELUExact's conditioned 16-bit allowance admits the measured Vulkan results and nothing beyond: moving every
/// tail result one storage step outside its allowance, on either side, makes each a violation.
#[uzu_test]
fn gelu_exact_allowance_rejects_larger_errors() {
    let fixture = KernelFixture::new();
    let input = corpus::<f16>();
    let gpu = gpu_outputs(&fixture, &kernels::<f16>(&fixture), &[(&input[..], ActivationType::GELUExact, false)]);
    let cpu = cpu_output(&input, ActivationType::GELUExact);
    assert_eq!(gelu_exact_conditioned(&input, &cpu, &gpu[0])[1], 0, "measured results");
    let from_ordinal = |ordinal: i64| {
        let bits = if ordinal < 0 {
            0x8000 | (-ordinal) as u16
        } else {
            ordinal as u16
        };
        f16::from_bits(bits)
    };
    let tail = input.iter().zip(&cpu).filter_map(|(&x, &cpu)| gelu_exact_allowance(x, cpu)).collect::<Vec<_>>();
    assert!(!tail.is_empty());
    for side in [0, 1] {
        let mut beyond = gpu[0].clone();
        for ((x, cpu), value) in input.iter().zip(&cpu).zip(&mut beyond) {
            if let Some((lo, hi)) = gelu_exact_allowance(*x, *cpu) {
                *value = from_ordinal([lo - 1, hi + 1][side]);
            }
        }
        assert_eq!(gelu_exact_conditioned(&input, &cpu, &beyond), [0, tail.len()], "side {side}");
    }
    fixture.assert_clean();
}

/// The shader's tiny-input path holds for the canonical CPU math: up to 2^-26 in magnitude SiLU and both GELUs return
/// x / 2 rounded to nearest even, for every subnormal and smallest-normal pattern, 64 Ki patterns on each side of the
/// threshold and every 4099th pattern between, with both signs.
#[uzu_test]
fn tiny_inputs_halve_exactly_on_cpu() {
    let magnitudes = (0..0x0100_0000).chain((0x0100_0000..TINY - 0x1_0000).step_by(4099));
    let mut above = [0usize; 2];
    for magnitude in magnitudes.chain(TINY - 0x1_0000..=TINY + 0x1_0000) {
        for x in [magnitude, magnitude | 0x8000_0000].map(f32::from_bits) {
            let half = (f64::from(x) / 2.0) as f32;
            for activation in [ActivationType::SILU, ActivationType::GELUApprox, ActivationType::GELUExact] {
                let same = activation.activate(x).to_bits() == half.to_bits();
                match magnitude <= TINY {
                    true => assert!(same, "{activation:?} x {x:e}: {:e}, not {half:e}", activation.activate(x)),
                    false => above[usize::from(same)] += 1,
                }
            }
        }
    }
    eprintln!("Activation above 2^-26: {} of {} results are still x / 2", above[1], above[0] + above[1]);
}

/// Run alone: `cargo test ... activation_test::throughput -- --ignored --nocapture`. Construction of both layouts'
/// kernels, then model-shaped flat vectors and a one-element submission floor, against the canonical CPU math on the
/// same values.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(fixture: &KernelFixture) {
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                kernels::<T>(fixture);
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!(
            "Activation {:?} construction of both layouts: first {first:?}, median of 11 {:?}",
            T::data_type(),
            construction[5]
        );
        let kernel = &kernels::<T>(fixture)[0];
        for n in [1, 128 * 4096, 1024 * 4096] {
            let values = spread::<T>(n);
            let (input, output) = (fixture.buffer(&values), fixture.buffer(&vec![T::zero(); n]));
            let bytes = 0..(n * size_of::<T>()) as u64;
            for activation in ACTIVATIONS {
                let (gpu, wall) = fixture.median_times(|encoding| {
                    // SAFETY: input and output each hold `n` elements and do not alias.
                    unsafe {
                        kernel.encode(
                            Some((&input, bytes.clone())),
                            (&output, bytes.clone()),
                            n as u32,
                            activation,
                            encoding,
                        )
                    };
                });
                let mut cpu = (0..11)
                    .map(|_| {
                        let start = Instant::now();
                        std::hint::black_box(values.iter().map(|&x| activation.activate(x)).collect::<Vec<_>>());
                        start.elapsed()
                    })
                    .collect::<Vec<_>>();
                cpu.sort();
                eprintln!(
                    "Activation {:?} {activation:?} n {n}: median of 10 after 3 warm-up: GPU {gpu:?}, wall {wall:?}; CPU math median of 11 {:?}",
                    T::data_type(),
                    cpu[5]
                );
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<f16>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
