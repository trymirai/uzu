use std::{
    fmt::Debug,
    mem::size_of,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use bytemuck::NoUninit;
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{NormalizationCase, check_bounds, kernel_fixture::KernelFixture, round32, staged_rms_bounds};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE, kernel::NormalizationKernel},
        cpu::Cpu,
        vulkan::{Error, NormalizationVulkanKernel, VkBuffer},
    },
    data_type::DataType,
    encodable_block::normalization::{PostLayerScalar, ShortcutMode},
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// Runs `$check::<InputT, AffineT, OutputT>()` for all 27 storage type combinations.
macro_rules! for_each_combination {
    ($check:ident) => { for_each_combination!(@input $check [f32 f16 bf16]); };
    (@input $check:ident [$($input:tt)*]) => { $(for_each_combination!(@affine $check $input [f32 f16 bf16]);)* };
    (@affine $check:ident $input:tt [$($affine:tt)*]) => {
        $(for_each_combination!(@output $check $input $affine [f32 f16 bf16]);)*
    };
    (@output $check:ident $input:tt $affine:tt [$($output:tt)*]) => { $($check::<$input, $affine, $output>();)* };
}

/// The CPU kernel through the shared trait, dispatched `repeats` times, one submission each. Returns the output, the
/// shortcut (empty when unbound) and the values the last dispatch normalized (see `stored_input`).
#[allow(clippy::type_complexity)]
fn cpu_output<I: ArrayElement + Float, A: ArrayElement + Float, O: ArrayElement + Float>(
    case: &NormalizationCase<I, A>,
    repeats: usize,
) -> (Vec<O>, Vec<I>, Vec<I>) {
    let context = create_context::<Cpu>();
    let s = case.specializations();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::NormalizationKernel::new(
        &context,
        I::data_type(),
        A::data_type(),
        O::data_type(),
        DataType::F32,
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
        s[9],
    )
    .expect("CPU Normalization");
    let input = (!case.in_place).then(|| create_buffer_with_data::<Cpu, I>(&context, &case.input));
    let scales = case.scales.as_ref().map(|scales| create_buffer_with_data::<Cpu, A>(&context, scales));
    let biases = case.biases.as_ref().map(|biases| create_buffer_with_data::<Cpu, A>(&context, biases));
    let factors = case.hadamard_factors.as_ref().map(|factors| create_buffer_with_data::<Cpu, i32>(&context, factors));
    let mut output = create_buffer_with_data::<Cpu, O>(&context, &case.initial_output(O::zero()));
    let mut shortcut =
        (case.shortcut_mode != ShortcutMode::None).then(|| create_buffer_with_data::<Cpu, I>(&context, &case.shortcut));
    let mut last_input = case.input.clone();
    for _ in 0..repeats {
        if case.in_place {
            last_input = buffer_to_vec::<Cpu, O>(&output).into_iter().map(|value| I::from(value).unwrap()).collect();
        }
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        kernel.encode(
            input.as_ref(),
            scales.as_ref(),
            biases.as_ref(),
            &mut output,
            shortcut.as_mut(),
            factors.as_ref(),
            case.batch_size,
            case.element_count,
            case.epsilon,
            case.scale_offset,
            case.post_layer_scalar_value(),
            &mut command_buffer,
        );
        submit_command_buffer(command_buffer);
    }
    let shortcut = shortcut.map(|shortcut| buffer_to_vec::<Cpu, I>(&shortcut)).unwrap_or_default();
    let stored = stored_input(case.shortcut_mode, &last_input, &shortcut);
    (buffer_to_vec::<Cpu, O>(&output), shortcut, stored)
}

/// The values a dispatch normalizes: the shortcut it stored after a residual add, otherwise its input.
fn stored_input<I: Clone>(
    mode: ShortcutMode,
    input: &[I],
    shortcut: &[I],
) -> Vec<I> {
    match mode {
        ShortcutMode::Add => shortcut.to_vec(),
        _ => input.to_vec(),
    }
}

fn kernel<I: ArrayElement + Float, A: ArrayElement + Float, O: ArrayElement>(
    fixture: &KernelFixture,
    case: &NormalizationCase<I, A>,
) -> NormalizationVulkanKernel {
    let s = case.specializations();
    NormalizationVulkanKernel::new(
        &fixture.context,
        I::data_type(),
        A::data_type(),
        O::data_type(),
        DataType::F32,
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
        s[9],
    )
    .expect("Vulkan Normalization")
}

/// The Vulkan kernel over guarded ranges: one submission per entry of `submissions`, each dispatching that many times,
/// all on the same buffers. Asserts every read-only payload and every guard is unchanged, and returns the output and
/// the shortcut (empty when unbound).
fn gpu_output<I: ArrayElement + Float, A: ArrayElement + Float, O: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &NormalizationVulkanKernel,
    case: &NormalizationCase<I, A>,
    submissions: &[usize],
) -> (Vec<O>, Vec<I>) {
    let (input_sentinel, affine_sentinel, output_sentinel) =
        (I::from(-7.0).unwrap(), A::from(-7.0).unwrap(), O::from(-7.0).unwrap());
    let input = (!case.in_place).then(|| fixture.guarded(&case.input, input_sentinel));
    let scales = case.scales.as_ref().map(|scales| fixture.guarded(scales, affine_sentinel));
    let biases = case.biases.as_ref().map(|biases| fixture.guarded(biases, affine_sentinel));
    let factors = case.hadamard_factors.as_ref().map(|factors| fixture.guarded(factors, -7));
    let output = fixture.guarded(&case.initial_output(output_sentinel), output_sentinel);
    let shortcut = (case.shortcut_mode != ShortcutMode::None).then(|| fixture.guarded(&case.shortcut, input_sentinel));
    fn range(guarded: &Option<(Arc<VkBuffer>, Range<u64>)>) -> Option<(&Arc<VkBuffer>, Range<u64>)> {
        guarded.as_ref().map(|(buffer, range)| (buffer, range.clone()))
    }
    for &repeats in submissions {
        let mut encoding = fixture.encoding();
        for _ in 0..repeats {
            // SAFETY: input, output and shortcut hold batch_size rows of element_count aligned elements; scales,
            // biases and factors hold element_count; the case satisfies the raw preconditions (residual_add implies
            // copy_to_shortcut, RHT rows are whole blocks, in-place cases share one type); output and shortcut alias
            // nothing.
            unsafe {
                kernel.encode(
                    range(&input),
                    range(&scales),
                    range(&biases),
                    (&output.0, output.1.clone()),
                    range(&shortcut),
                    range(&factors),
                    case.batch_size,
                    case.element_count,
                    case.epsilon,
                    case.scale_offset,
                    case.post_layer_scalar_value(),
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        if let Some(input) = &input {
            KernelFixture::assert_unchanged(input, input_sentinel, &case.input, "input");
        }
        for (guarded, payload, name) in [(&scales, &case.scales, "scales"), (&biases, &case.biases, "biases")] {
            if let (Some(guarded), Some(payload)) = (guarded, payload) {
                KernelFixture::assert_unchanged(guarded, affine_sentinel, payload, name);
            }
        }
        if let (Some(guarded), Some(payload)) = (&factors, &case.hadamard_factors) {
            KernelFixture::assert_unchanged(guarded, -7, payload, "hadamard factors");
        }
        let shortcut = shortcut.map(|shortcut| KernelFixture::read_guarded(&shortcut, input_sentinel));
        (KernelFixture::read_guarded(&output, output_sentinel), shortcut.unwrap_or_default())
    }
}

fn types<I: ArrayElement, A: ArrayElement, O: ArrayElement>() -> String {
    format!("{:?}/{:?}/{:?}", I::data_type(), A::data_type(), O::data_type())
}

/// Spacing of `T` at the magnitude of `value`.
fn ulp<T: Float>(value: f64) -> f64 {
    let magnitude = value.abs().max(T::min_positive_value().to_f64().unwrap());
    T::epsilon().to_f64().unwrap() * magnitude.log2().floor().exp2()
}

/// The stage-by-stage FP64 reference of one dispatch from `stored`, an implementation's own stored row values (its
/// shortcut after a residual add, otherwise the input), and each element's error allowance. Statistics and
/// normalization are FP64; values are rounded to the output type exactly where the contract converts them.
///
/// The allowance starts at the first stored stage p with the FP32 reference budget, 1e-5 of the normalization's
/// magnitude before cancellation, (|x - pivot| + |mean - pivot|) rms_inv |scale|, which covers the FP32 statistics and
/// normalization arithmetic, plus the rounding of p: 2 storage steps for 16-bit outputs (one for that budget crossing a
/// rounding boundary, one for the rounding itself), 1 for F32. Every later stored stage z adds q(z), one storage step of
/// z for its single rounding. The Hadamard transform outputs ±1/√H weighted sums, so it sums the incoming allowances
/// over √H and adds γ(log2 H + 2) Σ|v| / √H, with γ(n) = n u / (1 - n u) and u = 2^-24, for the FP32 rounding of the
/// log2 H butterfly additions, of the √H constant and of the division by it, then q of its output. Output scaling
/// multiplies the allowance by |scalar| and adds q.
pub fn stage_oracle<I: Float, A: Float, O: Float>(
    case: &NormalizationCase<I, A>,
    stored: &[I],
) -> (Vec<O>, Vec<f64>) {
    let low = size_of::<O>() == 2;
    let round = |value: f64| O::from(value).unwrap().to_f64().unwrap();
    let q = ulp::<O>;
    let n = case.element_count as usize;
    let (mut expected, mut allowance) = (Vec::new(), Vec::new());
    for row in stored.chunks(n) {
        let x = row.iter().map(|value| value.to_f64().unwrap()).collect::<Vec<_>>();
        let (pivot, mean) = match case.subtract_mean {
            true => (x[0], x.iter().sum::<f64>() / n as f64),
            false => (0.0, 0.0),
        };
        let variance = x.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n as f64;
        let rms_inv = 1.0 / (variance + f64::from(case.epsilon)).sqrt();
        let mut stage = (0..n)
            .map(|i| {
                let scale =
                    case.scales.as_ref().map(|scales| scales[i].to_f64().unwrap() + f64::from(case.scale_offset));
                let normalized = (x[i] - mean) * rms_inv;
                let p = match scale {
                    Some(scale) if case.full_layer => round(normalized * scale),
                    Some(scale) => round(round(normalized) * round(scale)),
                    None => round(normalized),
                };
                let magnitude = ((x[i] - pivot).abs() + (mean - pivot).abs()) * rms_inv * scale.map_or(1.0, f64::abs);
                let first = 1e-5 * magnitude
                    + if low {
                        2.0
                    } else {
                        1.0
                    } * q(p);
                match &case.biases {
                    Some(biases) => {
                        let v = round(p + biases[i].to_f64().unwrap());
                        (v, first + q(v))
                    },
                    None => (p, first),
                }
            })
            .collect::<Vec<_>>();
        if let Some(factors) = &case.hadamard_factors {
            let h = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            let root = (h as f64).sqrt();
            let operations = (h as f64).log2() + 2.0;
            let gamma = operations * 2f64.powi(-24) / (1.0 - operations * 2f64.powi(-24));
            stage = (0..n)
                .map(|j| {
                    let block = j / h * h..j / h * h + h;
                    let sign = |k: usize| {
                        if ((j % h) & (k % h)).count_ones().is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        }
                    };
                    let sum = block.clone().map(|k| sign(k) * stage[k].0 * f64::from(factors[k])).sum::<f64>();
                    let incoming = block.clone().map(|k| stage[k].1).sum::<f64>() / root;
                    let butterfly = gamma * block.map(|k| stage[k].0.abs()).sum::<f64>() / root;
                    let r = round(sum / root);
                    (r, incoming + butterfly + q(r))
                })
                .collect();
        }
        for (value, error) in stage {
            let (value, error) = match case.post_layer_scalar {
                PostLayerScalar::ScaleOutput(scalar) => {
                    let o = round(value * f64::from(scalar));
                    (o, f64::from(scalar).abs() * error + q(o))
                },
                _ => (value, error),
            };
            expected.push(O::from(value).unwrap());
            allowance.push(error);
        }
    }
    (expected, allowance)
}

/// Max absolute error, worst error/allowance and allowance violations of `actual` against the stage oracle. NaN must
/// occur exactly where the oracle has it and infinities must match it exactly; finite values within the allowance.
fn against_oracle<O: Float + Debug>(
    (expected, allowance): &(Vec<O>, Vec<f64>),
    actual: &[O],
    case: &str,
) -> [f64; 4] {
    assert!(expected.len() == allowance.len() && expected.len() == actual.len(), "{case}: lengths");
    let mut worst = [0.0f64; 4];
    for (index, ((&expected, &bound), &actual)) in expected.iter().zip(allowance).zip(actual).enumerate() {
        assert_eq!(
            expected.is_nan(),
            actual.is_nan(),
            "{case}: element {index}: FP64 stages {expected:?}, actual {actual:?}"
        );
        if expected.is_nan() || (expected.is_infinite() && actual == expected) {
            continue;
        }
        let error = (actual.to_f64().unwrap() - expected.to_f64().unwrap()).abs();
        let ratio = if error == 0.0 {
            0.0
        } else {
            error / bound
        };
        // A NaN ratio, from an infinite or NaN error, exceeds the allowance too.
        let exceeded = ratio.is_nan() || ratio > 1.0;
        if exceeded && worst[3] == 0.0 {
            eprintln!(
                "{case}: element {index} exceeds its allowance {bound:.3e}: FP64 stages {expected:?}, actual {actual:?}"
            );
        }
        worst = [worst[0].max(error), worst[1].max(ratio), 0.0, worst[3] + f64::from(u8::from(exceeded))];
    }
    worst
}

/// Bounds the CPU-Vulkan difference by |c_cpu - c_gpu| + e_cpu + e_gpu from each output's own FP64 center and
/// allowance (the triangle inequality); the centers differ only where the values each normalized did. Returns the max
/// pair error, worst error/bound, max center divergence and violations. Non-finite centers are left to `against_oracle`.
fn against_centers<O: Float + Debug>(
    (cpu_center, cpu_allowance): &(Vec<O>, Vec<f64>),
    (gpu_center, gpu_allowance): &(Vec<O>, Vec<f64>),
    cpu: &[O],
    gpu: &[O],
    case: &str,
) -> [f64; 4] {
    assert!(cpu_center.len() == gpu_center.len() && cpu.len() == gpu.len() && cpu.len() == cpu_center.len(), "{case}");
    let mut worst = [0.0f64; 4];
    for index in 0..cpu.len() {
        let (center_cpu, center_gpu) = (cpu_center[index].to_f64().unwrap(), gpu_center[index].to_f64().unwrap());
        if !center_cpu.is_finite() || !center_gpu.is_finite() {
            continue;
        }
        let divergence = (center_cpu - center_gpu).abs();
        let bound = divergence + cpu_allowance[index] + gpu_allowance[index];
        let error = (cpu[index].to_f64().unwrap() - gpu[index].to_f64().unwrap()).abs();
        let ratio = if error == 0.0 {
            0.0
        } else {
            error / bound
        };
        let exceeded = ratio.is_nan() || ratio > 1.0;
        if exceeded && worst[3] == 0.0 {
            eprintln!("{case}: element {index} exceeds {bound:.3e}: CPU {:?}, Vulkan {:?}", cpu[index], gpu[index]);
        }
        worst = [
            worst[0].max(error),
            worst[1].max(ratio),
            worst[2].max(divergence),
            worst[3] + f64::from(u8::from(exceeded)),
        ];
    }
    worst
}

/// Compares Vulkan with the CPU after the last dispatch from `cpu_stored` and `gpu_stored`, the values each normalized
/// last: each output against its own stage oracle, and the pair within the original bound on plain paths or, where a
/// subtracted mean, a bias or the Hadamard transform cancels, and after chains whose inputs diverged, within the bound
/// of `against_centers` (the relative column is error/bound, the steps column the center divergence), printing how many
/// elements the original bound would reject.
fn compare_dispatch<I: Float, A: Float, O: ArrayElement + Float + Debug>(
    case: &NormalizationCase<I, A>,
    (cpu, cpu_stored): (&[O], &[I]),
    (gpu, gpu_stored): (&[O], &[I]),
    chained: bool,
    label: &str,
    detail: &str,
) -> Vec<(String, [f64; 4])> {
    let (cpu_oracle, gpu_oracle) =
        (stage_oracle::<I, A, O>(case, cpu_stored), stage_oracle::<I, A, O>(case, gpu_stored));
    let mut errors = vec![
        (
            format!("{label} CPU vs FP64 stages"),
            against_oracle(&cpu_oracle, cpu, &format!("{detail} CPU vs FP64 stages")),
        ),
        (
            format!("{label} Vulkan vs FP64 stages"),
            against_oracle(&gpu_oracle, gpu, &format!("{detail} Vulkan vs FP64 stages")),
        ),
    ];
    let original = KernelFixture::compare(cpu, gpu, detail, 1e-5, 1e-6);
    match chained || case.subtract_mean || case.biases.is_some() || case.hadamard_factors.is_some() {
        true => {
            if original[3] > 0.0 {
                eprintln!(
                    "{detail}: {} elements beyond the original pair bound; the derived bound applies",
                    original[3]
                );
            }
            let derived = against_centers(&cpu_oracle, &gpu_oracle, cpu, gpu, &format!("{detail} derived pair"));
            errors.push((format!("{label} pair within FP64 center bound"), derived));
        },
        false => errors.push((label.to_owned(), original)),
    }
    errors
}

/// Runs one dispatch of a case on Vulkan and the CPU kernel and returns the errors with the Vulkan output and shortcut.
/// The shortcuts must match bit for bit (NaN matching NaN), so both normalize the same stored values; the outputs are
/// compared by `compare_dispatch`. Errors are labeled by types and path.
#[allow(clippy::type_complexity)]
fn check<I: ArrayElement + Float + Debug, A: ArrayElement + Float, O: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &NormalizationVulkanKernel,
    case: &NormalizationCase<I, A>,
) -> (Vec<(String, [f64; 4])>, Vec<O>, Vec<I>) {
    let label = format!("{} {}", types::<I, A, O>(), case.path());
    let detail = format!("{label} batch {} elements {}", case.batch_size, case.element_count);
    let (cpu, cpu_shortcut, cpu_stored) = cpu_output::<I, A, O>(case, 1);
    let (gpu, gpu_shortcut) = gpu_output::<I, A, O>(fixture, kernel, case, &[1]);
    KernelFixture::assert_bits(&cpu_shortcut, &gpu_shortcut, &format!("{detail} shortcut"));
    let gpu_stored = stored_input(case.shortcut_mode, &case.input, &gpu_shortcut);
    let errors = compare_dispatch(case, (&cpu, &cpu_stored), (&gpu, &gpu_stored), false, &label, &detail);
    (errors, gpu, gpu_shortcut)
}

/// Steps a chain one dispatch at a time from the previous Vulkan output and shortcut, so the shortcut every dispatch
/// stores is compared bit for bit with the CPU's from identical inputs; each step continues from the checked dispatch.
/// Returns the errors, the final Vulkan output and shortcut, and the values the last step normalized.
#[allow(clippy::type_complexity)]
fn check_steps<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &NormalizationVulkanKernel,
    mut case: NormalizationCase<T, T>,
    steps: usize,
) -> (Vec<(String, [f64; 4])>, Vec<T>, Vec<T>, Vec<T>) {
    let (mut errors, mut output, mut shortcut, mut stored) =
        (Vec::new(), Vec::new(), case.shortcut.clone(), Vec::new());
    for _ in 0..steps {
        let step_errors;
        (step_errors, output, shortcut) = check::<T, T, T>(fixture, kernel, &case);
        errors.extend(step_errors.into_iter().map(|(label, error)| (format!("{label} stepped"), error)));
        stored = stored_input(case.shortcut_mode, &case.input, &shortcut);
        if case.in_place {
            case.input = output.clone();
        }
        if case.shortcut_mode == ShortcutMode::Add {
            case.shortcut = shortcut.clone();
        }
    }
    (errors, output, shortcut, stored)
}

/// Runs a whole chain of `submissions` on Vulkan and the CPU kernel. The Vulkan chain must reproduce the stepped
/// trajectory bit for bit; the final outputs are compared by `compare_dispatch` from the values each last normalized,
/// and the chained shortcuts are reported under their own label.
fn check_chain<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &NormalizationVulkanKernel,
    case: &NormalizationCase<T, T>,
    submissions: &[usize],
    (stepped_output, stepped_shortcut, gpu_stored): (&[T], &[T], &[T]),
) -> Vec<(String, [f64; 4])> {
    let label = format!("{} {}", types::<T, T, T>(), case.path());
    let detail =
        format!("{label} batch {} elements {} submissions {submissions:?}", case.batch_size, case.element_count);
    let (gpu, gpu_shortcut) = gpu_output::<T, T, T>(fixture, kernel, case, submissions);
    KernelFixture::assert_bits(stepped_output, &gpu, &format!("{detail} output vs stepped"));
    KernelFixture::assert_bits(stepped_shortcut, &gpu_shortcut, &format!("{detail} shortcut vs stepped"));
    let (cpu, cpu_shortcut, cpu_stored) = cpu_output::<T, T, T>(case, submissions.iter().sum());
    let mut errors =
        compare_dispatch(case, (&cpu, &cpu_stored), (&gpu, gpu_stored), true, &format!("{label} chained"), &detail);
    let shortcut_detail = format!("{detail} shortcut");
    errors.push((
        format!("{label} chained shortcut"),
        KernelFixture::compare(&cpu_shortcut, &gpu_shortcut, &shortcut_detail, 1e-5, 1e-6),
    ));
    errors
}

/// The FP64 layer normalization of `row_length`-element rows of stored values, rounded to the storage type.
fn layer_norm_fp64<T: Float>(
    values: &[T],
    row_length: usize,
    epsilon: f32,
) -> Vec<T> {
    values
        .chunks(row_length)
        .flat_map(|row| {
            let row = row.iter().map(|value| value.to_f64().unwrap()).collect::<Vec<_>>();
            let mean = row.iter().sum::<f64>() / row_length as f64;
            let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / row_length as f64;
            row.into_iter().map(move |value| T::from((value - mean) / (variance + f64::from(epsilon)).sqrt()).unwrap())
        })
        .collect()
}

/// Checks each path over one and three rows of every length: RHT paths over whole-block lengths, including ones
/// that are not multiples of the workgroup, other paths over 1, 31, 33, 257, 1000, 1024 and 4096.
fn check_paths<I: ArrayElement + Float + Debug, A: ArrayElement + Float, O: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    paths: &[fn(NormalizationCase<I, A>) -> NormalizationCase<I, A>],
) -> Vec<(String, [f64; 4])> {
    let mut errors = Vec::new();
    for path in paths {
        let kernel = kernel::<I, A, O>(fixture, &path(NormalizationCase::new(1, 32, 0)));
        let rht = path(NormalizationCase::new(1, 32, 0)).hadamard_factors.is_some();
        let lengths: &[u32] = if rht {
            &[32, 96, 288, 1024, 4096]
        } else {
            &[1, 31, 33, 257, 1000, 1024, 4096]
        };
        for &element_count in lengths {
            for batch_size in [1, 3] {
                let case = path(NormalizationCase::new(batch_size, element_count, element_count as usize));
                errors.extend(check::<I, A, O>(fixture, &kernel, &case).0);
            }
        }
    }
    errors
}

/// Every representative valid path: each specialization branch and their interactions. `0.3` is not representable
/// in F16 or BF16, which separates the CPU's rounding of the output scalar from the Metal order the Vulkan kernel
/// follows; `0.5` is representable everywhere.
fn all_paths<I: Float, A: Float>() -> Vec<fn(NormalizationCase<I, A>) -> NormalizationCase<I, A>> {
    vec![
        |case| case,
        |case| case.subtract_mean(),
        |case| case.scales(true, 1.0),
        |case| case.scales(false, 0.0),
        |case| case.biases(),
        |case| case.subtract_mean().scales(true, 0.0).biases(),
        |case| case.scales(false, 1.0).biases(),
        |case| case.shortcut(ShortcutMode::Copy),
        |case| case.shortcut(ShortcutMode::Add),
        |case| case.shortcut(ShortcutMode::Add).post(PostLayerScalar::ScaleResidualSum(0.5)),
        |case| case.shortcut(ShortcutMode::Add).post(PostLayerScalar::ScaleResidualSum(0.3)).subtract_mean(),
        |case| case.shortcut(ShortcutMode::Copy).post(PostLayerScalar::ScaleOutput(0.5)),
        |case| case.scales(true, 0.0).post(PostLayerScalar::ScaleOutput(0.5)),
        |case| case.scales(true, 0.0).post(PostLayerScalar::ScaleOutput(0.3)),
        |case| case.hadamard(),
        |case| case.scales(true, 0.0).post(PostLayerScalar::ScaleOutput(0.3)).hadamard(),
        |case| {
            let case = case.subtract_mean().scales(false, 0.0).biases().shortcut(ShortcutMode::Add);
            case.post(PostLayerScalar::ScaleOutput(0.5)).hadamard()
        },
        |case| case.shortcut(ShortcutMode::Copy).post(PostLayerScalar::ScaleResidualSum(0.5)).hadamard(),
    ]
}

/// Every type combination out of place over three paths that exercise each conversion: the residual sum and its
/// scaling, both affine roundings, the bias sum, the RHT input and the output scaling.
fn matches_cpu<I: ArrayElement + Float + Debug, A: ArrayElement + Float, O: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let paths: [fn(NormalizationCase<I, A>) -> NormalizationCase<I, A>; 3] = [
        |case| {
            let case = case.subtract_mean().scales(true, 1.0).biases().shortcut(ShortcutMode::Add);
            case.post(PostLayerScalar::ScaleResidualSum(0.5))
        },
        |case| case.scales(false, 0.0).shortcut(ShortcutMode::Copy).post(PostLayerScalar::ScaleOutput(0.5)),
        |case| case.scales(false, 0.0).biases().post(PostLayerScalar::ScaleOutput(0.5)).hadamard(),
    ];
    KernelFixture::report("Normalization", check_paths::<I, A, O>(&fixture, &paths));
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_all_combinations() {
    for_each_combination!(matches_cpu);
}

/// Every path with the model layer's types: equal input and output types, FP32 affine.
#[uzu_test]
fn matches_cpu_all_paths() {
    let fixture = KernelFixture::new();
    let mut errors = check_paths::<f32, f32, f32>(&fixture, &all_paths());
    errors.extend(check_paths::<f16, f32, f16>(&fixture, &all_paths()));
    errors.extend(check_paths::<bf16, f32, bf16>(&fixture, &all_paths()));
    KernelFixture::report("Normalization", errors);
    fixture.assert_clean();
}

/// In place with one type throughout, and chained dispatches within and across submissions: in-place
/// renormalization and repeated residual accumulation into the shortcut.
fn in_place<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let paths: [fn(NormalizationCase<T, T>) -> NormalizationCase<T, T>; 3] = [
        |case| case.in_place(),
        |case| {
            case.in_place().scales(true, 0.0).shortcut(ShortcutMode::Add).post(PostLayerScalar::ScaleResidualSum(0.5))
        },
        |case| case.in_place().subtract_mean().scales(false, 0.0).hadamard(),
    ];
    let mut errors = check_paths::<T, T, T>(&fixture, &paths);
    let accumulate: fn(NormalizationCase<T, T>) -> NormalizationCase<T, T> = |case| case.shortcut(ShortcutMode::Add);
    for path in paths.into_iter().chain([accumulate]) {
        let case = path(NormalizationCase::new(3, 1024, 5));
        let kernel = kernel::<T, T, T>(&fixture, &case);
        let (step_errors, output, shortcut, stored) = check_steps(&fixture, &kernel, case.clone(), 3);
        errors.extend(step_errors);
        for submissions in [&[3][..], &[2, 1]] {
            errors.extend(check_chain(&fixture, &kernel, &case, submissions, (&output, &shortcut, &stored)));
        }
    }
    KernelFixture::report("Normalization", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn in_place_all_types() {
    in_place::<f32>();
    in_place::<f16>();
    in_place::<bf16>();
}

/// Zero rows, rows of one exactly representable value (zero variance), rows holding NaN or +Inf, a NaN shortcut, and
/// near-constant rows around ±1000 plus a uniform row, also checked against the FP64 layer normalization: F32 within
/// relative 2e-6 or absolute 1e-6, 16-bit types within 2 storage steps.
fn special_rows<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let mut errors = Vec::new();
    let paths: [fn(NormalizationCase<T, f32>) -> NormalizationCase<T, f32>; 3] = [
        |case| case.scales(true, 0.0).biases(),
        |case| case.subtract_mean().scales(true, 0.0).biases(),
        |case| case.subtract_mean().shortcut(ShortcutMode::Add).hadamard(),
    ];
    for path in paths {
        let kernel = kernel::<T, f32, T>(&fixture, &path(NormalizationCase::new(1, 32, 0)));
        let mut rows = vec![T::zero(); 256];
        rows.extend([T::from(0.5).unwrap(); 256]);
        rows.extend(NormalizationCase::<T, f32>::new(2, 256, 3).input);
        (rows[600], rows[900]) = (T::nan(), T::infinity());
        let mut case = path(NormalizationCase::new(4, 256, 0));
        case.input = rows;
        case.shortcut[700] = T::nan();
        errors.extend(check::<T, f32, T>(&fixture, &kernel, &case).0);
    }
    // One storage step at 1000, so every near-constant row keeps a variance in its type.
    let step = T::epsilon().to_f32().unwrap() * 512.0;
    let kernel = kernel::<T, f32, T>(&fixture, &NormalizationCase::new(1, 1, 0).subtract_mean());
    let near_constant = [33, 257, 4096].map(|element_count| {
        let mut case = NormalizationCase::<T, f32>::new(3, element_count, 0).subtract_mean();
        case.input = [(1000.0f32, 1.0), (-1000.0, 1.0), (1000.0, 0.0)]
            .into_iter()
            .flat_map(|(center, spread)| {
                (0..element_count as usize)
                    .map(move |i| T::from(center + ((i * 37) % 9) as f32 * step * spread).unwrap())
            })
            .collect();
        ("near-constant", case)
    });
    // The originally failing case: E[x²] - mean² cancelled to a NaN or a wrong variance.
    let mut regression = NormalizationCase::<T, f32>::new(2, 4096, 0).subtract_mean();
    regression.input = (0..8192).map(|i| T::from(1000.0 + ((i * 37) % 199) as f32 / 1024.0).unwrap()).collect();
    for (name, case) in near_constant.into_iter().chain([("cancellation regression", regression)]) {
        let label = format!("{} {name}", types::<T, f32, T>());
        let (case_errors, gpu, _) = check::<T, f32, T>(&fixture, &kernel, &case);
        errors.extend(case_errors.into_iter().map(|(oracle, error)| (format!("{label} {oracle}"), error)));
        let expected = layer_norm_fp64(&case.input, case.element_count as usize, case.epsilon);
        let detail = format!("{label} elements {} vs FP64", case.element_count);
        errors.push((format!("{label} vs FP64"), KernelFixture::compare(&expected, &gpu, &detail, 2e-6, 1e-6)));
    }
    KernelFixture::report("Normalization", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn special_rows_all_types() {
    special_rows::<f32>();
    special_rows::<f16>();
    special_rows::<bf16>();
}

#[uzu_test]
fn zero_rows_record_nothing() {
    let fixture = KernelFixture::new();
    for (batch_size, element_count) in [(0, 32), (3, 0), (0, 0)] {
        let case = NormalizationCase::<f32, f32>::new(batch_size, element_count, 0)
            .scales(true, 0.0)
            .biases()
            .shortcut(ShortcutMode::Add);
        let case = if element_count == 0 {
            case
        } else {
            case.hadamard()
        };
        let (output, shortcut) =
            gpu_output::<f32, f32, f32>(&fixture, &kernel::<f32, f32, f32>(&fixture, &case), &case, &[1]);
        assert!(output.is_empty() && shortcut.is_empty());
    }
    fixture.assert_clean();
}

#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    let new = |types: [DataType; 4]| {
        NormalizationVulkanKernel::new(
            &fixture.context,
            types[0],
            types[1],
            types[2],
            types[3],
            false,
            false,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
    };
    for types in [
        [DataType::F32, DataType::F32, DataType::F32, DataType::F16],
        [DataType::F32, DataType::F32, DataType::F32, DataType::BF16],
        [DataType::I32, DataType::F32, DataType::F32, DataType::F32],
        [DataType::F32, DataType::F32, DataType::I32, DataType::F32],
    ] {
        assert!(
            matches!(
                new(types),
                Err(Error::KernelVariant {
                    kernel: "Normalization",
                    ..
                })
            ),
            "{types:?} accepted"
        );
    }
    // Each case binds exactly one optional argument against its specialization.
    let values = fixture.buffer(&[1.0f32; 64]);
    let base = || NormalizationCase::<f32, f32>::new(1, 32, 0);
    let cases = [
        base().in_place(),
        base().scales(true, 0.0),
        base().biases(),
        base().shortcut(ShortcutMode::Copy),
        base().hadamard(),
    ];
    let mut encoding = fixture.encoding();
    for (wrong, case) in cases.iter().enumerate() {
        let kernel = kernel::<f32, f32, f32>(&fixture, case);
        let mut optionals = [
            !case.in_place,
            case.scales.is_some(),
            case.biases.is_some(),
            case.specializations()[3],
            case.hadamard_factors.is_some(),
        ];
        optionals[wrong] = !optionals[wrong];
        let bind = |present: bool| present.then_some((&values, 0..128));
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the optional-argument assertion fails before recording.
            kernel.encode(
                bind(optionals[0]),
                bind(optionals[1]),
                bind(optionals[2]),
                (&values, 128..256),
                bind(optionals[3]),
                bind(optionals[4]),
                1,
                32,
                1e-5,
                0.0,
                1.0,
                &mut encoding,
            );
        });
        assert!(catch_unwind(encode).is_err(), "{} accepted a wrong optional argument {wrong}", case.path());
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded no dispatch.
    assert_eq!(unsafe { KernelFixture::read::<f32>(&values) }, [1.0; 64]);
    fixture.assert_clean();
}

/// Statistics and differences the CPU keeps subnormal, which flushing turns into infinities, NaN or wrong normal outputs.
/// RMS: rows whose squares, sum or mean are subnormal, a normal row, a zero row and a row mixing normal and flushed-size
/// squares, under zero, subnormal, negative-zero and NaN epsilons, and without scales a normal row with subnormal
/// elements, normalized through the two differences from zero. LayerNorm without scales: a row around the smallest
/// normal whose deviations are one subnormal unit of T (once a NaN for an infinity, zeros for normal or subnormal
/// outputs) and an ordinary near-constant row, under zero, subnormal, ordinary and negative-zero epsilons. Long rows of 1000 elements, an RMS row and a near-constant LayerNorm row, sum
/// over more elements than invocations, in the device's order and the CPU's. The stage oracle's FP64 statistics within
/// their 1e-5 budget cannot represent a staged underflow, so CPU and Vulkan are each checked against the staged
/// endpoints of the workgroup of 256, the guards and the input unchanged.
fn vanishing_rows_match_staging<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let two = |exponent: i32| 2f32.powi(exponent);
    let (smallest, unit) = (f32::MIN_POSITIVE, (T::min_positive_value() * T::epsilon()).to_f32().unwrap());
    let rms: &[[f32; 4]] = &[
        [two(-70), -1.5 * two(-70), 0.75 * two(-69), two(-71)],
        [two(-62), -two(-62), two(-62), -1.25 * two(-62)],
        [0.0, -0.0, 0.0, 0.0],
        [1.5 * two(-75), -two(-74), 0.0, two(-76)],
        [1.0, two(-70), -1.5 * two(-75), 3.0 * two(-76)],
    ];
    let layer_norm: &[[f32; 4]] =
        &[[smallest, smallest + unit, smallest - unit, smallest], [1.0 + two(-6), 1.0, 1.0 - two(-7), 1.0]];
    let tiny = f32::from_bits(0x0001_0000);
    let mut cases = Vec::new();
    for (subtract_mean, rows, epsilons) in
        [(false, rms, [0.0, tiny, -0.0, f32::NAN]), (true, layer_norm, [0.0, tiny, 1e-5, -0.0])]
    {
        for epsilon in epsilons {
            let case = NormalizationCase::<T, f32>::new(rows.len() as u32, 4, 0);
            let mut case = if subtract_mean {
                case.subtract_mean()
            } else {
                case.scales(true, 0.0)
            };
            case.epsilon = epsilon;
            case.input = rows.iter().flatten().map(|&x| T::from(x).unwrap()).collect();
            cases.push(case);
        }
    }
    let mut near_constant = NormalizationCase::<T, f32>::new(2, 1000, 0).subtract_mean().scales(false, 1.0);
    let step = T::epsilon().to_f32().unwrap() * 512.0;
    near_constant.input = (0..2000).map(|i| T::from(1000.0 + ((i * 37) % 9) as f32 * step).unwrap()).collect();
    let mut subnormal_inputs = NormalizationCase::<T, f32>::new(1, 4, 0);
    subnormal_inputs.input = [1.5, 3.0 * unit, -unit, 0.5].map(|x| T::from(x).unwrap()).to_vec();
    cases.extend([subnormal_inputs, NormalizationCase::<T, f32>::new(2, 1000, 5).scales(true, 0.0), near_constant]);
    for case in &cases {
        let kernel = kernel::<T, f32, T>(&fixture, case);
        let (cpu, _, _) = cpu_output::<T, f32, T>(case, 1);
        let (gpu, _) = gpu_output::<T, f32, T>(&fixture, &kernel, case, &[1]);
        let label = format!("Normalization {} {} epsilon {:e}", types::<T, f32, T>(), case.path(), case.epsilon);
        check_bounds(&staged_rms_bounds::<T, f32, T>(case, 256), &cpu, &gpu, &label);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn vanishing_rows_match_staging_all_types() {
    vanishing_rows_match_staging::<f32>();
    vanishing_rows_match_staging::<bf16>();
}

/// `staged_rms_bounds` of the case without its biases and output scalar, then those stages: the bias added in FP32 to
/// the stored value and rounded to T, the output scalar's FP32 product rounded to T. Both are correctly rounded and
/// monotonic in the value, so the endpoints propagate, swapped by a negative scalar.
fn affine_tail_bounds<T: Float>(case: &NormalizationCase<T, f32>) -> Vec<((f64, f64), f64)> {
    let plain = NormalizationCase {
        biases: None,
        post_layer_scalar: PostLayerScalar::None,
        ..case.clone()
    };
    let to_t = |value: f64| T::from(value).unwrap().to_f64().unwrap();
    let stage = |index: usize, value: f64| {
        let biases = case.biases.as_ref();
        let value = biases.map_or(value, |biases| to_t(round32(value + f64::from(biases[index % biases.len()]))));
        match case.post_layer_scalar {
            PostLayerScalar::ScaleOutput(scalar) => to_t(round32(value * f64::from(scalar))),
            _ => value,
        }
    };
    let bounds = staged_rms_bounds::<T, f32, T>(&plain, 256).into_iter().enumerate();
    bounds
        .map(|(index, ((lo, hi), center))| {
            let (a, b) = (stage(index, lo), stage(index, hi));
            ((a.min(b), a.max(b)), stage(index, center))
        })
        .collect()
}

/// The elementwise stages on tiny values, which a device flushing subnormal operands or results zeroes: a normal row's
/// elements of 3 and -1 subnormal units of T normalize to subnormal outputs under full_layer and only-normalization
/// scales; subnormal biases on them, one cancelling to +0; an output scalar halving them, to -0 for the negative one,
/// and one scaling them to normal outputs; residual sums and their scaled halves that are subnormal, one by
/// cancellation. Shortcuts must match the CPU bit for bit, and both outputs the staged bounds of the stored values.
fn tiny_affine_stages_match_staging<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let (unit, two) = ((T::min_positive_value() * T::epsilon()).to_f32().unwrap(), |e: i32| 2f32.powi(e));
    let row = |values: [f32; 4]| values.map(|x| T::from(x).unwrap()).to_vec();
    let base = || {
        let mut case = NormalizationCase::<T, f32>::new(1, 4, 0);
        case.input = row([1.5, 3.0 * unit, -unit, 0.5]);
        case
    };
    let biased = |case: NormalizationCase<T, f32>| NormalizationCase {
        biases: Some(vec![0.0, unit, unit, -0.0]),
        ..case
    };
    let residual = |post| {
        let mut case = base().scales(true, 0.0).shortcut(ShortcutMode::Add).post(post);
        (case.input, case.shortcut) = (row([1.5, two(-127), -two(-126), 0.5]), row([0.0, two(-128), two(-127), -0.0]));
        case
    };
    let cases = [
        base().scales(true, 0.0),
        base().scales(false, 1.0),
        biased(base().scales(true, 0.0)),
        biased(base()),
        base().scales(true, 0.0).post(PostLayerScalar::ScaleOutput(0.5)),
        base().post(PostLayerScalar::ScaleOutput(two(30))),
        residual(PostLayerScalar::None),
        residual(PostLayerScalar::ScaleResidualSum(0.5)),
    ];
    for case in &cases {
        let label = format!("Normalization {} {}", types::<T, f32, T>(), case.path());
        let (cpu, cpu_shortcut, stored) = cpu_output::<T, f32, T>(case, 1);
        let (gpu, gpu_shortcut) = gpu_output::<T, f32, T>(&fixture, &kernel::<T, f32, T>(&fixture, case), case, &[1]);
        KernelFixture::assert_bits(&cpu_shortcut, &gpu_shortcut, &format!("{label} shortcut"));
        let tiny = |values: &[T]| values.iter().any(|value| value.is_subnormal());
        let scaled_up = matches!(case.post_layer_scalar, PostLayerScalar::ScaleOutput(scalar) if scalar > 1.0);
        assert!(tiny(&cpu) || tiny(&cpu_shortcut) || scaled_up && cpu[1].is_normal(), "{label}: no tiny witness");
        let stored = NormalizationCase {
            input: stored,
            shortcut_mode: ShortcutMode::None,
            ..case.clone()
        };
        check_bounds(&affine_tail_bounds(&stored), &cpu, &gpu, &label);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn tiny_affine_stages_match_staging_all_types() {
    tiny_affine_stages_match_staging::<f32>();
    tiny_affine_stages_match_staging::<bf16>();
}

/// Run alone: `cargo test ... normalization_test::throughput -- --ignored --nocapture`. Pipelines are created before
/// timing; each sample is one submission. Bytes count the input, shortcut read and write, affine and output.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(
        fixture: &KernelFixture,
        (batch_size, element_count): (u32, u32),
        path: fn(NormalizationCase<T, f32>) -> NormalizationCase<T, f32>,
    ) {
        let case = path(NormalizationCase::new(batch_size, element_count, 1));
        let kernel = kernel::<T, f32, T>(fixture, &case);
        let length = case.input.len();
        let rows = (length * size_of::<T>()) as u64;
        let affine = (element_count as usize * size_of::<f32>()) as u64;
        let input = fixture.buffer(&case.input);
        let output = fixture.buffer(&case.input);
        let shortcut = fixture.buffer(&case.shortcut);
        let scales = case.scales.as_ref().map(|scales| fixture.buffer(scales));
        let factors = case.hadamard_factors.as_ref().map(|factors| fixture.buffer(factors));
        let copy = case.shortcut_mode != ShortcutMode::None;
        let (gpu, wall) = fixture.median_times(|encoding| {
            // SAFETY: rows hold batch_size * element_count elements, scales and factors element_count; output and
            // shortcut alias nothing.
            unsafe {
                kernel.encode(
                    Some((&input, 0..rows)),
                    scales.as_ref().map(|scales| (scales, 0..affine)),
                    None,
                    (&output, 0..rows),
                    copy.then_some((&shortcut, 0..rows)),
                    factors.as_ref().map(|factors| (factors, 0..affine)),
                    case.batch_size,
                    case.element_count,
                    case.epsilon,
                    case.scale_offset,
                    case.post_layer_scalar_value(),
                    encoding,
                );
            }
        });
        let shortcut_bytes = match case.shortcut_mode {
            ShortcutMode::None => 0,
            ShortcutMode::Copy => rows,
            ShortcutMode::Add => 2 * rows,
        };
        let bytes = 2 * rows + shortcut_bytes + affine * (1 + u64::from(factors.is_some()));
        let rate = |time: std::time::Duration| bytes as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "Normalization {:?} {batch_size}x{element_count} {}: {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            T::data_type(),
            case.path(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    for shape in [(512, 4096), (8, 4096), (64, 8192)] {
        measure::<f32>(&fixture, shape, |case| case.scales(true, 0.0));
        measure::<bf16>(&fixture, shape, |case| case.scales(true, 0.0));
        measure::<bf16>(&fixture, shape, |case| case.scales(true, 0.0).shortcut(ShortcutMode::Add));
        measure::<bf16>(&fixture, shape, |case| case.scales(true, 0.0).hadamard());
        measure::<bf16>(&fixture, shape, |case| case.subtract_mean().scales(true, 0.0));
    }
    fixture.assert_clean();
}
