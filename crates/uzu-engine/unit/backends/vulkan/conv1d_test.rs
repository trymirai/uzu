use std::{fmt::Debug, mem::size_of, ops::Range, sync::Arc, time::Duration};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    activation_check, arg, assert_same_bits, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture, oracle,
    specials,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::ActivationType,
            kernel::{Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel},
        },
        cpu::Cpu,
        vulkan::{
            Conv1dDecodeVulkanKernel, Conv1dPackVulkanKernel, Conv1dScanVulkanKernel, VkBuffer, VkCommandBufferEncoding,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

const ACTIVATIONS: [ActivationType; 5] = [
    ActivationType::IDENTITY,
    ActivationType::SILU,
    ActivationType::GELUApprox,
    ActivationType::GELUExact,
    ActivationType::SOFTPLUS,
];

/// [channels, kernel_size, suffix_len, row_stride, state_stride, inner_dim, proj_dim] of Decode and Scan: kernel sizes 0
/// to 4; rows padded past the channels and states past kernel_size - 1; a channel past inner_dim + 2 proj_dim, ignored
/// as an output but still a state; x_out truncated to fewer channels than inner_dim; only ignored outputs; an empty suffix,
/// which Scan still turns into state; no channels; a Mamba2 decode shape; output segment sums past u32::MAX.
const SHAPES: [[u32; 7]; 11] = [
    [1, 0, 1, 1, 0, 1, 0],
    [5, 1, 2, 5, 0, 2, 1],
    [7, 2, 3, 9, 1, 3, 2],
    [33, 4, 3, 40, 5, 16, 8],
    [31, 4, 1, 31, 3, 40, 0],
    [12, 3, 3, 12, 2, 0, 0],
    [31, 4, 0, 31, 3, 16, 8],
    [0, 4, 3, 4, 3, 0, 0],
    [10240, 4, 1, 10240, 3, 8192, 1024],
    [7, 4, 1, 7, 3, 3, u32::MAX],
    [7, 4, 1, 7, 3, 2, 1 << 31],
];

/// [channels, suffix_len, row_stride, state_stride] of Pack: one element; padded rows and states; no state rows; an empty
/// suffix; no channels; a Mamba2 prefill shape; more token rows than the 65535 row groups every device allows.
const PACK_SHAPES: [[u32; 4]; 9] = [
    [1, 1, 1, 0],
    [5, 3, 5, 2],
    [33, 17, 40, 3],
    [257, 3, 300, 6],
    [31, 0, 31, 3],
    [7, 4, 7, 0],
    [0, 3, 4, 3],
    [10240, 17, 10240, 3],
    [1, 70000, 1, 1],
];

fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

/// Elements of `rows` rows of `stride` whose last holds `width`.
fn span(
    rows: u64,
    stride: u64,
    width: u64,
) -> usize {
    match rows == 0 || width == 0 {
        true => 0,
        false => ((rows - 1) * stride + width) as usize,
    }
}

/// Valid spans [x (Scan: padded), w, b, state, x_out, b_out, c_out] of a Decode or Scan shape.
fn spans(
    shape: [u32; 7],
    scan: bool,
) -> [usize; 7] {
    let [channels, k, suffix, row, s, inner, proj] = shape.map(u64::from);
    let rows = match (scan, k) {
        (false, _) => suffix,
        (true, 0) => 0,
        (true, k) => suffix + k - 1,
    };
    let x_width = inner.min(channels);
    let b_width = proj.min(channels - x_width);
    let c_width = proj.min(channels - x_width - b_width);
    [
        span(rows, row, channels),
        (channels * k) as usize,
        channels as usize,
        span(channels, s, k.saturating_sub(1)),
        span(suffix, inner, x_width),
        span(suffix, proj, b_width),
        span(suffix, proj, c_width),
    ]
}

/// Valid spans [state_in, x, padded] of a Pack shape.
fn pack_spans([channels, suffix, row, s]: [u32; 4]) -> [usize; 3] {
    let [channels, suffix, row, s] = [channels, suffix, row, s].map(u64::from);
    [(channels * s) as usize, span(suffix, row, channels), span(s + suffix, row, channels)]
}

/// Finite eighths, with the specials at every `every`-th element when `every` > 0.
pub fn values<T: ArrayElement + Float>(
    len: usize,
    seed: usize,
    every: usize,
) -> Vec<T> {
    let specials = specials::<T>();
    (0..len)
        .map(|i| match every > 0 && i % every == 0 {
            true => specials[(i / every + seed) % specials.len()],
            false => T::from((((i + seed) * 37) % 61) as f32 / 8.0 - 3.75).unwrap(),
        })
        .collect()
}

/// [x (Scan: padded), w, b, state] over the valid spans: specials in every fifth input and third state element.
fn inputs<T: ArrayElement + Float>(
    shape: [u32; 7],
    scan: bool,
) -> [Vec<T>; 4] {
    let [x, w, b, state, ..] = spans(shape, scan);
    [values(x, 0, 5), values(w, 1, 0), values(b, 2, 0), values(state, 3, 3)]
}

/// The CPU Decode in `submissions` timed submissions: [x_out, b_out, c_out, next_state], outputs starting as sentinels and
/// next_state as the state in place, and each submission's wall time.
fn cpu_decode<T: ArrayElement + Float + Default>(
    shape: [u32; 7],
    inputs: &[Vec<T>; 4],
    has_bias: bool,
    in_place: bool,
    activation: ActivationType,
    submissions: usize,
) -> ([Vec<T>; 4], Vec<Duration>) {
    let [channels, k, suffix, row, s, inner, proj] = shape;
    let [_, _, _, state_len, x_len, b_len, c_len] = spans(shape, false);
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::Conv1dDecodeKernel::new(&context, T::data_type(), has_bias, in_place)
            .expect("CPU Conv1dDecode");
    let [x, w, b, state] = inputs.each_ref().map(|values| cpu_buffer(&context, values));
    let mut outputs = [x_len, b_len, c_len].map(|len| cpu_buffer(&context, &vec![sentinel::<T>(); len]));
    let mut next_state = cpu_buffer(
        &context,
        &if in_place {
            inputs[3].clone()
        } else {
            vec![sentinel(); state_len]
        },
    );
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let [x_out, b_out, c_out] = &mut outputs;
        let (b, state) = (has_bias.then_some(&b), (!in_place).then_some(&state));
        kernel.encode(
            &x,
            &w,
            b,
            state,
            x_out,
            b_out,
            c_out,
            &mut next_state,
            k,
            row,
            s,
            channels,
            suffix,
            inner,
            proj,
            activation,
            command_buffer,
        );
    });
    let [x_out, b_out, c_out] = &outputs;
    let lens = [(x_out, x_len), (b_out, b_len), (c_out, c_len), (&next_state, state_len)];
    (lens.map(|(buffer, len)| buffer_prefix_to_vec::<Cpu, T>(buffer, len)), times)
}

/// The Vulkan counterpart of `cpu_decode` over guarded ranges of exactly the valid spans, recorded once or, when `timed`,
/// in `median_times` submissions; asserts the inputs and guards.
fn gpu_decode<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &Conv1dDecodeVulkanKernel,
    shape: [u32; 7],
    inputs: &[Vec<T>; 4],
    has_bias: bool,
    in_place: bool,
    activation: ActivationType,
    timed: bool,
) -> ([Vec<T>; 4], Option<(Duration, Duration)>) {
    let [channels, k, suffix, row, s, inner, proj] = shape;
    let [_, _, _, state_len, x_len, b_len, c_len] = spans(shape, false);
    let fill = sentinel::<T>();
    let [x, w, b, state] = inputs.each_ref().map(|values| fixture.guarded(values, fill));
    let outputs = [x_len, b_len, c_len].map(|len| fixture.guarded(&vec![fill; len], fill));
    let next_state = fixture.guarded(
        &if in_place {
            inputs[3].clone()
        } else {
            vec![fill; state_len]
        },
        fill,
    );
    // SAFETY: each range holds exactly the aligned valid span of its argument and written ranges alias nothing.
    let mut record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let [x_out, b_out, c_out] = outputs.each_ref().map(arg);
        kernel.encode(
            arg(&x),
            arg(&w),
            has_bias.then(|| arg(&b)),
            (!in_place).then(|| arg(&state)),
            x_out,
            b_out,
            c_out,
            arg(&next_state),
            k,
            row,
            s,
            channels,
            suffix,
            inner,
            proj,
            activation,
            encoding,
        )
    };
    let times = match timed {
        true => Some(fixture.median_times(&mut record)),
        false => {
            let mut encoding = fixture.encoding();
            record(&mut encoding);
            KernelFixture::complete(encoding);
            None
        },
    };
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for ((name, guarded), values) in ["x", "w", "b", "state"].into_iter().zip([&x, &w, &b, &state]).zip(inputs) {
            KernelFixture::assert_unchanged(guarded, fill, values, name);
        }
        let [x_out, b_out, c_out] = outputs.each_ref().map(|output| KernelFixture::read_guarded(output, fill));
        ([x_out, b_out, c_out, KernelFixture::read_guarded(&next_state, fill)], times)
    }
}

/// The CPU Scan in `submissions` timed submissions over [padded, w, b, _] or, when `pack`, after a Pack of [tokens, _, _,
/// state] into its padded input over one state (state_stride = kernel_size - 1). Returns [x_out, b_out, c_out,
/// state_out], outputs starting as sentinels and state_out as the state when `pack`, Pack's padded and the wall times.
fn cpu_scan<T: ArrayElement + Float + Default>(
    shape: [u32; 7],
    inputs: &[Vec<T>; 4],
    has_bias: bool,
    activation: ActivationType,
    pack: bool,
    submissions: usize,
) -> ([Vec<T>; 4], Option<Vec<T>>, Vec<Duration>) {
    let [channels, k, suffix, row, s, inner, proj] = shape;
    let [padded_len, _, _, state_len, x_len, b_len, c_len] = spans(shape, true);
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::Conv1dScanKernel::new(&context, T::data_type(), has_bias)
        .expect("CPU Conv1dScan");
    let [x, w, b] = [&inputs[0], &inputs[1], &inputs[2]].map(|values| cpu_buffer(&context, values));
    let fill = sentinel::<T>();
    let state = if pack {
        inputs[3].clone()
    } else {
        vec![fill; state_len]
    };
    let lens = [x_len, b_len, c_len, state_len];
    let mut outputs =
        [vec![fill; x_len], vec![fill; b_len], vec![fill; c_len], state].map(|values| cpu_buffer(&context, &values));
    let mut pack = pack.then(|| {
        let pack =
            <<Cpu as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, T::data_type(), T::data_type())
                .expect("CPU Conv1dPack");
        (pack, cpu_buffer(&context, &vec![fill; padded_len]))
    });
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let [x_out, b_out, c_out, state_out] = &mut outputs;
        let b = has_bias.then_some(&b);
        let input = match &mut pack {
            Some((pack, padded)) => {
                pack.encode(&*state_out, &x, &mut *padded, s, row, suffix, channels, command_buffer);
                &*padded
            },
            None => &x,
        };
        kernel.encode(
            input,
            &w,
            b,
            x_out,
            b_out,
            c_out,
            state_out,
            suffix,
            k,
            row,
            s,
            channels,
            inner,
            proj,
            activation,
            command_buffer,
        );
    });
    let padded = pack.map(|(_, padded)| buffer_prefix_to_vec::<Cpu, T>(&padded, padded_len));
    (std::array::from_fn(|i| buffer_prefix_to_vec::<Cpu, T>(&outputs[i], lens[i])), padded, times)
}

/// The Vulkan counterpart of `cpu_scan` over guarded ranges of exactly the valid spans, Pack and Scan in one command
/// buffer when `pack` holds Pack, recorded once or, when `timed`, in `median_times` submissions; asserts the inputs and
/// guards.
fn gpu_scan<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &Conv1dScanVulkanKernel,
    shape: [u32; 7],
    inputs: &[Vec<T>; 4],
    has_bias: bool,
    activation: ActivationType,
    pack: Option<&Conv1dPackVulkanKernel>,
    timed: bool,
) -> ([Vec<T>; 4], Option<Vec<T>>, Option<(Duration, Duration)>) {
    let [channels, k, suffix, row, s, inner, proj] = shape;
    let [padded_len, _, _, state_len, x_len, b_len, c_len] = spans(shape, true);
    let fill = sentinel::<T>();
    let [x, w, b] = [&inputs[0], &inputs[1], &inputs[2]].map(|values| fixture.guarded(values, fill));
    let state = match pack {
        Some(_) => inputs[3].clone(),
        None => vec![fill; state_len],
    };
    let outputs =
        [vec![fill; x_len], vec![fill; b_len], vec![fill; c_len], state].map(|values| fixture.guarded(&values, fill));
    let padded = pack.map(|_| fixture.guarded(&vec![fill; padded_len], fill));
    // SAFETY: each range holds exactly the aligned valid span of its argument and written ranges alias nothing; Scan
    // writes the state Pack read, ordered by the command buffer.
    let mut record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let [x_out, b_out, c_out, state_out] = outputs.each_ref().map(arg);
        let input = match pack {
            Some(pack) => {
                let padded = padded.as_ref().expect("Pack padded");
                pack.encode(state_out.clone(), arg(&x), arg(padded), s, row, suffix, channels, encoding);
                padded
            },
            None => &x,
        };
        kernel.encode(
            arg(input),
            arg(&w),
            has_bias.then(|| arg(&b)),
            x_out,
            b_out,
            c_out,
            state_out,
            suffix,
            k,
            row,
            s,
            channels,
            inner,
            proj,
            activation,
            encoding,
        )
    };
    let times = match timed {
        true => Some(fixture.median_times(&mut record)),
        false => {
            let mut encoding = fixture.encoding();
            record(&mut encoding);
            KernelFixture::complete(encoding);
            None
        },
    };
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for ((name, guarded), values) in ["x", "w", "b"].into_iter().zip([&x, &w, &b]).zip(inputs) {
            KernelFixture::assert_unchanged(guarded, fill, values, name);
        }
        let padded = padded.map(|padded| KernelFixture::read_guarded(&padded, fill));
        (outputs.each_ref().map(|output| KernelFixture::read_guarded(output, fill)), padded, times)
    }
}

/// Pack on the CPU and Vulkan over exactly the valid spans: [CPU padded, Vulkan padded], padded starting as sentinels.
fn pack<S: ArrayElement + Float + Default, I: ArrayElement + Float + Default>(
    fixture: &KernelFixture,
    shape: [u32; 4],
    state: &[S],
    x: &[I],
) -> [Vec<S>; 2] {
    let [channels, suffix, row, s] = shape;
    let padded_len = pack_spans(shape)[2];
    let context = create_context::<Cpu>();
    let cpu_kernel =
        <<Cpu as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, S::data_type(), I::data_type())
            .expect("CPU Conv1dPack");
    let (state_buffer, x_buffer) = (cpu_buffer(&context, state), cpu_buffer(&context, x));
    let mut padded = cpu_buffer(&context, &vec![sentinel::<S>(); padded_len]);
    cpu_submissions(&context, 1, |command_buffer| {
        cpu_kernel.encode(&state_buffer, &x_buffer, &mut padded, s, row, suffix, channels, command_buffer);
    });
    let cpu = buffer_prefix_to_vec::<Cpu, S>(&padded, padded_len);

    let kernel =
        Conv1dPackVulkanKernel::new(&fixture.context, S::data_type(), I::data_type()).expect("Vulkan Conv1dPack");
    let (state_fill, x_fill) = (sentinel::<S>(), sentinel::<I>());
    let (state_in, x_in) = (fixture.guarded(state, state_fill), fixture.guarded(x, x_fill));
    let padded = fixture.guarded(&vec![state_fill; padded_len], state_fill);
    let mut encoding = fixture.encoding();
    // SAFETY: each range holds exactly the aligned valid span of its argument and padded aliases nothing.
    unsafe { kernel.encode(arg(&state_in), arg(&x_in), arg(&padded), s, row, suffix, channels, &mut encoding) };
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&state_in, state_fill, state, "state_in");
        KernelFixture::assert_unchanged(&x_in, x_fill, x, "x");
        [cpu, KernelFixture::read_guarded(&padded, state_fill)]
    }
}

/// Raw copies bit for bit with NaN payloads; slots holding a sample converted through FP32 too, except that F16
/// conversions keep only the NaN class.
fn assert_state<T: ArrayElement + Float + Debug>(
    expected: &[T],
    actual: &[T],
    converted: impl Fn(usize) -> bool,
    case: &str,
) {
    assert_eq!(expected.len(), actual.len(), "{case}: length");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let same = match T::data_type() == DataType::F16 && converted(index) && expected.is_nan() {
            true => actual.is_nan(),
            false => bytemuck::bytes_of(&expected) == bytemuck::bytes_of(&actual),
        };
        assert!(same, "{case}: element {index}: CPU {expected:?}, Vulkan {actual:?}");
    }
}

/// Outputs bit for bit up to the NaN payload with IDENTITY, otherwise within activation_check's oracle of the CPU's
/// T(acc), its IDENTITY outputs; the state by `assert_state`.
fn compare<T: ArrayElement + Float + Debug>(
    identity: &[Vec<T>; 4],
    cpu: &[Vec<T>; 4],
    gpu: &[Vec<T>; 4],
    activation: ActivationType,
    converted: impl Fn(usize) -> bool,
    label: &str,
) {
    for (index, name) in ["x_out", "b_out", "c_out"].into_iter().enumerate() {
        let case = format!("{label} {name}");
        match activation {
            ActivationType::IDENTITY => KernelFixture::assert_bits(&cpu[index], &gpu[index], &case),
            _ => {
                activation_check(&identity[index], activation, &cpu[index], &gpu[index], &case);
            },
        }
    }
    assert_state(&cpu[3], &gpu[3], converted, &format!("{label} state"));
}

/// State slots holding converted samples after Decode: the newest `suffix_len` taps in place, the tail otherwise.
fn decode_converted(
    shape: [u32; 7],
    in_place: bool,
) -> impl Fn(usize) -> bool {
    let [_, k, suffix, _, s, ..] = shape.map(|value| value as usize);
    let (taps, shifts) = (
        k.saturating_sub(1),
        if in_place {
            suffix
        } else {
            suffix.min(1)
        },
    );
    move |index| s > 0 && index % s < taps && index % s + shifts >= taps
}

fn decode_matches_cpu<T: ArrayElement + Float + Debug + Default>() {
    let fixture = KernelFixture::new();
    for (has_bias, in_place) in [(false, false), (false, true), (true, false), (true, true)] {
        let kernel = Conv1dDecodeVulkanKernel::new(&fixture.context, T::data_type(), has_bias, in_place)
            .expect("Vulkan Conv1dDecode");
        for shape in SHAPES {
            let inputs = inputs::<T>(shape, false);
            let (identity, _) = cpu_decode(shape, &inputs, has_bias, in_place, ActivationType::IDENTITY, 1);
            for activation in ACTIVATIONS {
                let label =
                    format!("{:?} Decode {shape:?} bias {has_bias} in place {in_place} {activation:?}", T::data_type());
                let (cpu, _) = cpu_decode(shape, &inputs, has_bias, in_place, activation, 1);
                let (gpu, _) = gpu_decode(&fixture, &kernel, shape, &inputs, has_bias, in_place, activation, false);
                compare(&identity, &cpu, &gpu, activation, decode_converted(shape, in_place), &label);
            }
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn decode_matches_cpu_all_types() {
    decode_matches_cpu::<f32>();
    decode_matches_cpu::<f16>();
    decode_matches_cpu::<bf16>();
}

fn scan_matches_cpu<T: ArrayElement + Float + Debug + Default>() {
    let fixture = KernelFixture::new();
    for has_bias in [false, true] {
        let kernel =
            Conv1dScanVulkanKernel::new(&fixture.context, T::data_type(), has_bias).expect("Vulkan Conv1dScan");
        for shape in SHAPES {
            let inputs = inputs::<T>(shape, true);
            let (identity, ..) = cpu_scan(shape, &inputs, has_bias, ActivationType::IDENTITY, false, 1);
            let [_, k, _, _, s, ..] = shape.map(|value| value as usize);
            for activation in ACTIVATIONS {
                let label = format!("{:?} Scan {shape:?} bias {has_bias} {activation:?}", T::data_type());
                let (cpu, ..) = cpu_scan(shape, &inputs, has_bias, activation, false, 1);
                let (gpu, ..) = gpu_scan(&fixture, &kernel, shape, &inputs, has_bias, activation, None, false);
                compare(&identity, &cpu, &gpu, activation, |index| s > 0 && index % s + 1 < k, &label);
            }
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn scan_matches_cpu_all_types() {
    scan_matches_cpu::<f32>();
    scan_matches_cpu::<f16>();
    scan_matches_cpu::<bf16>();
}

/// Raw state rows with every special's payload and converted token rows, each pair of state and input types.
fn pack_matches_cpu<S: ArrayElement + Float + Debug + Default, I: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture
) {
    for shape in PACK_SHAPES {
        let [state_len, x_len, _] = pack_spans(shape);
        let [cpu, gpu] = pack(fixture, shape, &values::<S>(state_len, 3, 3), &values::<I>(x_len, 0, 5));
        assert_same_bits(&cpu, &gpu, &format!("Pack {:?} <- {:?} {shape:?}", S::data_type(), I::data_type()));
    }
}

#[uzu_test]
fn pack_matches_cpu_all_types() {
    let fixture = KernelFixture::new();
    pack_matches_cpu::<f32, f32>(&fixture);
    pack_matches_cpu::<f32, bf16>(&fixture);
    pack_matches_cpu::<bf16, f32>(&fixture);
    pack_matches_cpu::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}

/// One channel and one token: Decode in place over `state` and `x`, and Scan over the padded rows `state`, `x`, on the
/// CPU and Vulkan. Returns [CPU Decode, Vulkan Decode, CPU Scan, Vulkan Scan], each [x_out, b_out, c_out, state].
fn single<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    w: &[T],
    bias: Option<T>,
    state: &[T],
    x: T,
    activation: ActivationType,
) -> [[Vec<T>; 4]; 4] {
    let k = w.len() as u32;
    let shape = [1, k, 1, 1, k.saturating_sub(1), 1, 0];
    let (has_bias, b) = (bias.is_some(), vec![bias.unwrap_or_else(T::zero)]);
    let padded = if k == 0 {
        Vec::new()
    } else {
        [state, &[x][..]].concat()
    };
    let decode = [vec![x], w.to_vec(), b.clone(), state.to_vec()];
    let scan = [padded, w.to_vec(), b, Vec::new()];
    let decode_kernel =
        Conv1dDecodeVulkanKernel::new(&fixture.context, T::data_type(), has_bias, true).expect("Vulkan Conv1dDecode");
    let scan_kernel =
        Conv1dScanVulkanKernel::new(&fixture.context, T::data_type(), has_bias).expect("Vulkan Conv1dScan");
    [
        cpu_decode(shape, &decode, has_bias, true, activation, 1).0,
        gpu_decode(fixture, &decode_kernel, shape, &decode, has_bias, true, activation, false).0,
        cpu_scan(shape, &scan, has_bias, activation, false, 1).0,
        gpu_scan(fixture, &scan_kernel, shape, &scan, has_bias, activation, None, false).0,
    ]
}

/// `single` with IDENTITY must give exactly `expected` (any NaN for a NaN) from all four kernels.
fn witness<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    name: &str,
    w: &[T],
    bias: Option<T>,
    state: &[T],
    x: T,
    expected: T,
) {
    let results = single(fixture, w, bias, state, x, ActivationType::IDENTITY);
    for (kernel, result) in ["CPU Decode", "Vulkan Decode", "CPU Scan", "Vulkan Scan"].into_iter().zip(&results) {
        let value = result[0][0];
        let same = match expected.is_nan() {
            true => value.is_nan(),
            false => bytemuck::bytes_of(&value) == bytemuck::bytes_of(&expected),
        };
        assert!(same, "{name}: {kernel} gave {value:?}, expected {expected:?}");
    }
}

/// Independent FP32 results of the staged accumulation: unfused products, the bias first and taps in order, the +0 start
/// and a -0 bias, subnormal products and sums, overflow before the bias, inf - inf and kernel_size 0. A BF16 product of
/// finite values can overflow FP32: ordered, bias + w x is +inf, where an FMA would keep bf16::MAX.
#[uzu_test]
fn arithmetic_witnesses() {
    let fixture = KernelFixture::new();
    let (p, tiny) = (|e: i32| 2f32.powi(e), f32::from_bits(1));
    let cases: [(&str, &[f32], Option<f32>, &[f32], f32, f32); 8] = [
        ("unfused: an FMA keeps 2^-24", &[-1.0, 1.0 + p(-12)], None, &[1.0 + p(-11)], 1.0 + p(-12), 0.0),
        ("bias first, taps in order", &[1.0, -1.0], Some(p(24)), &[1.0], p(24), 0.0),
        ("+0 start", &[1.0], None, &[], -0.0, 0.0),
        ("-0 bias", &[1.0], Some(-0.0), &[], -0.0, -0.0),
        ("subnormal products and sums", &[p(-74), 1.0], Some(tiny), &[p(-75)], tiny, 3.0 * tiny),
        ("product overflow before the bias", &[f32::MAX], Some(-f32::MAX), &[], 2.0, f32::INFINITY),
        ("inf - inf", &[1.0], Some(f32::NEG_INFINITY), &[], f32::INFINITY, f32::NAN),
        ("kernel_size 0: the bias alone", &[], Some(1.5), &[], 7.0, 1.5),
    ];
    for (name, w, bias, state, x, expected) in cases {
        witness(&fixture, name, w, bias, state, x, expected);
    }
    let two = bf16::from_f32(2.0);
    witness(&fixture, "BF16 product overflow", &[bf16::MAX], Some(-bf16::MAX), &[], two, bf16::INFINITY);
    fixture.assert_clean();
}

/// Stored state moves bit for bit and samples converted through FP32 quiet BF16 NaNs: Decode shifts a signalling NaN and
/// converts its new tail, Scan converts every state tap, Pack copies state rows and converts token rows, rounding F32 to
/// nearest even.
#[uzu_test]
fn copies_keep_and_conversions_quiet_nans() {
    let fixture = KernelFixture::new();
    let [one, signalling, quiet] = [0x3F80, 0x7F81, 0x7FC1].map(bf16::from_bits);
    let [cpu_decode, gpu_decode, cpu_scan, gpu_scan] =
        single(&fixture, &[one; 3], None, &[one, signalling], signalling, ActivationType::IDENTITY);
    for (name, state, expected) in [
        ("CPU Decode", &cpu_decode[3], [signalling, quiet]),
        ("Vulkan Decode", &gpu_decode[3], [signalling, quiet]),
        ("CPU Scan", &cpu_scan[3], [quiet, quiet]),
        ("Vulkan Scan", &gpu_scan[3], [quiet, quiet]),
    ] {
        assert_same_bits(&expected, state, name);
    }
    for (padded, name) in pack(&fixture, [1, 1, 1, 1], &[signalling], &[signalling]).iter().zip(["CPU", "Vulkan"]) {
        assert_same_bits(&[signalling, quiet], padded, &format!("{name} Pack BF16"));
    }
    let ties = [0x3F80_8000, 0x3F81_8000].map(f32::from_bits);
    for (padded, name) in pack(&fixture, [1, 2, 1, 1], &[one], &ties).iter().zip(["CPU", "Vulkan"]) {
        assert_same_bits(
            &[one, bf16::from_bits(0x3F80), bf16::from_bits(0x3F82)],
            padded,
            &format!("{name} Pack ties"),
        );
    }
    fixture.assert_clean();
}

/// Tokens in order: in place each token reads the state the previous one stored, out of place every token reads the
/// original state, and next_state ends as that state shifted once with the last token's input.
#[uzu_test]
fn decode_keeps_token_order() {
    let fixture = KernelFixture::new();
    let shape = [1, 4, 3, 1, 3, 1, 0];
    let inputs = [vec![10.0, 20.0, 30.0], vec![1.0, 0.0, 0.0, 0.0], vec![0.0], vec![1.0, 2.0, 3.0]];
    for (in_place, outputs, state) in [(true, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), (false, [1.0; 3], [2.0, 3.0, 30.0])]
    {
        let kernel = Conv1dDecodeVulkanKernel::new(&fixture.context, DataType::F32, false, in_place).expect("Decode");
        let (cpu, _) = cpu_decode::<f32>(shape, &inputs, false, in_place, ActivationType::IDENTITY, 1);
        let (gpu, _) = gpu_decode(&fixture, &kernel, shape, &inputs, false, in_place, ActivationType::IDENTITY, false);
        for (name, result) in [("CPU", cpu), ("Vulkan", gpu)] {
            assert_same_bits(&outputs, &result[0], &format!("{name} in place {in_place} outputs"));
            assert_same_bits(&state, &result[3], &format!("{name} in place {in_place} state"));
        }
    }
    fixture.assert_clean();
}

/// A K = 1 Decode or Scan shape of one token over `channels` channels, all stored in x_out.
fn rounding_shape(channels: usize) -> [u32; 7] {
    let channels = channels as u32;
    [channels, 1, 1, channels, 0, channels, 0]
}

/// Witnesses that the accumulator is rounded to T before the activation, from a bounded grid: the bias every T value with
/// magnitude in [1/8, 32) and x a quarter or three quarters of the bias's ULP either way, w = 1. A witness's CPU result
/// T(act(T(bias + x))) differs from T(act(bias + x)) and is the only T value within the activation oracle's bounds of
/// T(bias + x). Returns the inputs [x, w, b, _] and the CPU results of at most 256 witnesses.
fn rounding_witnesses<T: ArrayElement + Float + Debug + Default>(activation: ActivationType) -> ([Vec<T>; 4], Vec<T>) {
    let mut candidates = Vec::new();
    for bits in 0..=u16::MAX {
        let bias: T = bytemuck::pod_read_unaligned(&bits.to_ne_bytes()[..size_of::<T>()]);
        let magnitude = bias.to_f32().unwrap().abs();
        if (0.125..32.0).contains(&magnitude) {
            let quarter = T::epsilon().to_f32().unwrap() * magnitude.log2().floor().exp2() / 4.0;
            for offset in [quarter, -quarter, 3.0 * quarter, -3.0 * quarter] {
                candidates.push((bias, T::from(offset).unwrap()));
            }
        }
    }
    let shape = rounding_shape(candidates.len());
    let inputs: [Vec<T>; 4] = [
        candidates.iter().map(|&(_, x)| x).collect(),
        vec![T::one(); candidates.len()],
        candidates.iter().map(|&(bias, _)| bias).collect(),
        Vec::new(),
    ];
    let (rounded, _) = cpu_decode(shape, &inputs, true, true, ActivationType::IDENTITY, 1);
    let (activated, _) = cpu_decode(shape, &inputs, true, true, activation, 1);
    let to_t = |value: f64| T::from(value).unwrap();
    let same = |a: T, b: T| bytemuck::bytes_of(&a) == bytemuck::bytes_of(&b);
    let witnesses = (0..candidates.len())
        .filter(|&i| {
            let (bias, x) = (inputs[2][i].to_f32().unwrap(), inputs[0][i].to_f32().unwrap());
            let unrounded = T::from(activation.activate(bias + x)).unwrap();
            let ((low, high), _) = oracle(rounded[0][i].to_f64().unwrap(), activation);
            let result = activated[0][i];
            result.is_finite() && !same(result, unrounded) && same(to_t(low), result) && same(to_t(high), result)
        })
        .take(256)
        .collect::<Vec<_>>();
    let pick = |values: &[T]| witnesses.iter().map(|&i| values[i]).collect::<Vec<_>>();
    ([pick(&inputs[0]), pick(&inputs[1]), pick(&inputs[2]), Vec::new()], pick(&activated[0]))
}

/// The bounded search finds rounding witnesses for every 16-bit type and non-identity activation (CPU only).
#[uzu_test]
fn rounding_witnesses_exist() {
    fn count<T: ArrayElement + Float + Debug + Default>() {
        for activation in &ACTIVATIONS[1..] {
            let found = rounding_witnesses::<T>(*activation).1.len();
            eprintln!("{:?} {activation:?}: {found} rounding witnesses", T::data_type());
            assert!(found > 0, "{:?} {activation:?}: the bounded grid holds no rounding witness", T::data_type());
        }
    }
    count::<f16>();
    count::<bf16>();
}

/// Decode and Scan give every rounding witness's CPU result exactly.
#[uzu_test]
fn rounding_witnesses_match_vulkan() {
    fn check<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let decode = Conv1dDecodeVulkanKernel::new(&fixture.context, T::data_type(), true, true).expect("Decode");
        let scan = Conv1dScanVulkanKernel::new(&fixture.context, T::data_type(), true).expect("Scan");
        for activation in &ACTIVATIONS[1..] {
            let (inputs, expected) = rounding_witnesses::<T>(*activation);
            let shape = rounding_shape(expected.len());
            let label = format!("{:?} {activation:?} rounding witnesses", T::data_type());
            let (decoded, _) = gpu_decode(fixture, &decode, shape, &inputs, true, true, *activation, false);
            assert_same_bits(&expected, &decoded[0], &format!("{label} Decode"));
            let (scanned, ..) = gpu_scan(fixture, &scan, shape, &inputs, true, *activation, None, false);
            assert_same_bits(&expected, &scanned[0], &format!("{label} Scan"));
        }
    }
    let fixture = KernelFixture::new();
    check::<f16>(&fixture);
    check::<bf16>(&fixture);
    fixture.assert_clean();
}

/// `guarded` from element `skip` on, for one dispatch of a chain.
fn from_element<T>(
    (buffer, range): &(Arc<VkBuffer>, Range<u64>),
    skip: usize,
) -> (&Arc<VkBuffer>, Range<u64>) {
    (buffer, range.start + (skip * size_of::<T>()) as u64..range.end)
}

/// Producer and consumer in one command buffer: Pack then Scan over one state range, as Mamba2's prefill, and two in-place
/// Decode dispatches of one token each against one Decode of both, all against the CPU.
#[uzu_test]
fn chained_dispatches_match_cpu() {
    let fixture = KernelFixture::new();
    let fill = sentinel::<f32>();
    let shape = [33, 4, 5, 33, 3, 16, 8];
    let [channels, k, suffix, row, s, inner, proj] = shape;
    let [state_len, x_len, _] = pack_spans([channels, suffix, row, s]);
    let [_, w, b, _] = inputs::<f32>(shape, true);
    let pair = [values::<f32>(x_len, 0, 5), w, b, values::<f32>(state_len, 3, 3)];
    // With state_stride = kernel_size - 1, Scan rewrites every state element Pack read.
    let (cpu, cpu_padded, _) = cpu_scan(shape, &pair, true, ActivationType::IDENTITY, true, 1);
    let pack = Conv1dPackVulkanKernel::new(&fixture.context, DataType::F32, DataType::F32).expect("Pack");
    let scan = Conv1dScanVulkanKernel::new(&fixture.context, DataType::F32, true).expect("Scan");
    let (gpu, gpu_padded, _) =
        gpu_scan(&fixture, &scan, shape, &pair, true, ActivationType::IDENTITY, Some(&pack), false);
    compare(&cpu, &cpu, &gpu, ActivationType::IDENTITY, |_| false, "Pack then Scan");
    KernelFixture::assert_bits(&cpu_padded.expect("padded"), &gpu_padded.expect("padded"), "Pack then Scan padded");

    // Decode: token by token in place in one command buffer, against both tokens at once.
    let decode_shape = [channels, k, 2, row, s, inner, proj];
    let decode_inputs = inputs::<f32>(decode_shape, false);
    let (cpu, _) = cpu_decode(decode_shape, &decode_inputs, true, true, ActivationType::IDENTITY, 1);
    let decode = Conv1dDecodeVulkanKernel::new(&fixture.context, DataType::F32, true, true).expect("Decode");
    let [x_in, w_in, b_in, next_state] = decode_inputs.each_ref().map(|values| fixture.guarded(values, fill));
    let [_, _, _, _, x_out_len, b_out_len, c_out_len] = spans(decode_shape, false);
    let decoded = [x_out_len, b_out_len, c_out_len].map(|len| fixture.guarded(&vec![fill; len], fill));
    let mut encoding = fixture.encoding();
    for token in 0..2 {
        let [x_out, b_out, c_out] = [(0, inner), (1, proj), (2, proj)]
            .map(|(output, width)| from_element::<f32>(&decoded[output], token * width as usize));
        let x_token = from_element::<f32>(&x_in, token * row as usize);
        // SAFETY: each range starts at the token's elements of a valid span; the dispatches write disjoint output rows and
        // continue one state in command buffer order.
        unsafe {
            decode.encode(
                x_token,
                arg(&w_in),
                Some(arg(&b_in)),
                None,
                x_out,
                b_out,
                c_out,
                arg(&next_state),
                k,
                row,
                s,
                channels,
                1,
                inner,
                proj,
                ActivationType::IDENTITY,
                &mut encoding,
            )
        };
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for (index, name) in ["x_out", "b_out", "c_out"].into_iter().enumerate() {
            let gpu = KernelFixture::read_guarded(&decoded[index], fill);
            KernelFixture::assert_bits(&cpu[index], &gpu, &format!("chained Decode {name}"));
        }
        assert_same_bits(&cpu[3], &KernelFixture::read_guarded(&next_state, fill), "chained Decode state");
    }
    fixture.assert_clean();
}

/// No work at u32::MAX scalars, where the CPU loops would take seconds: no dispatch is recorded and nothing changes.
#[uzu_test]
fn zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let fill = sentinel::<f32>();
    let empty = fixture.guarded::<f32>(&[], fill);
    let pack = Conv1dPackVulkanKernel::new(&fixture.context, DataType::F32, DataType::F32).expect("Pack");
    let decode = Conv1dDecodeVulkanKernel::new(&fixture.context, DataType::F32, false, true).expect("Decode");
    let scan = Conv1dScanVulkanKernel::new(&fixture.context, DataType::F32, false).expect("Scan");
    let identity = ActivationType::IDENTITY;
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    // SAFETY: no shape below indexes any element, and nothing is recorded.
    unsafe {
        pack.encode(e(), e(), e(), u32::MAX, 4, u32::MAX, 0, &mut encoding);
        for [channels, k, suffix, row, s, inner, proj] in [[0, 4, u32::MAX, 4, 3, 1, 1], [u32::MAX, 0, 0, 4, 0, 1, 1]] {
            decode.encode(
                e(),
                e(),
                None,
                None,
                e(),
                e(),
                e(),
                e(),
                k,
                row,
                s,
                channels,
                suffix,
                inner,
                proj,
                identity,
                &mut encoding,
            );
        }
        for [channels, k, suffix, row, s, inner, proj] in
            [[0, u32::MAX, u32::MAX, 4, 3, 1, 1], [u32::MAX, 1, 0, 4, 0, 1, 1]]
        {
            scan.encode(
                e(),
                e(),
                None,
                e(),
                e(),
                e(),
                e(),
                suffix,
                k,
                row,
                s,
                channels,
                inner,
                proj,
                identity,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, fill, &[], "empty") };
    fixture.assert_clean();
}

/// Run alone, without sync validation: `... conv1d_test::throughput -- --ignored --nocapture`. At Mamba2's 10240 channels
/// of kernel size 4 (x 8192, B and C 1024, a bias, SILU), times Decode of one token in place, and Pack then Scan of 512
/// tokens over one state range in one command buffer, from nonzero odd sixteenths. Prints the GPU and wall medians of 10
/// Vulkan submissions after 3 warm-up ones and the CPU kernels' wall median. The buffers after all 13 submissions must
/// match the CPU's after as many. BF16 measures the kernels in another dtype, not a BF16 Mamba2 model.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let (silu, identity, dtype) = (ActivationType::SILU, ActivationType::IDENTITY, T::data_type());
        let decode = Conv1dDecodeVulkanKernel::new(&fixture.context, dtype, true, true).expect("Decode");
        let scan = Conv1dScanVulkanKernel::new(&fixture.context, dtype, true).expect("Scan");
        let pack = Conv1dPackVulkanKernel::new(&fixture.context, dtype, dtype).expect("Pack");
        let print = |kernel: &str, times: Option<(Duration, Duration)>, cpu: Vec<Duration>| {
            let ((gpu, wall), mut cpu) = (times.expect("timed"), cpu[3..].to_vec());
            cpu.sort();
            eprintln!("Conv1d{kernel} {dtype:?}: GPU {gpu:?}, wall {wall:?}; CPU wall {:?}", cpu[cpu.len() / 2]);
        };
        let nonzero = |len, seed| values::<T>(len, seed, 0).into_iter().map(|v| v + T::from(0.0625).unwrap());
        for suffix in [1, 512] {
            let shape = [10240, 4, suffix, 10240, 3, 8192, 1024];
            let [x, w, b, state, ..] = spans(shape, false);
            let inputs: [Vec<T>; 4] =
                [(x, 0), (w, 1), (b, 2), (state, 3)].map(|(len, seed)| nonzero(len, seed).collect());
            let label = format!("{dtype:?} {shape:?}");
            if suffix == 1 {
                let (expected, _) = cpu_decode(shape, &inputs, true, true, identity, 13);
                let (cpu, cpu_times) = cpu_decode(shape, &inputs, true, true, silu, 13);
                let (gpu, times) = gpu_decode(fixture, &decode, shape, &inputs, true, true, silu, true);
                compare(&expected, &cpu, &gpu, silu, |_| false, &format!("Decode {label}"));
                print("Decode", times, cpu_times);
            } else {
                let (expected, ..) = cpu_scan(shape, &inputs, true, identity, true, 13);
                let (cpu, cpu_padded, cpu_times) = cpu_scan(shape, &inputs, true, silu, true, 13);
                let (gpu, gpu_padded, times) = gpu_scan(fixture, &scan, shape, &inputs, true, silu, Some(&pack), true);
                compare(&expected, &cpu, &gpu, silu, |_| false, &format!("Pack then Scan {label}"));
                let padded = (cpu_padded.expect("padded"), gpu_padded.expect("padded"));
                assert_same_bits(&padded.0, &padded.1, &format!("Pack then Scan {label} padded"));
                print("Pack+Scan", times, cpu_times);
            }
        }
    }
    let fixture = KernelFixture::new();
    // Round 0 settles the GPU clocks; compare round 1.
    for round in 0..2 {
        eprintln!("Conv1d throughput round {round}");
        measure::<f32>(&fixture);
        measure::<bf16>(&fixture);
    }
    fixture.assert_clean();
}
