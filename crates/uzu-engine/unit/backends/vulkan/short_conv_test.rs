use std::{
    fmt::Debug,
    mem::size_of,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{ShortConvCase, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBuffer, Context, Kernels,
            kernel::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel, ShortConvTrieKernel},
        },
        cpu::Cpu,
        vulkan::{
            ShortConvDecodeVulkanKernel, ShortConvPackVulkanKernel, ShortConvPrefillVulkanKernel,
            ShortConvTrieVulkanKernel, VkBuffer, VkCommandBufferEncoding,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// (model_dim, kernel_size, suffix_len, state_stride, in_proj_stride): odd and real model sizes, kernel sizes 1 to 5,
/// 1 to 17-token suffixes, states padded past kernel_size - 1 and in_proj rows padded past 3 model_dim; an empty suffix
/// whose Prefill still copies the state, no channels, and neither taps nor tokens.
const SHAPES: [(u32, u32, u32, u32, u32); 10] = [
    (1, 1, 1, 0, 3),
    (31, 2, 3, 1, 98),
    (33, 4, 17, 3, 106),
    (257, 5, 3, 6, 772),
    (31, 5, 1, 4, 93),
    (33, 4, 0, 3, 99),
    (2048, 4, 17, 3, 6144),
    (257, 2, 17, 3, 771),
    (0, 4, 3, 3, 0),
    (31, 1, 0, 0, 93),
];

/// Trie parents of 17 nodes: three roots, chains and branches.
const BRANCH: [i32; 17] = [-1, 0, 0, 1, 2, -1, 5, 3, 4, 8, 6, 10, -1, 12, 7, 9, 13];

fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

pub fn arg(guarded: &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
    (&guarded.0, guarded.1.clone())
}

/// CPU buffers cannot be empty: an empty payload, which the kernels never read, gets one placeholder element.
pub fn cpu_buffer<T: ArrayElement + Default>(
    context: &<Cpu as Backend>::Context,
    values: &[T],
) -> <Cpu as Backend>::GlobalBuffer {
    let placeholder = [T::default()];
    create_buffer_with_data::<Cpu, T>(
        context,
        if values.is_empty() {
            &placeholder
        } else {
            values
        },
    )
}

/// Wall time of each of `submissions` CPU submissions of what `encode` records.
pub fn cpu_submissions(
    context: &<Cpu as Backend>::Context,
    submissions: usize,
    mut encode: impl FnMut(&mut <<Cpu as Backend>::CommandBuffer as CommandBuffer>::Encoding),
) -> Vec<Duration> {
    (0..submissions)
        .map(|_| {
            let start = Instant::now();
            let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
            encode(&mut command_buffer);
            submit_command_buffer(command_buffer);
            start.elapsed()
        })
        .collect()
}

/// The CPU kernels through the shared traits, recorded once per submission in `submissions` timed submissions:
/// Prefill after a Pack into its padded input when `pack`, otherwise from `case.padded`. Returns (padded, out,
/// state_out) and each submission's wall time.
fn cpu_prefill<T: ArrayElement + Float + Default>(
    case: &ShortConvCase<T>,
    pack: bool,
    submissions: usize,
) -> ([Vec<T>; 3], Vec<Duration>) {
    let context = create_context::<Cpu>();
    let in_proj = cpu_buffer(&context, &case.in_proj);
    let w = cpu_buffer(&context, &case.w);
    let b = case.b.as_ref().map(|b| cpu_buffer(&context, b));
    let mut padded = cpu_buffer(&context, &case.padded);
    let mut out = cpu_buffer(&context, &vec![sentinel::<T>(); (case.suffix_len * case.model_dim) as usize]);
    let mut state_out = cpu_buffer(&context, &case.next_state);
    let state = cpu_buffer(&context, &case.state);
    let pack = pack.then(|| {
        <<Cpu as Backend>::Kernels as Kernels>::ShortConvPackKernel::new(&context, T::data_type())
            .expect("CPU ShortConvPack")
    });
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ShortConvPrefillKernel::new(
        &context,
        T::data_type(),
        DataType::F32,
        b.is_some(),
    )
    .expect("CPU ShortConvPrefill");
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        if let Some(pack) = &pack {
            let (stride, suffix, row, dim) = (case.state_stride, case.suffix_len, case.in_proj_stride, case.model_dim);
            pack.encode(&state, &in_proj, &mut padded, stride, suffix, row, dim, command_buffer);
        }
        kernel.encode(
            &padded,
            &in_proj,
            &w,
            b.as_ref(),
            &mut out,
            &mut state_out,
            case.suffix_len,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            command_buffer,
        );
    });
    let outputs = [
        buffer_prefix_to_vec::<Cpu, T>(&padded, case.padded.len()),
        buffer_prefix_to_vec::<Cpu, T>(&out, (case.suffix_len * case.model_dim) as usize),
        buffer_prefix_to_vec::<Cpu, T>(&state_out, case.next_state.len()),
    ];
    (outputs, times)
}

/// Returns (out, next_state); next_state starts as `case.state` in place, otherwise as `case.next_state`.
fn cpu_decode<T: ArrayElement + Float + Default>(
    case: &ShortConvCase<T>,
    state_in_place: bool,
    submissions: usize,
) -> ([Vec<T>; 2], Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ShortConvDecodeKernel::new(
        &context,
        T::data_type(),
        DataType::F32,
        case.b.is_some(),
        state_in_place,
    )
    .expect("CPU ShortConvDecode");
    let b = case.b.as_ref().map(|b| cpu_buffer(&context, b));
    let state = (!state_in_place).then(|| cpu_buffer(&context, &case.state));
    let mut out = cpu_buffer(&context, &vec![sentinel::<T>(); (case.suffix_len * case.model_dim) as usize]);
    let mut next_state = cpu_buffer(
        &context,
        match state_in_place {
            true => &case.state,
            false => &case.next_state,
        },
    );
    let (in_proj, w) = (cpu_buffer(&context, &case.in_proj), cpu_buffer(&context, &case.w));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        kernel.encode(
            &in_proj,
            &w,
            b.as_ref(),
            state.as_ref(),
            &mut out,
            &mut next_state,
            case.suffix_len,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            command_buffer,
        )
    });
    let outputs = [
        buffer_prefix_to_vec::<Cpu, T>(&out, (case.suffix_len * case.model_dim) as usize),
        buffer_prefix_to_vec::<Cpu, T>(&next_state, case.state.len()),
    ];
    (outputs, times)
}

/// Returns (out, suffix_state); suffix_state starts as sentinels.
fn cpu_trie<T: ArrayElement + Float + Default>(
    case: &ShortConvCase<T>,
    submissions: usize,
) -> ([Vec<T>; 2], Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ShortConvTrieKernel::new(
        &context,
        T::data_type(),
        DataType::F32,
        case.b.is_some(),
    )
    .expect("CPU ShortConvTrie");
    let b = case.b.as_ref().map(|b| cpu_buffer(&context, b));
    let suffix_state_len = case.suffix_len as usize * case.state.len();
    let mut out = cpu_buffer(&context, &vec![sentinel::<T>(); (case.suffix_len * case.model_dim) as usize]);
    let mut suffix_state = cpu_buffer(&context, &vec![sentinel::<T>(); suffix_state_len]);
    let (in_proj, w) = (cpu_buffer(&context, &case.in_proj), cpu_buffer(&context, &case.w));
    let (base_state, parents) = (cpu_buffer(&context, &case.state), cpu_buffer(&context, &case.parents));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        kernel.encode(
            &in_proj,
            &w,
            b.as_ref(),
            &base_state,
            &parents,
            &mut out,
            &mut suffix_state,
            case.suffix_len,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            command_buffer,
        )
    });
    let outputs = [
        buffer_prefix_to_vec::<Cpu, T>(&out, (case.suffix_len * case.model_dim) as usize),
        buffer_prefix_to_vec::<Cpu, T>(&suffix_state, suffix_state_len),
    ];
    (outputs, times)
}

/// Guarded Vulkan buffers of the read-only payloads shared by the kernels: (in_proj, w, b, state).
fn gpu_inputs<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    case: &ShortConvCase<T>,
) -> (
    (Arc<VkBuffer>, Range<u64>),
    (Arc<VkBuffer>, Range<u64>),
    Option<(Arc<VkBuffer>, Range<u64>)>,
    (Arc<VkBuffer>, Range<u64>),
) {
    let s = sentinel::<T>();
    (
        fixture.guarded(&case.in_proj, s),
        fixture.guarded(&case.w, sentinel::<f32>()),
        case.b.as_ref().map(|b| fixture.guarded(b, sentinel::<f32>())),
        fixture.guarded(&case.state, s),
    )
}

/// Asserts the read-only payloads of `gpu_inputs` and every guard unchanged.
///
/// # Safety
/// Every command buffer using the buffers has completed.
unsafe fn assert_inputs_unchanged<T: ArrayElement + Float>(
    case: &ShortConvCase<T>,
    (in_proj, w, b, state): &(
        (Arc<VkBuffer>, Range<u64>),
        (Arc<VkBuffer>, Range<u64>),
        Option<(Arc<VkBuffer>, Range<u64>)>,
        (Arc<VkBuffer>, Range<u64>),
    ),
) {
    unsafe {
        KernelFixture::assert_unchanged(in_proj, sentinel::<T>(), &case.in_proj, "in_proj");
        KernelFixture::assert_unchanged(w, sentinel::<f32>(), &case.w, "w");
        if let (Some(b), Some(payload)) = (b, &case.b) {
            KernelFixture::assert_unchanged(b, sentinel::<f32>(), payload, "b");
        }
        KernelFixture::assert_unchanged(state, sentinel::<T>(), &case.state, "state");
    }
}

/// GPU and wall time of each of `submissions` Vulkan submissions of what `encode` records.
fn gpu_submissions(
    fixture: &KernelFixture,
    submissions: usize,
    mut encode: impl FnMut(&mut VkCommandBufferEncoding),
) -> Vec<(Duration, Duration)> {
    (0..submissions)
        .map(|_| {
            let start = Instant::now();
            let mut encoding = fixture.encoding();
            encode(&mut encoding);
            (KernelFixture::complete(encoding).gpu_execution_time(), start.elapsed())
        })
        .collect()
}

/// The Vulkan counterpart of `cpu_prefill` over guarded ranges, Pack and Prefill in one command buffer.
fn gpu_prefill<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    case: &ShortConvCase<T>,
    pack: bool,
    submissions: usize,
) -> ([Vec<T>; 3], Vec<(Duration, Duration)>) {
    let s = sentinel::<T>();
    let inputs = gpu_inputs(fixture, case);
    let (in_proj, w, b, state) = &inputs;
    let padded = fixture.guarded(&case.padded, s);
    let out = fixture.guarded(&vec![s; (case.suffix_len * case.model_dim) as usize], s);
    let state_out = fixture.guarded(&case.next_state, s);
    let kernel = ShortConvPrefillVulkanKernel::new(&fixture.context, T::data_type(), DataType::F32, case.b.is_some())
        .expect("Vulkan ShortConvPrefill");
    let pack =
        pack.then(|| ShortConvPackVulkanKernel::new(&fixture.context, T::data_type()).expect("Vulkan ShortConvPack"));
    // SAFETY: every range holds the elements its shape indexes, aligned; outputs do not alias other arguments.
    let times = gpu_submissions(fixture, submissions, |encoding| unsafe {
        if let Some(pack) = &pack {
            let (stride, suffix, row, dim) = (case.state_stride, case.suffix_len, case.in_proj_stride, case.model_dim);
            pack.encode(arg(state), arg(in_proj), arg(&padded), stride, suffix, row, dim, encoding);
        }
        kernel.encode(
            arg(&padded),
            arg(in_proj),
            arg(w),
            b.as_ref().map(arg),
            arg(&out),
            arg(&state_out),
            case.suffix_len,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            encoding,
        );
    });
    // SAFETY: every command buffer has completed.
    unsafe {
        assert_inputs_unchanged(case, &inputs);
        let outputs = [
            KernelFixture::read_guarded(&padded, s),
            KernelFixture::read_guarded(&out, s),
            KernelFixture::read_guarded(&state_out, s),
        ];
        (outputs, times)
    }
}

/// The Vulkan counterpart of `cpu_decode`. When `split`, each submission dispatches one in-place token, over the
/// token's rows of the same in_proj and out ranges, so `submissions` must be the suffix length.
fn gpu_decode<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    case: &ShortConvCase<T>,
    state_in_place: bool,
    split: bool,
    submissions: usize,
) -> ([Vec<T>; 2], Vec<(Duration, Duration)>) {
    assert!(!split || state_in_place && submissions == case.suffix_len as usize, "split tokens continue in place");
    let s = sentinel::<T>();
    let inputs = gpu_inputs(fixture, case);
    let (in_proj, w, b, state) = &inputs;
    let out = fixture.guarded(&vec![s; (case.suffix_len * case.model_dim) as usize], s);
    let next_state = fixture.guarded(
        match state_in_place {
            true => &case.state,
            false => &case.next_state,
        },
        s,
    );
    let kernel = ShortConvDecodeVulkanKernel::new(
        &fixture.context,
        T::data_type(),
        DataType::F32,
        case.b.is_some(),
        state_in_place,
    )
    .expect("Vulkan ShortConvDecode");
    let mut submission = 0;
    // SAFETY: every range holds the elements its shape indexes, aligned; outputs do not alias other arguments.
    let times = gpu_submissions(fixture, submissions, |encoding| unsafe {
        let tokens = match split {
            true => submission..submission + 1,
            false => 0..case.suffix_len,
        };
        submission += 1;
        let rows = |range: &Range<u64>, row: u32| {
            let row = (row as usize * size_of::<T>()) as u64;
            range.start + u64::from(tokens.start) * row..range.start + u64::from(tokens.end) * row
        };
        kernel.encode(
            (&in_proj.0, rows(&in_proj.1, case.in_proj_stride)),
            arg(w),
            b.as_ref().map(arg),
            (!state_in_place).then(|| arg(state)),
            (&out.0, rows(&out.1, case.model_dim)),
            arg(&next_state),
            tokens.len() as u32,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            encoding,
        );
    });
    // SAFETY: every command buffer has completed.
    unsafe {
        assert_inputs_unchanged(case, &inputs);
        ([KernelFixture::read_guarded(&out, s), KernelFixture::read_guarded(&next_state, s)], times)
    }
}

/// The Vulkan counterpart of `cpu_trie`.
fn gpu_trie<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    case: &ShortConvCase<T>,
    submissions: usize,
) -> ([Vec<T>; 2], Vec<(Duration, Duration)>) {
    let s = sentinel::<T>();
    let inputs = gpu_inputs(fixture, case);
    let (in_proj, w, b, base_state) = &inputs;
    let parents = fixture.guarded(&case.parents, i32::MIN);
    let out = fixture.guarded(&vec![s; (case.suffix_len * case.model_dim) as usize], s);
    let suffix_state = fixture.guarded(&vec![s; case.suffix_len as usize * case.state.len()], s);
    let kernel = ShortConvTrieVulkanKernel::new(&fixture.context, T::data_type(), DataType::F32, case.b.is_some())
        .expect("Vulkan ShortConvTrie");
    // SAFETY: every range holds the elements its shape indexes, aligned; parents precede their nodes; outputs do not
    // alias other arguments.
    let times = gpu_submissions(fixture, submissions, |encoding| unsafe {
        kernel.encode(
            arg(in_proj),
            arg(w),
            b.as_ref().map(arg),
            arg(base_state),
            arg(&parents),
            arg(&out),
            arg(&suffix_state),
            case.suffix_len,
            case.kernel_size,
            case.in_proj_stride,
            case.state_stride,
            case.model_dim,
            encoding,
        );
    });
    // SAFETY: every command buffer has completed.
    unsafe {
        assert_inputs_unchanged(case, &inputs);
        KernelFixture::assert_unchanged(&parents, i32::MIN, &case.parents, "parents");
        ([KernelFixture::read_guarded(&out, s), KernelFixture::read_guarded(&suffix_state, s)], times)
    }
}

/// Bit equality including NaN payloads, for stored and copied state.
pub fn assert_same_bits<T: ArrayElement + Debug>(
    expected: &[T],
    actual: &[T],
    case: &str,
) {
    assert_eq!(expected.len(), actual.len(), "{case}: length");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            bytemuck::bytes_of(expected) == bytemuck::bytes_of(actual),
            "{case}: element {index}: {expected:?} != {actual:?}"
        );
    }
}

/// Pack's padded output: each channel's state, then T(pre_gate · input) of every token.
fn packed<T: Float>(case: &ShortConvCase<T>) -> Vec<T> {
    let (dim, stride, row) = (case.model_dim as usize, case.state_stride as usize, case.in_proj_stride as usize);
    (0..case.padded.len())
        .map(|i| match (i / dim, i % dim) {
            (padded_row, channel) if padded_row < stride => case.state[channel * stride + padded_row],
            (padded_row, channel) => {
                let in_proj = (padded_row - stride) * row + channel;
                T::from(case.in_proj[in_proj].to_f32().unwrap() * case.in_proj[in_proj + 2 * dim].to_f32().unwrap())
                    .unwrap()
            },
        })
        .collect()
}

/// Prefill's state_out: each channel's first kernel_size - 1 elements of `case.next_state` replaced by padded rows
/// suffix_len.., the rest unchanged.
fn copied_state<T: Copy>(
    case: &ShortConvCase<T>,
    padded: &[T],
) -> Vec<T> {
    let (dim, stride, suffix) = (case.model_dim as usize, case.state_stride as usize, case.suffix_len as usize);
    let mut state = case.next_state.clone();
    for channel in 0..dim {
        for tap in 0..case.kernel_size as usize - 1 {
            state[channel * stride + tap] = padded[(suffix + tap) * dim + channel];
        }
    }
    state
}

/// Zero post-gates, zero pre-gates, all-zero weights and exact cancellation by channel: weights alternating ±1 over equal
/// padded, state and x = 1 · sample, with zero bias, so every sum is exactly zero. Needs an even kernel size.
fn zeros_and_cancellation<T: Float>(mut case: ShortConvCase<T>) -> ShortConvCase<T> {
    let (dim, k, stride) = (case.model_dim as usize, case.kernel_size as usize, case.state_stride as usize);
    let (suffix, row) = (case.suffix_len as usize, case.in_proj_stride as usize);
    assert!(k.is_multiple_of(2));
    for channel in 0..dim {
        let sample = case.state[channel * stride];
        match channel % 5 {
            0 => (0..suffix).for_each(|token| case.in_proj[token * row + dim + channel] = T::zero()),
            1 => (0..suffix).for_each(|token| case.in_proj[token * row + channel] = T::zero()),
            2 => case.w[channel * k..(channel + 1) * k].fill(0.0),
            3 => {
                for (tap, w) in case.w[channel * k..(channel + 1) * k].iter_mut().enumerate() {
                    *w = if tap.is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                }
                if let Some(b) = &mut case.b {
                    b[channel] = 0.0;
                }
                case.state[channel * stride..(channel + 1) * stride].fill(sample);
                (0..stride + suffix).for_each(|padded_row| case.padded[padded_row * dim + channel] = sample);
                for token in 0..suffix {
                    case.in_proj[token * row + channel] = T::one();
                    case.in_proj[token * row + 2 * dim + channel] = sample;
                }
            },
            _ => {},
        }
    }
    case
}

/// FP64 reference of every output, asserting the CPU and Vulkan outputs within its error budget. Prefill convolves the
/// stored samples `padded[(token + tap) · model_dim + channel]`. Without `padded`, Decode and Trie convolve, node by
/// node, the parent's stored state (negative: the channel's first kernel_size - 1 `case.state` samples) and the exact
/// x = pre_gate · input, then store the shifted samples and T(x). The budget is γ(n) times the absolute terms times
/// |post_gate|, n being the roundings on the longest weighted path: a weight product, at most K accumulations onto the
/// bias and the gate (K + 2), or for the current x its own product, its weight, its accumulation and the gate (4); plus
/// the rounding to 16-bit storage and half its smallest subnormal. Returns the largest error / budget ratios of the CPU
/// and Vulkan.
fn assert_reference<T: ArrayElement + Float + Debug>(
    case: &ShortConvCase<T>,
    parents: &[i32],
    padded: Option<&[T]>,
    [cpu, gpu]: [&[T]; 2],
    label: &str,
) -> [f64; 2] {
    let (dim, k, stride, suffix) =
        (case.model_dim as usize, case.kernel_size as usize, case.state_stride as usize, case.suffix_len as usize);
    assert!(cpu.len() == suffix * dim && gpu.len() == suffix * dim, "{label}: output length");
    assert!(padded.map_or(parents.len() == suffix, |padded| padded.len() >= (suffix + k - 1) * dim), "{label}: input");
    let n = match padded {
        Some(_) => k + 2,
        None => (k + 2).max(4),
    } as f64
        * f64::from(f32::EPSILON)
        / 2.0;
    let gamma = n / (1.0 - n);
    let unit = match size_of::<T>() {
        4 => 0.0,
        _ => T::epsilon().to_f64().unwrap() / 2.0,
    };
    let subnormal = unit * T::min_positive_value().to_f64().unwrap();
    let value = |value: T| value.to_f64().unwrap();
    let mut worst = [0.0f64; 2];
    for channel in 0..dim {
        let base = case.state[channel * stride..channel * stride + k - 1].to_vec();
        let mut states = Vec::<Vec<T>>::new();
        for node in 0..suffix {
            let row = node * case.in_proj_stride as usize + channel;
            let samples = match padded {
                Some(padded) => (0..k).map(|tap| value(padded[(node + tap) * dim + channel])).collect::<Vec<_>>(),
                None => {
                    let x = value(case.in_proj[row]) * value(case.in_proj[row + 2 * dim]);
                    let source = if parents[node] < 0 {
                        &base
                    } else {
                        &states[parents[node] as usize]
                    };
                    let samples = source.iter().map(|&sample| value(sample)).chain([x]).collect();
                    let stored = (k > 1).then(|| T::from(x as f32).unwrap());
                    let next = source.iter().skip(1).copied().chain(stored).collect();
                    states.push(next);
                    samples
                },
            };
            let bias = case.b.as_ref().map_or(0.0, |b| f64::from(b[channel]));
            let terms = samples.iter().zip(&case.w[channel * k..]).map(|(sample, &w)| f64::from(w) * sample);
            let (sum, magnitude) =
                terms.fold((bias, bias.abs()), |(sum, magnitude), term| (sum + term, magnitude + term.abs()));
            let gate = value(case.in_proj[row + dim]);
            let exact = sum * gate;
            let accumulation = gamma * magnitude * gate.abs();
            let budget = accumulation + unit * (exact.abs() + accumulation) + subnormal;
            for (worst, actual) in worst.iter_mut().zip([cpu, gpu]) {
                let actual = value(actual[node * dim + channel]);
                let error = (actual - exact).abs();
                assert!(
                    error == 0.0 || error <= budget,
                    "{label}: node {node} channel {channel}: FP64 {exact:e}, actual {actual:e}, budget {budget:e}"
                );
                if error != 0.0 {
                    *worst = worst.max(error / budget);
                }
            }
        }
    }
    worst
}

/// Every kernel against the CPU over `SHAPES` and one case of zeros and cancellation, alternating bias and Trie
/// topologies: Pack, Prefill's state copy and stored states bit for bit (Pack and the copy also against their
/// independent expectations), outputs within the KernelFixture bounds and within the FP64 reference budget.
fn matches_cpu<T: ArrayElement + Float + Debug + Default>() {
    let fixture = KernelFixture::new();
    let ty = format!("{:?}", T::data_type());
    let (mut errors, mut worst) = (Vec::new(), [0.0f64; 2]);
    let mut reference =
        |case: &ShortConvCase<T>, parents: &[i32], padded: Option<&[T]>, outputs: [&[T]; 2], label: &str| {
            let ratios = assert_reference(case, parents, padded, outputs, label);
            worst = [worst[0].max(ratios[0]), worst[1].max(ratios[1])];
        };
    let cases = SHAPES.iter().enumerate().map(|(index, &shape)| {
        let mut case = ShortConvCase::<T>::new(shape, index);
        if index % 2 == 1 {
            case = case.bias();
        }
        match case.suffix_len {
            17 => case.parents(&BRANCH),
            suffix if index % 3 == 0 => case.parents(&vec![-1; suffix as usize]),
            _ => case,
        }
    });
    let zeros = zeros_and_cancellation(ShortConvCase::new((33, 4, 17, 3, 106), 11).bias().parents(&BRANCH));
    for case in cases.chain([zeros]) {
        let label = format!("{ty} {}", case.label());
        let chain = (-1..case.suffix_len as i32 - 1).collect::<Vec<_>>();
        for pack in [true, false] {
            let ([cpu_padded, cpu_out, cpu_state], _) = cpu_prefill(&case, pack, 1);
            let ([padded, out, state], _) = gpu_prefill(&fixture, &case, pack, 1);
            let prefill = format!("{label} Prefill pack {pack}");
            assert_same_bits(&cpu_padded, &padded, &format!("{prefill} padded"));
            if pack {
                assert_same_bits(&packed(&case), &padded, &format!("{prefill} Pack"));
            }
            let copied = copied_state(&case, &padded);
            assert_same_bits(&copied, &cpu_state, &format!("{prefill} CPU state_out"));
            assert_same_bits(&copied, &state, &format!("{prefill} state_out"));
            errors.push((format!("Prefill {ty}"), KernelFixture::compare(&cpu_out, &out, &prefill, 1e-5, 1e-6)));
            reference(&case, &chain, Some(&padded), [&cpu_out, &out], &prefill);
        }
        for state_in_place in [true, false] {
            let ([cpu_out, cpu_state], _) = cpu_decode(&case, state_in_place, 1);
            let ([out, state], _) = gpu_decode(&fixture, &case, state_in_place, false, 1);
            let decode = format!("{label} Decode in place {state_in_place}");
            assert_same_bits(&cpu_state, &state, &decode);
            errors.push((format!("Decode {ty}"), KernelFixture::compare(&cpu_out, &out, &decode, 1e-5, 1e-6)));
            reference(&case, &chain, None, [&cpu_out, &out], &decode);
            if state_in_place {
                let suffix = case.suffix_len as usize;
                let ([split_out, split_state], _) = gpu_decode(&fixture, &case, true, true, suffix);
                assert_same_bits(&out, &split_out, &format!("{decode} split out"));
                assert_same_bits(&state, &split_state, &format!("{decode} split state"));
            }
        }
        let ([cpu_out, cpu_state], _) = cpu_trie(&case, 1);
        let ([out, state], _) = gpu_trie(&fixture, &case, 1);
        assert_same_bits(&cpu_state, &state, &format!("{label} Trie suffix_state"));
        errors.push((format!("Trie {ty}"), KernelFixture::compare(&cpu_out, &out, &label, 1e-5, 1e-6)));
        reference(&case, &case.parents, None, [&cpu_out, &out], &format!("{label} Trie"));
    }
    eprintln!("ShortConv {ty}: FP64 reference error / budget at most CPU {:.3}, Vulkan {:.3}", worst[0], worst[1]);
    KernelFixture::report("ShortConv", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_f32() {
    matches_cpu::<f32>();
}

#[uzu_test]
fn matches_cpu_f16() {
    matches_cpu::<f16>();
}

#[uzu_test]
fn matches_cpu_bf16() {
    matches_cpu::<bf16>();
}

/// Signalling and quiet NaNs with payloads, signed zero, subnormals, infinity and one, as stored bits.
pub fn specials<T: ArrayElement>() -> Vec<T> {
    let bits: Vec<u8> = match T::data_type() {
        DataType::F32 => {
            [0x7F80_0001u32, 0xFFA0_0F00, 0x7FC1_2345, 0x8000_0000, 0x0000_0001, 0x807F_FFFF, 0xFF80_0000, 0x3F80_0000]
                .iter()
                .flat_map(|bits| bits.to_ne_bytes())
                .collect()
        },
        DataType::F16 => [0x7C01u16, 0xFD55, 0x7E01, 0x8000, 0x0001, 0x83FF, 0xFC00, 0x3C00]
            .iter()
            .flat_map(|bits| bits.to_ne_bytes())
            .collect(),
        _ => [0x7F81u16, 0xFFA5, 0x7FC1, 0x8000, 0x0001, 0x807F, 0xFF80, 0x3F80]
            .iter()
            .flat_map(|bits| bits.to_ne_bytes())
            .collect(),
    };
    bytemuck::pod_collect_to_vec(&bits)
}

/// Pure copies keep every bit of special values on the CPU and Vulkan: Pack's state rows, Prefill's state_out taps (with
/// the padding past them unchanged) and the Decode and Trie state shifts. Outputs computed from them have the CPU's NaN
/// positions and infinities.
fn copies_bits<T: ArrayElement + Float + Debug + Default>() {
    let fixture = KernelFixture::new();
    let ty = format!("{:?}", T::data_type());
    let mut case = ShortConvCase::<T>::new((33, 4, 3, 5, 99), 9).bias();
    let specials = specials::<T>();
    case.state = specials.iter().cycle().take(case.state.len()).copied().collect();
    case.padded = specials.iter().cycle().skip(3).take(case.padded.len()).copied().collect();
    let mut errors = Vec::new();

    let ([cpu_padded, ..], _) = cpu_prefill(&case, true, 1);
    let ([padded, ..], _) = gpu_prefill(&fixture, &case, true, 1);
    assert_same_bits(&packed(&case), &cpu_padded, &format!("{ty} CPU Pack"));
    assert_same_bits(&packed(&case), &padded, &format!("{ty} Pack"));

    let ([_, cpu_out, cpu_state_out], _) = cpu_prefill(&case, false, 1);
    let ([_, out, state_out], _) = gpu_prefill(&fixture, &case, false, 1);
    let copied = copied_state(&case, &case.padded);
    assert_same_bits(&copied, &cpu_state_out, &format!("{ty} CPU Prefill state_out"));
    assert_same_bits(&copied, &state_out, &format!("{ty} Prefill state_out"));
    errors.push((format!("Prefill {ty}"), KernelFixture::compare(&cpu_out, &out, &ty, 1e-5, 1e-6)));

    for state_in_place in [true, false] {
        let ([cpu_out, cpu_state], _) = cpu_decode(&case, state_in_place, 1);
        let ([out, state], _) = gpu_decode(&fixture, &case, state_in_place, false, 1);
        assert_same_bits(&cpu_state, &state, &format!("{ty} Decode in place {state_in_place} next_state"));
        errors.push((format!("Decode {ty}"), KernelFixture::compare(&cpu_out, &out, &ty, 1e-5, 1e-6)));
    }
    let case = case.parents(&[-1, 0, -1]);
    let ([cpu_out, cpu_state], _) = cpu_trie(&case, 1);
    let ([out, state], _) = gpu_trie(&fixture, &case, 1);
    assert_same_bits(&cpu_state, &state, &format!("{ty} Trie suffix_state"));
    errors.push((format!("Trie {ty}"), KernelFixture::compare(&cpu_out, &out, &ty, 1e-5, 1e-6)));
    KernelFixture::report("ShortConv specials", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn copies_bits_all_types() {
    copies_bits::<f32>();
    copies_bits::<f16>();
    copies_bits::<bf16>();
}

/// Run alone, without sync validation: `... short_conv_test::throughput -- --ignored --nocapture`. At model shapes of
/// 2048 channels and kernel size 4, prints the GPU and wall medians of 10 Vulkan submissions after 3 warm-up ones,
/// with the bytes each must move, and the CPU kernels' wall median. Pack is timed with the Prefill it feeds, as the
/// model records them.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let prefill = ShortConvCase::<T>::new((2048, 4, 128, 3, 6144), 0).bias();
        let decode = ShortConvCase::<T>::new((2048, 4, 1, 3, 6144), 1).bias();
        let trie = ShortConvCase::<T>::new((2048, 4, 16, 3, 6144), 2).bias().parents(&BRANCH[..16]);
        let print = |kernel: &str, case: &ShortConvCase<T>, bytes: usize, gpu: Vec<(Duration, Duration)>, cpu| {
            let median = |mut samples: Vec<Duration>| {
                samples.drain(..3);
                samples.sort();
                samples[samples.len() / 2]
            };
            let (gpu, wall, cpu) = (
                median(gpu.iter().map(|time| time.0).collect()),
                median(gpu.iter().map(|time| time.1).collect()),
                median(cpu),
            );
            let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
            eprintln!(
                "ShortConv{kernel} {:?} {}: {bytes} B; GPU {gpu:?} ({rate:.1} GB/s), wall {wall:?}; CPU wall {cpu:?}",
                T::data_type(),
                case.label()
            );
        };
        let (dim, size, weights) = (2048, size_of::<T>(), 2048 * 5 * size_of::<f32>());
        let (suffix, taps) = (prefill.suffix_len as usize, prefill.kernel_size as usize - 1);
        let pack_bytes = (prefill.state.len() + 2 * suffix * dim + prefill.padded.len()) * size;
        let prefill_bytes = ((suffix + taps) * dim + 2 * suffix * dim + taps * dim) * size + weights;
        let (_, gpu) = gpu_prefill(fixture, &prefill, true, 13);
        let (_, cpu) = cpu_prefill(&prefill, true, 13);
        print("Pack+Prefill", &prefill, pack_bytes + prefill_bytes, gpu, cpu);
        let (_, gpu) = gpu_prefill(fixture, &prefill, false, 13);
        print("Prefill", &prefill, prefill_bytes, gpu, cpu_prefill(&prefill, false, 13).1);
        let (_, gpu) = gpu_decode(fixture, &decode, true, false, 13);
        print(
            "Decode",
            &decode,
            (3 * dim + 2 * taps * dim + dim) * size + weights,
            gpu,
            cpu_decode(&decode, true, 13).1,
        );
        let nodes = trie.suffix_len as usize;
        let trie_bytes = (3 * nodes * dim + taps * dim + nodes * dim + nodes * taps * dim) * size + 4 * nodes + weights;
        print("Trie", &trie, trie_bytes, gpu_trie(fixture, &trie, 13).1, cpu_trie(&trie, 13).1);
    }
    let fixture = KernelFixture::new();
    // Round 0 settles the GPU clocks; compare round 1.
    for round in 0..2 {
        eprintln!("ShortConv throughput round {round}");
        measure::<f32>(&fixture);
        measure::<bf16>(&fixture);
    }
    fixture.assert_clean();
}
