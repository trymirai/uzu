use std::{
    collections::HashSet,
    fmt::Debug,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    NAN, NEG_ZERO, NormalizationCase, POS_INF, POS_ZERO, add, arg, assert_same_bits, bounds, conv1d_values, cpu_buffer,
    cpu_submissions, decay, exp, interval, kernel_fixture::KernelFixture, mean_bounds, mul, oracle, point,
    reciprocal_root_bounds, round32, silu_oracle, staged_rms_bounds, sum_bounds, union,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::ActivationType,
            kernel::{
                DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel, DeltaNetPrefillKernel,
                DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            },
        },
        cpu::Cpu,
        vulkan::{
            DeltaNetConvScanVulkanKernel, DeltaNetConvUpdateVulkanKernel, DeltaNetNormGateVulkanKernel,
            DeltaNetPrefillPrepVulkanKernel, DeltaNetPrefillVulkanKernel, DeltaNetUpdateVulkanKernel, Error, VkBuffer,
            VkCommandBufferEncoding,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// [C, K, Q, R, S, O] (conv_dim, kernel_size, suffix_len, row_stride, state_stride, out_stride) of Scan: a model-like
/// shape with padded rows and output slack; no taps; one tap without state; more state rows than K - 1 and fewer;
/// state rows only; no channels; a channel tail past one workgroup at out_stride = conv_dim; more rows than 65535 groups.
const SCAN_SHAPES: [[u32; 6]; 9] = [
    [40, 4, 5, 53, 3, 53],
    [9, 0, 3, 9, 2, 11],
    [9, 1, 3, 9, 0, 9],
    [6, 3, 4, 8, 5, 7],
    [6, 4, 3, 6, 1, 6],
    [7, 3, 0, 7, 3, 7],
    [0, 4, 3, 4, 3, 4],
    [67, 4, 3, 70, 3, 67],
    [1, 2, 70000, 1, 1, 1],
];

/// [C, K, S] of Update: the smallest model kernel; a model-like row; state slack past K - 1; one channel whose row holds
/// fewer than K - 1 taps, valid as no other channel shares it; no channels.
const UPDATE_SHAPES: [[u32; 3]; 5] = [[40, 2, 1], [40, 4, 3], [67, 4, 6], [1, 4, 1], [0, 4, 3]];

/// [H, D, V, conv_dim, P, Q] (num_v_heads, head_v_dim, value_dim, conv_dim, total_proj_dim, suffix_len) of NormGate: a
/// model-like shape; one element per head; elements striding past the 128-thread workgroup with row slack; no
/// elements, heads or tokens; slack between token rows.
const NORM_SHAPES: [[u32; 6]; 7] = [
    [2, 128, 256, 640, 1028, 3],
    [3, 1, 3, 2, 6, 4],
    [2, 129, 300, 5, 270, 2],
    [2, 0, 4, 1, 4, 3],
    [0, 4, 4, 1, 4, 3],
    [2, 4, 8, 3, 16, 0],
    [2, 4, 11, 3, 16, 2],
];

pub const CPU_FAILURE: &str = "called `Result::unwrap()` on an `Err` value: CommandBufferExecutionFailed(RecvError)";
const UPDATE_KERNEL_SIZE: &str =
    "DeltaNetConvUpdate: precondition kernel_size >= 2 || kernel_size == 1 && conv_dim == 0 violated";
const UPDATE_ROWS: &str = "DeltaNetConvUpdate: precondition conv_dim <= 1 || state_stride >= kernel_size - 1 violated";
const SCAN_ROWS: &str = "DeltaNetConvScan: precondition suffix_len <= 1 || conv_dim <= out_stride violated";
const NORM_ROWS: &str = "DeltaNetNormGate: precondition suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || \
                         num_v_heads <= value_dim / head_v_dim violated";

pub fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

/// The values U(silu(x)) may take under the reviewed SiLU oracle: the exact value where it is one, keeping its class,
/// otherwise its bounds rounded to U.
pub fn silu_set<U: Float>(x: f64) -> ((f64, f64), u8) {
    let ((lo, hi), _) = silu_oracle(x, 1.0);
    match lo.to_bits() == hi.to_bits() {
        true => point(U::from(lo).unwrap().to_f64().unwrap()),
        false => bounds::<U>((lo, hi)),
    }
}

pub fn member(
    ((lo, hi), mask): ((f64, f64), u8),
    actual: f64,
) -> bool {
    match point(actual) {
        ((value, _), 0) => lo <= value && value <= hi,
        (_, class) => mask & class != 0,
    }
}

/// Every element with an owner is a member of its set; every other keeps its initial bits.
pub fn check<T: ArrayElement + Float + Debug>(
    sets: &[((f64, f64), u8)],
    owner: &[Option<usize>],
    initial: &[T],
    actual: &[T],
    label: &str,
) {
    assert_eq!((actual.len(), initial.len()), (owner.len(), owner.len()), "{label}: length");
    let mut violations = 0;
    for (index, ((&value, &initial), &owner)) in actual.iter().zip(initial).zip(owner).enumerate() {
        let valid = match owner {
            Some(slot) => member(sets[slot], value.to_f64().unwrap()),
            None => bytemuck::bytes_of(&value) == bytemuck::bytes_of(&initial),
        };
        violations += usize::from(!valid);
        if !valid && violations <= 5 {
            eprintln!("{label}: element {index}: {value:?} outside {:?}", owner.map(|slot| sets[slot]));
        }
    }
    assert_eq!(violations, 0, "{label}: {violations} elements outside the oracle");
}

/// # Safety
/// Every command buffer using the buffers has completed.
pub unsafe fn assert_inputs<T: ArrayElement + Float>(
    buffers: &[(Arc<VkBuffer>, Range<u64>)],
    inputs: &[&[T]],
    label: &str,
) {
    for (guarded, values) in buffers.iter().zip(inputs) {
        unsafe { KernelFixture::assert_unchanged(guarded, sentinel::<T>(), values, label) };
    }
}

/// Records once, or in `median_times` submissions when `timed`.
pub fn submit(
    fixture: &KernelFixture,
    timed: bool,
    mut record: impl FnMut(&mut VkCommandBufferEncoding),
) -> Option<(Duration, Duration)> {
    match timed {
        true => Some(fixture.median_times(record)),
        false => {
            let mut encoding = fixture.encoding();
            record(&mut encoding);
            KernelFixture::complete(encoding);
            None
        },
    }
}

/// Elements of [conv_padded, conv_weight, bias, in_proj, state_out] Scan reads or writes.
fn scan_extents(
    shape: [u32; 6],
    has_bias: bool,
) -> [usize; 5] {
    let [c, k, q, r, s, o] = shape.map(u64::from);
    let outputs = q != 0 && c != 0;
    let taps = outputs && k != 0;
    let rows = [taps.then(|| (q - 1 + k - 1) * r + c), (s != 0 && c != 0).then(|| (q + s - 1) * r + c)];
    let padded = rows.into_iter().flatten().max().unwrap_or(0);
    let weights = if taps {
        c * k
    } else {
        0
    };
    let bias = if has_bias && outputs {
        c
    } else {
        0
    };
    let in_proj = if outputs {
        (q - 1) * o + c
    } else {
        0
    };
    [padded, weights, bias, in_proj, c * s].map(|len| len as usize)
}

/// Scan's FP32 accumulators in token-major order, separately rounded as the CPU states them.
fn scan_accs(
    shape: [u32; 6],
    has_bias: bool,
    [padded, weight, bias]: &[Vec<f32>; 3],
) -> Vec<f32> {
    let [c, k, q, r, ..] = shape.map(|n| n as usize);
    (0..q * c)
        .map(|i| {
            let (t, channel) = (i / c, i % c);
            let start = if has_bias {
                bias[channel]
            } else {
                0.0
            };
            (0..k).fold(start, |acc, tap| acc + weight[channel * k + tap] * padded[(t + tap) * r + channel])
        })
        .collect()
}

/// The raw padded rows from suffix_len on that Scan copies into state_out.
fn scan_state(
    shape: [u32; 6],
    padded: &[f32],
) -> Vec<f32> {
    let [c, _, q, r, s, _] = shape.map(|n| n as usize);
    (0..c * s).map(|i| padded[(q + i % s) * r + i / s]).collect()
}

/// For every in_proj element, the accumulator of the output written there.
fn scan_owner(
    shape: [u32; 6],
    len: usize,
) -> Vec<Option<usize>> {
    let [c, _, q, _, _, o] = shape.map(|n| n as usize);
    let mut owner = vec![None; len];
    for (slot, (t, channel)) in (0..q).flat_map(|t| (0..c).map(move |channel| (t, channel))).enumerate() {
        owner[t * o + channel] = Some(slot);
    }
    owner
}

/// The CPU Scan on a fresh context in `submissions` submissions: [in_proj, state_out] and the wall times.
fn cpu_scan<T: ArrayElement + Float + Default>(
    shape: [u32; 6],
    has_bias: bool,
    f32s: &[Vec<f32>; 3],
    projection: &[T],
    submissions: usize,
) -> (Vec<T>, Vec<f32>, Vec<Duration>) {
    let [c, k, q, r, s, o] = shape;
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::DeltaNetConvScanKernel::new(&context, T::data_type(), has_bias)
            .expect("CPU DeltaNetConvScan");
    let [padded, weight, bias] = f32s.each_ref().map(|values| cpu_buffer(&context, values));
    let state_len = c as usize * s as usize;
    let mut in_proj = cpu_buffer(&context, projection);
    let mut state = cpu_buffer(&context, &vec![sentinel::<f32>(); state_len]);
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let bias = has_bias.then_some(&bias);
        kernel.encode(&padded, &weight, bias, &mut in_proj, &mut state, q, k, r, s, c, o, command_buffer);
    });
    let in_proj = buffer_prefix_to_vec::<Cpu, T>(&in_proj, projection.len());
    (in_proj, buffer_prefix_to_vec::<Cpu, f32>(&state, state_len), times)
}

/// The Vulkan Scan over guarded ranges of exactly the given payloads, state_out starting as sentinels; asserts the
/// read-only inputs and every guard.
fn gpu_scan<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetConvScanVulkanKernel,
    shape: [u32; 6],
    has_bias: bool,
    f32s: &[Vec<f32>; 3],
    projection: &[T],
    timed: bool,
) -> (Vec<T>, Vec<f32>, Option<(Duration, Duration)>) {
    let [c, k, q, r, s, o] = shape;
    let inputs = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    let in_proj = fixture.guarded(projection, sentinel::<T>());
    let state = fixture.guarded(&vec![sentinel::<f32>(); c as usize * s as usize], sentinel::<f32>());
    // SAFETY: each range holds every element the shape addresses, aligned; the written ranges alias nothing.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let [padded, weight, bias] = inputs.each_ref().map(arg);
        kernel.encode(padded, weight, has_bias.then_some(bias), arg(&in_proj), arg(&state), q, k, r, s, c, o, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&inputs, &f32s.each_ref().map(Vec::as_slice), "Scan input");
        (KernelFixture::read_guarded(&in_proj, sentinel()), KernelFixture::read_guarded(&state, sentinel()), times)
    }
}

/// Scan on the CPU and Vulkan: both outputs in the SiLU oracle of `accs` (the slack keeping its initial bits) and both
/// states equal to `state` bit for bit.
fn scan_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 6],
    has_bias: bool,
    f32s: &[Vec<f32>; 3],
    projection: &[T],
    (accs, state): (&[f32], &[f32]),
    label: &str,
) {
    let kernel = DeltaNetConvScanVulkanKernel::new(&fixture.context, T::data_type(), has_bias).expect("Scan");
    let sets = accs.iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
    let owner = scan_owner(shape, projection.len());
    let (cpu, cpu_state, _) = cpu_scan(shape, has_bias, f32s, projection, 1);
    let (gpu, gpu_state, _) = gpu_scan(fixture, &kernel, shape, has_bias, f32s, projection, false);
    for (side, output, output_state) in [("CPU", &cpu, &cpu_state), ("Vulkan", &gpu, &gpu_state)] {
        check(&sets, &owner, projection, output, &format!("{label} {side} in_proj"));
        assert_same_bits(state, output_state, &format!("{label} {side} state_out"));
    }
}

/// Elements of [conv_weight, bias, in_out, state] Update reads or writes.
fn update_extents(
    [c, k, s]: [u32; 3],
    has_bias: bool,
) -> [usize; 4] {
    let [c, k, s] = [c, k, s].map(|n| n as usize);
    match c {
        0 => [0; 4],
        _ => [
            c * k,
            if has_bias {
                c
            } else {
                0
            },
            c,
            (c - 1) * s + k - 1,
        ],
    }
}

/// Update's FP32 accumulators and next state, as the CPU states them for kernel_size >= 2.
fn update_expected<T: ArrayElement + Float>(
    [c, k, s]: [u32; 3],
    has_bias: bool,
    [weight, bias, state]: &[Vec<f32>; 3],
    in_out: &[T],
) -> (Vec<f32>, Vec<f32>) {
    let [c, k, s] = [c, k, s].map(|n| n as usize);
    let mut next = state.clone();
    let accs = (0..c)
        .map(|channel| {
            let (x, row, taps) = (in_out[channel].to_f32().unwrap(), channel * s, k - 1);
            let start = if has_bias {
                bias[channel]
            } else {
                0.0
            };
            let acc = (0..taps).fold(start, |acc, tap| acc + state[row + tap] * weight[channel * k + tap]);
            next[row..row + taps - 1].copy_from_slice(&state[row + 1..row + taps]);
            next[row + taps - 1] = x;
            acc + x * weight[channel * k + taps]
        })
        .collect();
    (accs, next)
}

/// The CPU Update on a fresh context, one fresh [in_out, state] bundle per submission: the first bundle's results
/// (every bundle does the same work) and the wall times.
fn cpu_update<T: ArrayElement + Float + Default>(
    [c, k, s]: [u32; 3],
    has_bias: bool,
    [weight, bias, state]: &[Vec<f32>; 3],
    in_out: &[T],
    submissions: usize,
) -> (Vec<T>, Vec<f32>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(&context, T::data_type(), has_bias)
            .expect("CPU DeltaNetConvUpdate");
    let [weight, bias] = [weight, bias].map(|values| cpu_buffer(&context, values));
    let mut bundles =
        (0..submissions).map(|_| (cpu_buffer(&context, in_out), cpu_buffer(&context, state))).collect::<Vec<_>>();
    let mut next = bundles.iter_mut();
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let (in_out, state) = next.next().expect("one bundle per submission");
        kernel.encode(&weight, has_bias.then_some(&bias), in_out, state, k, c, s, command_buffer);
    });
    let (first_in_out, first_state) = &bundles[0];
    let results = (
        buffer_prefix_to_vec::<Cpu, T>(first_in_out, in_out.len()),
        buffer_prefix_to_vec::<Cpu, f32>(first_state, state.len()),
    );
    (results.0, results.1, times)
}

/// The Vulkan Update, one fresh guarded [in_out, state] bundle per submission (13 when `timed`): every bundle's
/// results after asserting the read-only inputs and every guard.
fn gpu_update<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetConvUpdateVulkanKernel,
    [c, k, s]: [u32; 3],
    has_bias: bool,
    f32s: &[Vec<f32>; 3],
    in_out: &[T],
    timed: bool,
) -> (Vec<(Vec<T>, Vec<f32>)>, Option<(Duration, Duration)>) {
    let inputs = [&f32s[0], &f32s[1]].map(|values| fixture.guarded(values, sentinel::<f32>()));
    let count = if timed {
        13
    } else {
        1
    };
    let bundles = (0..count)
        .map(|_| (fixture.guarded(in_out, sentinel::<T>()), fixture.guarded(&f32s[2], sentinel::<f32>())))
        .collect::<Vec<_>>();
    let mut next = bundles.iter();
    // SAFETY: each range holds every element the shape addresses, aligned; each submission writes its own bundle.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let (in_out, state) = next.next().expect("one bundle per submission");
        let [weight, bias] = inputs.each_ref().map(arg);
        kernel.encode(weight, has_bias.then_some(bias), arg(in_out), arg(state), k, c, s, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&inputs, &[&f32s[0][..], &f32s[1][..]], "Update input");
        let results = bundles.iter().map(|(in_out, state)| {
            (KernelFixture::read_guarded(in_out, sentinel()), KernelFixture::read_guarded(state, sentinel()))
        });
        (results.collect(), times)
    }
}

/// Update on the CPU and Vulkan: both outputs in the SiLU oracle of `accs` and both states equal to `state` bit for
/// bit.
fn update_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 3],
    has_bias: bool,
    f32s: &[Vec<f32>; 3],
    in_out: &[T],
    (accs, state): (&[f32], &[f32]),
    label: &str,
) {
    let kernel = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, T::data_type(), has_bias).expect("Update");
    let sets = accs.iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
    let owner = (0..in_out.len()).map(Some).collect::<Vec<_>>();
    let (cpu, cpu_state, _) = cpu_update(shape, has_bias, f32s, in_out, 1);
    let (mut gpu, _) = gpu_update(fixture, &kernel, shape, has_bias, f32s, in_out, false);
    let (gpu, gpu_state) = gpu.remove(0);
    for (side, output, output_state) in [("CPU", &cpu, &cpu_state), ("Vulkan", &gpu, &gpu_state)] {
        check(&sets, &owner, in_out, output, &format!("{label} {side} in_out"));
        assert_same_bits(state, output_state, &format!("{label} {side} state"));
    }
}

/// Elements of [in_out, in_proj, norm_weight] NormGate reads or writes.
fn norm_extents(shape: [u32; 6]) -> [usize; 3] {
    let [h, d, v, conv, p, q] = shape.map(|n| n as usize);
    match q * h * d {
        0 => [0; 3],
        _ => [(q - 1) * v + (h - 1) * d + d, (q - 1) * p + conv + (h - 1) * d + d, d],
    }
}

/// For every in_out element, its index among the head rows in (token, head, element) order.
fn norm_owner(
    shape: [u32; 6],
    len: usize,
) -> Vec<Option<usize>> {
    let [h, d, v, _, _, q] = shape.map(|n| n as usize);
    let mut owner = vec![None; len];
    for slot in 0..q * h * d {
        let (row, i) = (slot / d, slot % d);
        owner[row / h * v + row % h * d + i] = Some(slot);
    }
    owner
}

/// The values every NormGate output may take, in `norm_owner` order, as three FP32 product stages: the unscaled
/// `staged_rms_bounds`' FP32 endpoints or class of o inv, times the raw norm_weight (its sign kept, no offset added),
/// times the FP32 SiLU oracle of z, only that last product rounded to T.
fn norm_sets<T: ArrayElement + Float>(
    shape: [u32; 6],
    epsilon: f32,
    [in_out, in_proj]: [&[T]; 2],
    weight: &[f32],
) -> Vec<((f64, f64), u8)> {
    let [h, d, v, conv, p, q] = shape.map(|n| n as usize);
    if q * h * d == 0 {
        return Vec::new();
    }
    let position = |slot: usize, base: usize, stride: usize| {
        let (row, i) = (slot / d, slot % d);
        row / h * stride + base + row % h * d + i
    };
    let mut case = NormalizationCase::<T, f32>::new((q * h) as u32, d as u32, 0);
    case.input = (0..q * h * d).map(|slot| in_out[position(slot, 0, v)]).collect();
    (case.scales, case.epsilon) = (None, epsilon);
    let staged = staged_rms_bounds::<T, f32, f32>(&case, 128);
    staged
        .into_iter()
        .enumerate()
        .map(|(slot, ((lo, hi), center))| {
            let normed = match () {
                _ if center.is_nan() => point(f64::NAN),
                _ if lo.to_bits() == hi.to_bits() => point(lo),
                _ if lo == hi => union(point(-0.0), point(0.0)),
                _ => bounds::<f32>((lo, hi)),
            };
            let scaled = mul::<f32>(normed, point(f64::from(weight[slot % d])));
            mul::<T>(scaled, silu_set::<f32>(in_proj[position(slot, conv, p)].to_f64().unwrap()))
        })
        .collect()
}

/// The CPU NormGate on a fresh context, one fresh in_out per submission: the first one's result and the wall times.
fn cpu_norm<T: ArrayElement + Float + Default>(
    shape: [u32; 6],
    epsilon: f32,
    [in_out, in_proj]: [&[T]; 2],
    weight: &[f32],
    submissions: usize,
) -> (Vec<T>, Vec<Duration>) {
    let [h, d, v, conv, p, q] = shape;
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, T::data_type())
        .expect("CPU DeltaNetNormGate");
    let (gates, weights) = (cpu_buffer(&context, in_proj), cpu_buffer(&context, weight));
    let mut bundles = (0..submissions).map(|_| cpu_buffer(&context, in_out)).collect::<Vec<_>>();
    let mut next = bundles.iter_mut();
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let output = next.next().expect("one in_out per submission");
        kernel.encode(output, &gates, &weights, h, d, v, conv, p, epsilon, q, command_buffer);
    });
    (buffer_prefix_to_vec::<Cpu, T>(&bundles[0], in_out.len()), times)
}

/// The Vulkan NormGate, one fresh guarded in_out per submission (13 when `timed`): every result after asserting the
/// read-only inputs and every guard.
fn gpu_norm<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetNormGateVulkanKernel,
    shape: [u32; 6],
    epsilon: f32,
    [in_out, in_proj]: [&[T]; 2],
    weight: &[f32],
    timed: bool,
) -> (Vec<Vec<T>>, Option<(Duration, Duration)>) {
    let [h, d, v, conv, p, q] = shape;
    let (gates, weights) = (fixture.guarded(in_proj, sentinel::<T>()), fixture.guarded(weight, sentinel::<f32>()));
    let bundles = (0..if timed {
        13
    } else {
        1
    })
        .map(|_| fixture.guarded(in_out, sentinel::<T>()))
        .collect::<Vec<_>>();
    let mut next = bundles.iter();
    // SAFETY: each range holds every element the shape addresses, aligned; each submission writes its own in_out.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let output = next.next().expect("one in_out per submission");
        kernel.encode(arg(output), arg(&gates), arg(&weights), h, d, v, conv, p, epsilon, q, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&gates, sentinel::<T>(), in_proj, "NormGate in_proj");
        KernelFixture::assert_unchanged(&weights, sentinel::<f32>(), weight, "NormGate norm_weight");
        (bundles.iter().map(|output| KernelFixture::read_guarded(output, sentinel())).collect(), times)
    }
}

/// NormGate on the CPU and Vulkan: every head element a member of `sets`, every other in_out element unchanged.
fn norm_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 6],
    epsilon: f32,
    inputs: [&[T]; 2],
    weight: &[f32],
    sets: &[((f64, f64), u8)],
    label: &str,
) {
    let kernel = DeltaNetNormGateVulkanKernel::new(&fixture.context, T::data_type()).expect("NormGate");
    let owner = norm_owner(shape, inputs[0].len());
    let (cpu, _) = cpu_norm(shape, epsilon, inputs, weight, 1);
    let (gpu, _) = gpu_norm(fixture, &kernel, shape, epsilon, inputs, weight, false);
    check(sets, &owner, inputs[0], &cpu, &format!("{label} CPU"));
    check(sets, &owner, inputs[0], &gpu[0], &format!("{label} Vulkan"));
}

fn scan_matches_oracle<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for shape in SCAN_SHAPES {
        for has_bias in [false, true] {
            let [padded, weight, bias, in_proj, _] = scan_extents(shape, has_bias);
            let f32s = [conv1d_values(padded, 0, 5), conv1d_values(weight, 1, 5), conv1d_values(bias, 2, 0)];
            let projection = conv1d_values::<T>(in_proj, 3, 7);
            let expected = (&scan_accs(shape, has_bias, &f32s)[..], &scan_state(shape, &f32s[0])[..]);
            let label = format!("Scan {:?} {shape:?} bias {has_bias}", T::data_type());
            scan_check(fixture, shape, has_bias, &f32s, &projection, expected, &label);
        }
    }
}

/// Every Scan shape with and without a bias over exactly its addressed extents: outputs within the SiLU oracle on the
/// CPU and Vulkan, the in_proj slack and every input and guard unchanged, state_out the padded rows bit for bit.
#[uzu_test]
fn conv_scan_matches_oracle() {
    let fixture = KernelFixture::new();
    scan_matches_oracle::<f32>(&fixture);
    scan_matches_oracle::<bf16>(&fixture);
    fixture.assert_clean();
}

fn update_matches_oracle<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for shape in UPDATE_SHAPES {
        for has_bias in [false, true] {
            let [weight, bias, in_out, state] = update_extents(shape, has_bias);
            let f32s = [conv1d_values(weight, 1, 5), conv1d_values(bias, 2, 0), conv1d_values(state, 3, 3)];
            let in_out = conv1d_values::<T>(in_out, 0, 5);
            let (accs, next) = update_expected(shape, has_bias, &f32s, &in_out);
            let label = format!("Update {:?} {shape:?} bias {has_bias}", T::data_type());
            update_check(fixture, shape, has_bias, &f32s, &in_out, (&accs, &next), &label);
        }
    }
}

/// Every Update shape with and without a bias: outputs within the SiLU oracle on the CPU and Vulkan, the shifted state
/// with the widened input last and every slack tap bit for bit, every input and guard unchanged.
#[uzu_test]
fn conv_update_matches_oracle() {
    let fixture = KernelFixture::new();
    update_matches_oracle::<f32>(&fixture);
    update_matches_oracle::<bf16>(&fixture);
    fixture.assert_clean();
}

fn chain<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (c, k, s, q) = (40usize, 4u32, 3usize, 5usize);
    let [weight, bias, state] = [(c * 4, 6), (c, 7), (c * s, 5)].map(|(len, seed)| conv1d_values::<f32>(len, seed, 0));
    let tokens = conv1d_values::<T>(q * c, 4, 0);
    let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, T::data_type(), true).expect("Update");
    let (mut gpu_state, mut cpu_state) = (state.clone(), state.clone());
    let (mut gpu_outputs, mut cpu_outputs) = (Vec::new(), Vec::new());
    for token in tokens.chunks(c) {
        let shape = [c as u32, k, s as u32];
        let (mut gpu, _) =
            gpu_update(fixture, &update, shape, true, &[weight.clone(), bias.clone(), gpu_state], token, false);
        let (output, next) = gpu.remove(0);
        (gpu_outputs, gpu_state) = ([gpu_outputs, output].concat(), next);
        let (output, next, _) = cpu_update(shape, true, &[weight.clone(), bias.clone(), cpu_state], token, 1);
        (cpu_outputs, cpu_state) = ([cpu_outputs, output].concat(), next);
    }
    // Conv1dPack's layout: the state's taps as rows, then the widened tokens.
    let history = (0..s * c).map(|i| state[i % c * s + i / c]);
    let padded = history.chain(tokens.iter().map(|value| value.to_f32().unwrap())).collect::<Vec<_>>();
    let shape = [c as u32, k, q as u32, c as u32, s as u32, c as u32];
    let f32s = [padded, weight, bias];
    let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, T::data_type(), true).expect("Scan");
    let projection = vec![sentinel::<T>(); q * c];
    let (scan_outputs, scan_state, _) = gpu_scan(fixture, &scan, shape, true, &f32s, &projection, false);
    let label = format!("{:?} chain", T::data_type());
    KernelFixture::assert_bits(&scan_outputs, &gpu_outputs, &format!("{label}: Vulkan Scan against Update outputs"));
    assert_same_bits(&scan_state, &gpu_state, &format!("{label}: Vulkan Scan against Update state"));
    assert_same_bits(&cpu_state, &gpu_state, &format!("{label}: CPU against Vulkan state"));
    let sets = scan_accs(shape, true, &f32s).iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
    let owner = (0..q * c).map(Some).collect::<Vec<_>>();
    check(&sets, &owner, &projection, &cpu_outputs, &format!("{label}: CPU Update outputs"));
    let (cpu_scan_outputs, cpu_scan_state, _) = cpu_scan(shape, true, &f32s, &projection, 1);
    check(&sets, &owner, &projection, &cpu_scan_outputs, &format!("{label}: CPU Scan outputs"));
    assert_same_bits(&cpu_scan_state, &gpu_state, &format!("{label}: CPU Scan state"));
}

/// Five chained Updates and one Scan over the same five tokens, with the history as Conv1dPack lays it out: the same
/// accumulators in the same order, so the Vulkan outputs agree bit for bit up to NaN payloads and the final states
/// exactly, and the CPU's are within the oracle and exact.
#[uzu_test]
fn update_chain_equals_scan() {
    let fixture = KernelFixture::new();
    chain::<f32>(&fixture);
    chain::<bf16>(&fixture);
    fixture.assert_clean();
}

fn conv_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (big, step, tiny, small) = (2f32.powi(24), 1.0 + 2f32.powi(-7), 2f32.powi(-17), 2f32.powi(-65));
    let to_t = |values: &[f32]| values.iter().map(|&value| T::from(value).unwrap()).collect::<Vec<_>>();
    let ty = format!("{:?}", T::data_type());
    // [shape, bias, padded, weight, bias values, accumulators]: products 2^24, 1, -2^24 cancel to +0 only ascending;
    // -(1 + 2^-7 + 2^-17) then (1 + 2^-7)(1 + 2^-17), which rounds to its magnitude, give +0 only unfused; the bias
    // first absorbs 1 before -2^24 cancels it; +0 starts and -0 biases keep their signs; 2^-130 stays subnormal.
    let scans: [([u32; 6], bool, Vec<f32>, Vec<f32>, Vec<f32>, f32); 8] = [
        ([1, 3, 1, 1, 0, 1], false, vec![1.0; 3], vec![big, 1.0, -big], vec![], 0.0),
        ([1, 2, 1, 1, 0, 1], false, vec![1.0 + 2f32.powi(-7) + tiny, 1.0 + tiny], vec![-1.0, step], vec![], 0.0),
        ([1, 2, 1, 1, 0, 1], true, vec![1.0, 1.0], vec![1.0, -big], vec![big], 0.0),
        ([1, 1, 1, 1, 0, 1], false, vec![-0.0], vec![1.0], vec![], 0.0),
        ([1, 1, 1, 1, 0, 1], true, vec![-0.0], vec![1.0], vec![-0.0], -0.0),
        ([1, 0, 1, 1, 0, 1], true, vec![], vec![], vec![-0.0], -0.0),
        ([1, 0, 1, 1, 0, 1], false, vec![], vec![], vec![], 0.0),
        ([1, 1, 1, 1, 0, 1], false, vec![small], vec![small], vec![], small * small),
    ];
    for (index, (shape, has_bias, padded, weight, bias, acc)) in scans.into_iter().enumerate() {
        let label = format!("{ty} Scan witness {index}");
        let f32s = [padded, weight, bias];
        scan_check(fixture, shape, has_bias, &f32s, &[sentinel::<T>()], (&[acc], &[]), &label);
    }
    // State only: every one of the state_stride rows copied raw, past kernel_size - 1, with nothing productive read.
    let raw = [1.5, -0.0, f32::from_bits(0x7FC1_2345), f32::from_bits(1), -3.25];
    let f32s = [raw.to_vec(), vec![], vec![]];
    scan_check::<T>(fixture, [1, 3, 0, 1, 5, 1], false, &f32s, &[], (&[], &raw), &format!("{ty} Scan state only"));

    // [shape, bias, weight, bias values, state, x, accumulator, next state]: the same order, fusion, bias and sign
    // witnesses through the state taps and x; zero weights shifting [1, 2, 4] with x = 8 to [2, 4, 8], which a descending
    // shift would leave [4, 4, 8]; a subnormal product.
    let updates: [([u32; 3], bool, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, Vec<f32>); 7] = [
        ([1, 3, 2], false, vec![big, 1.0, -big], vec![], vec![1.0, 1.0], 1.0, 0.0, vec![1.0, 1.0]),
        ([1, 2, 1], false, vec![-1.0, 1.0 + tiny], vec![], vec![1.0 + 2f32.powi(-7) + tiny], step, 0.0, vec![step]),
        ([1, 2, 1], true, vec![1.0, -big], vec![big], vec![1.0], 1.0, 0.0, vec![1.0]),
        ([1, 2, 1], false, vec![1.0, 1.0], vec![], vec![-0.0], -0.0, 0.0, vec![-0.0]),
        ([1, 2, 1], true, vec![1.0, 1.0], vec![-0.0], vec![-0.0], -0.0, -0.0, vec![-0.0]),
        ([1, 4, 3], false, vec![0.0; 4], vec![], vec![1.0, 2.0, 4.0], 8.0, 0.0, vec![2.0, 4.0, 8.0]),
        ([1, 2, 1], false, vec![small, 0.0], vec![], vec![small], 0.0, small * small, vec![0.0]),
    ];
    for (index, (shape, has_bias, weight, bias, state, x, acc, next)) in updates.into_iter().enumerate() {
        let label = format!("{ty} Update witness {index}");
        update_check(fixture, shape, has_bias, &[weight, bias, state], &to_t(&[x]), (&[acc], &next), &label);
    }
}

/// Exact witnesses derived here, independently of the CPU code, which the CPU and Vulkan must both produce: each
/// accumulator is exact and, being at most 2^-26 in magnitude, so is its SiLU, a halving that keeps the sign.
#[uzu_test]
fn conv_exact_witnesses() {
    let fixture = KernelFixture::new();
    conv_witnesses::<f32>(&fixture);
    conv_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

fn norm_matches_oracle<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for shape in NORM_SHAPES {
        let [in_out, in_proj, weight] = norm_extents(shape);
        let (in_out, in_proj) = (conv1d_values::<T>(in_out, 0, 0), conv1d_values::<T>(in_proj, 1, 0));
        let weight = conv1d_values::<f32>(weight, 2, 0);
        let sets = norm_sets(shape, 1e-5, [&in_out, &in_proj], &weight);
        let label = format!("NormGate {:?} {shape:?}", T::data_type());
        norm_check(fixture, shape, 1e-5, [&in_out, &in_proj], &weight, &sets, &label);
    }
}

/// Every NormGate shape over exactly its addressed extents: head elements within the composed RMS and SiLU oracle on
/// the CPU and Vulkan, every other in_out element, in_proj, the weights and every guard unchanged.
#[uzu_test]
fn norm_gate_matches_oracle() {
    let fixture = KernelFixture::new();
    norm_matches_oracle::<f32>(&fixture);
    norm_matches_oracle::<bf16>(&fixture);
    fixture.assert_clean();
}

fn norm_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (inf, tiny) = (f32::INFINITY, 2f32.powi(-27));
    let (subnormal, huge) = (f32::from_bits(0x0001_0000), f32::from_bits(0x7F00_0000));
    let shape = [1, 1, 1, 0, 1, 1];
    // [o, z, norm_weight, epsilon, expected]: (o inv) w with o = 2^-133 squaring to 0 and inv about 4 is exactly 2^-4
    // whatever inv's error, times silu(2^-27) = 2^-28 exactly, where o (inv w) would overflow to +inf; the zero row;
    // 0 inf with epsilon 0; the square root of -1; a square overflowing to +inf, whose inverse is +0; the gate's classes;
    // a -0 norm_weight, whose sign the products keep where adding an offset to it would give +0.
    let witnesses = [
        (1.0, tiny, -0.0, 0.0, point(-0.0)),
        (subnormal, tiny, huge, 0.0625, point(2f64.powi(-32))),
        (0.0, 1.0, 1.0, 0.0625, point(0.0)),
        (0.0, 1.0, 1.0, 0.0, point(f64::NAN)),
        (0.0, 1.0, 1.0, -1.0, point(f64::NAN)),
        (2f32.powi(64), 1.0, 1.0, 0.0625, point(0.0)),
        (1.0, f32::NAN, 1.0, 0.0625, point(f64::NAN)),
        (1.0, inf, 1.0, 0.0625, point(f64::INFINITY)),
        (1.0, -inf, 1.0, 0.0625, point(f64::NAN)),
    ];
    let ty = format!("{:?}", T::data_type());
    for (index, (o, z, weight, epsilon, expected)) in witnesses.into_iter().enumerate() {
        let inputs = [[T::from(o).unwrap()], [T::from(z).unwrap()]];
        norm_check(fixture, shape, epsilon, [&inputs[0], &inputs[1]], &[weight], &[expected], &format!("{ty} {index}"));
    }
    // The oracle itself keeps the -0 norm_weight's sign.
    let inputs = [[T::one()], [T::from(tiny).unwrap()]];
    assert_eq!(norm_sets(shape, 0.0, [&inputs[0], &inputs[1]], &[-0.0]), [point(-0.0)], "{ty}: -0 weight oracle");
    // Epsilon before the square root: 1 / sqrt(1 + 3) is 0.5 within the root's bound, giving about 2^-29, where
    // 1 / (sqrt(1) + 3) would give 2^-30.
    let inputs = [[T::one()], [T::from(tiny).unwrap()]];
    let sets = norm_sets(shape, 3.0, [&inputs[0], &inputs[1]], &[1.0]);
    assert!(!member(sets[0], 2f64.powi(-30)) && member(sets[0], 2f64.powi(-29)), "{ty}: epsilon oracle {sets:?}");
    norm_check(fixture, shape, 3.0, [&inputs[0], &inputs[1]], &[1.0], &sets, &format!("{ty} epsilon"));
}

/// Exact NormGate witnesses for both storage types: the product order, epsilon inside the root, zero and overflowing
/// rows and the gate's NaN and infinities, each a single class or value on the CPU and Vulkan.
#[uzu_test]
fn norm_gate_witnesses() {
    let fixture = KernelFixture::new();
    norm_witnesses::<f32>(&fixture);
    norm_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// The expected panic message of a closure that must fail before recording.
pub fn panics(run: impl FnOnce()) -> String {
    let error = catch_unwind(AssertUnwindSafe(run)).expect_err("accepted");
    *error.downcast::<String>().expect("panic message")
}

/// The DSL's presence check on Scan and Update, an API invariant of the generated bindings rather than of the CPU
/// kernels: a missing active bias and a present inactive one fail its assert_eq before anything is recorded, also
/// without work, while an active bias without work may be an empty range.
#[uzu_test]
fn optional_bias_presence() {
    let fixture = KernelFixture::new();
    let buffers = [16, 16, 4, 16, 16].map(|len| fixture.guarded(&vec![sentinel::<f32>(); len], sentinel::<f32>()));
    let empty = fixture.guarded::<f32>(&[], sentinel());
    let presence = |kernel: &str, has_bias: bool| {
        let message = format!("{kernel}: argument 'bias' must be present exactly when has_bias");
        format!("assertion `left == right` failed: {message}\n  left: {}\n right: {has_bias}", !has_bias)
    };
    for (has_bias, q) in [(true, 2), (true, 0), (false, 2), (false, 0)] {
        let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, DataType::F32, has_bias).expect("Scan");
        let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, DataType::F32, has_bias).expect("Update");
        let [padded, weight, bias, in_out, state] = buffers.each_ref().map(arg);
        let wrong = (!has_bias).then_some(bias.clone());
        let mut encoding = fixture.encoding();
        // SAFETY: the presence check panics before anything is recorded.
        let message = panics(|| unsafe {
            let (padded, weight, in_out, state) = (padded.clone(), weight.clone(), in_out.clone(), state.clone());
            scan.encode(padded, weight, wrong.clone(), in_out, state, q, 2, 4, 1, 4, 4, &mut encoding)
        });
        assert_eq!(message, presence("DeltaNetConvScan", has_bias), "Scan bias {has_bias}, Q {q}");
        let channels = if q == 0 {
            0
        } else {
            4
        };
        // SAFETY: as above.
        let message = panics(|| unsafe { update.encode(weight, wrong, in_out, state, 2, channels, 1, &mut encoding) });
        assert_eq!(message, presence("DeltaNetConvUpdate", has_bias), "Update bias {has_bias}, C {channels}");
        KernelFixture::complete(encoding);
    }
    let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, DataType::F32, true).expect("Scan");
    let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, DataType::F32, true).expect("Update");
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    // SAFETY: without work nothing is indexed or recorded.
    unsafe {
        scan.encode(e(), e(), Some(e()), e(), e(), 0, 2, 4, 0, 4, 4, &mut encoding);
        update.encode(e(), Some(e()), e(), e(), 2, 0, 1, &mut encoding);
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for (buffer, len) in buffers.iter().zip([16, 16, 4, 16, 16]) {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &vec![sentinel::<f32>(); len], "buffer");
        }
        KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty");
    }
    fixture.assert_clean();
}

/// Update's kernel_size precondition, where the CPU itself fails (as the failed wait of its own fresh context), and the
/// three ownership preconditions, where the CPU would complete sequentially but rows of different invocations overlap:
/// each fails before anything is recorded. Only F32 and BF16 exist.
#[uzu_test]
fn rejects_invalid_configurations() {
    let fixture = KernelFixture::new();
    let buffers = [64, 64, 64, 64].map(|len| fixture.guarded(&vec![sentinel::<f32>(); len], sentinel::<f32>()));
    let [a, b, c, d] = buffers.each_ref().map(arg);
    let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, DataType::F32, false).expect("Update");
    for ([channels, k, s], message) in [
        ([0, 0, 1], UPDATE_KERNEL_SIZE),
        ([4, 0, 1], UPDATE_KERNEL_SIZE),
        ([4, 1, 1], UPDATE_KERNEL_SIZE),
        ([2, 4, 2], UPDATE_ROWS),
    ] {
        let mut encoding = fixture.encoding();
        // SAFETY: the precondition panics before anything is recorded.
        let actual =
            panics(|| unsafe { update.encode(a.clone(), None, b.clone(), c.clone(), k, channels, s, &mut encoding) });
        assert_eq!(actual, message, "Update {channels} {k} {s}");
        KernelFixture::complete(encoding);
        if message == UPDATE_KERNEL_SIZE {
            let f32s = [vec![1.0; 16], vec![], vec![1.0; 16]];
            let cpu = panics(|| drop(cpu_update([channels, k, s], false, &f32s, &[1.0f32; 4][..channels as usize], 1)));
            assert_eq!(cpu, CPU_FAILURE, "CPU Update {channels} {k} {s}");
        }
    }
    let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, DataType::F32, false).expect("Scan");
    let mut encoding = fixture.encoding();
    // SAFETY: as above.
    let actual = panics(|| unsafe {
        scan.encode(a.clone(), b.clone(), None, c.clone(), d.clone(), 2, 1, 4, 0, 4, 3, &mut encoding)
    });
    assert_eq!(actual, SCAN_ROWS);
    let norm = DeltaNetNormGateVulkanKernel::new(&fixture.context, DataType::F32).expect("NormGate");
    // SAFETY: as above.
    let actual =
        panics(|| unsafe { norm.encode(a.clone(), b.clone(), c.clone(), 2, 4, 6, 0, 16, 1e-5, 2, &mut encoding) });
    assert_eq!(actual, NORM_ROWS);
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for buffer in &buffers {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &[sentinel::<f32>(); 64], "buffer");
        }
    }
    let context = &fixture.context;
    let f16 = DataType::F16;
    let variant = |kernel: &str, error: Option<Error>| match error {
        Some(Error::KernelVariant {
            kernel: name,
            ..
        }) => assert_eq!(name, kernel),
        _ => panic!("{kernel}: F16 accepted"),
    };
    variant("DeltaNetConvScan", DeltaNetConvScanVulkanKernel::new(context, f16, false).err());
    variant("DeltaNetConvUpdate", DeltaNetConvUpdateVulkanKernel::new(context, f16, false).err());
    variant("DeltaNetNormGate", DeltaNetNormGateVulkanKernel::new(context, f16).err());
    fixture.assert_clean();
}

/// No work at u32::MAX scalars and empty ranges: Scan without channels or rows, Update without channels (also with
/// kernel_size 1, which the CPU completes too), NormGate without tokens, heads or elements. Nothing is recorded or
/// changed.
#[uzu_test]
fn zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (empty, m) = (fixture.guarded::<f32>(&[], sentinel()), u32::MAX);
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    for has_bias in [false, true] {
        let bias = has_bias.then(e);
        let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, DataType::F32, has_bias).expect("Scan");
        let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, DataType::F32, has_bias).expect("Update");
        // SAFETY: without work nothing is indexed or recorded.
        unsafe {
            scan.encode(e(), e(), bias.clone(), e(), e(), m, m, m, m, 0, m, &mut encoding);
            scan.encode(e(), e(), bias.clone(), e(), e(), 0, m, m, 0, m, m, &mut encoding);
            update.encode(e(), bias.clone(), e(), e(), 4, 0, m, &mut encoding);
            update.encode(e(), bias, e(), e(), 1, 0, m, &mut encoding);
        }
    }
    let norm = DeltaNetNormGateVulkanKernel::new(&fixture.context, DataType::F32).expect("NormGate");
    for [h, d, q] in [[m, m, 0], [0, m, m], [m, 0, m]] {
        // SAFETY: as above.
        unsafe { norm.encode(e(), e(), e(), h, d, m, m, m, 1e-5, q, &mut encoding) };
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty") };
    let (output, state, _) = cpu_update::<f32>([0, 1, 0], false, &[vec![], vec![], vec![]], &[], 1);
    assert!(output.is_empty() && state.is_empty(), "CPU Update kernel_size 1 without channels");
    fixture.assert_clean();
}

fn measure<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    lengths: &[u32],
) {
    let (c, k, s, h, d, conv, p) = (10240u32, 4u32, 3u32, 48u32, 128u32, 10240u32, 16480u32);
    let (v, size) = (h * d, size_of::<T>() as u64);
    let ty = format!("{:?}", T::data_type());
    let median = |mut samples: Vec<Duration>| {
        samples.drain(..3);
        samples.sort();
        samples[samples.len() / 2]
    };
    let print = |kernel: &str, q: u32, bytes: u64, (gpu, wall): (Duration, Duration), cpu: Duration| {
        let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
        eprintln!(
            "DeltaNet{kernel} {ty} Q {q}: {bytes} B logical ({rate:.1} GB/s effective); GPU {gpu:?}, wall {wall:?}; CPU wall {cpu:?}"
        );
    };
    let shape = [c, k, s];
    let [weight, bias, in_out, state] = update_extents(shape, true);
    let f32s = [conv1d_values(weight, 1, 0), conv1d_values(bias, 2, 0), conv1d_values(state, 3, 0)];
    let in_out = conv1d_values::<T>(in_out, 0, 0);
    let (accs, next) = update_expected(shape, true, &f32s, &in_out);
    let sets = accs.iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
    let owner = (0..in_out.len()).map(Some).collect::<Vec<_>>();
    let update = DeltaNetConvUpdateVulkanKernel::new(&fixture.context, T::data_type(), true).expect("Update");
    let (results, times) = gpu_update(fixture, &update, shape, true, &f32s, &in_out, true);
    for (index, (output, output_state)) in results.iter().enumerate() {
        check(&sets, &owner, &in_out, output, &format!("{ty} Update bundle {index}"));
        assert_same_bits(&next, output_state, &format!("{ty} Update bundle {index} state"));
    }
    let (c64, k64, s64) = (u64::from(c), u64::from(k), u64::from(s));
    let bytes = c64 * (2 * size + 4 * (k64 - 1) + 4 * k64 + 4 + 8 * (k64 - 2) + 4);
    let (cpu_output, cpu_state, cpu_times) = cpu_update(shape, true, &f32s, &in_out, 13);
    check(&sets, &owner, &in_out, &cpu_output, &format!("{ty} CPU Update"));
    assert_same_bits(&next, &cpu_state, &format!("{ty} CPU Update state"));
    print("ConvUpdate", 1, bytes, times.expect("timed"), median(cpu_times));
    let scan = DeltaNetConvScanVulkanKernel::new(&fixture.context, T::data_type(), true).expect("Scan");
    let norm = DeltaNetNormGateVulkanKernel::new(&fixture.context, T::data_type()).expect("NormGate");
    for &q in lengths {
        let shape = [c, k, q, p, s, p];
        let [padded, weight, bias, in_proj, _] = scan_extents(shape, true);
        let f32s = [conv1d_values(padded, 0, 0), conv1d_values(weight, 1, 0), conv1d_values(bias, 2, 0)];
        let projection = conv1d_values::<T>(in_proj, 3, 0);
        let sets = scan_accs(shape, true, &f32s).iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
        let (owner, state) = (scan_owner(shape, projection.len()), scan_state(shape, &f32s[0]));
        let (once, once_state, _) = gpu_scan(fixture, &scan, shape, true, &f32s, &projection, false);
        check(&sets, &owner, &projection, &once, &format!("{ty} Scan Q {q} before timing"));
        assert_same_bits(&state, &once_state, &format!("{ty} Scan Q {q} state before timing"));
        let (output, output_state, times) = gpu_scan(fixture, &scan, shape, true, &f32s, &projection, true);
        check(&sets, &owner, &projection, &output, &format!("{ty} Scan Q {q} after timing"));
        assert_same_bits(&state, &output_state, &format!("{ty} Scan Q {q} state after timing"));
        let q64 = u64::from(q);
        let bytes = q64 * c64 * (8 * k64 + 4 + size) + 8 * c64 * s64;
        let (cpu_output, cpu_state, cpu_times) = cpu_scan(shape, true, &f32s, &projection, 13);
        check(&sets, &owner, &projection, &cpu_output, &format!("{ty} CPU Scan Q {q}"));
        assert_same_bits(&state, &cpu_state, &format!("{ty} CPU Scan Q {q} state"));
        print("ConvScan", q, bytes, times.expect("timed"), median(cpu_times));

        let shape = [h, d, v, conv, p, q];
        let [in_out, in_proj, weight] = norm_extents(shape);
        let (in_out, in_proj) = (conv1d_values::<T>(in_out, 0, 0), conv1d_values::<T>(in_proj, 1, 0));
        let weight = conv1d_values::<f32>(weight, 2, 0);
        let sets = norm_sets(shape, 1e-6, [&in_out, &in_proj], &weight);
        let owner = norm_owner(shape, in_out.len());
        let (results, times) = gpu_norm(fixture, &norm, shape, 1e-6, [&in_out, &in_proj], &weight, true);
        for (index, output) in results.iter().enumerate() {
            check(&sets, &owner, &in_out, output, &format!("{ty} NormGate Q {q} bundle {index}"));
        }
        let bytes = q64 * u64::from(h) * u64::from(d) * (4 * size + 4);
        let (cpu_output, cpu_times) = cpu_norm(shape, 1e-6, [&in_out, &in_proj], &weight, 13);
        check(&sets, &owner, &in_out, &cpu_output, &format!("{ty} CPU NormGate Q {q}"));
        print("NormGate", q, bytes, times.expect("timed"), median(cpu_times));
    }
}

/// Run alone, without sync validation: `... delta_net_test::throughput -- --ignored --nocapture`. A synthetic family at
/// the common test's Qwen3.5-labelled shapes (48 value heads, 16 key heads, head dimensions 128, so conv_dim 10240 and
/// total_proj_dim 16480), kernel size 4, not verified model files: Update for one token, Scan and NormGate for 64 and
/// 1024, in two rounds of opposite order. Prints the GPU and wall medians of 10 Vulkan submissions after 3 warm-up ones
/// and the CPU kernels' wall medians. Update and NormGate write their inputs, so every submission gets its own fresh
/// bundle, each checked after completion; Scan's outputs are checked before and after timing. The bytes are the logical
/// loads and stores each kernel executes, not measured memory bandwidth.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for (round, lengths) in [[64, 1024], [1024, 64]].iter().enumerate() {
        eprintln!("DeltaNet throughput round {round}");
        measure::<f32>(&fixture, lengths);
        measure::<bf16>(&fixture, lengths);
    }
    fixture.assert_clean();
}

/// [H, G, Dv, key_dim, value_dim] of the recurrence Update: two v heads sharing one k head; the model's G k heads of
/// 128 with grouped v heads; Dv 129 rows striding past the workgroup; q and k rows overlapping where key_dim is below
/// G 128, which the CPU reads alike; key and value slack.
const RECURRENCE_UPDATE_SHAPES: [[u32; 5]; 5] =
    [[2, 1, 4, 128, 8], [4, 2, 3, 256, 12], [1, 1, 129, 128, 129], [2, 2, 2, 100, 4], [3, 3, 5, 400, 17]];

/// [H, G, Dv, key_dim, value_dim, Q] of Prefill: 1024 productive tokens; 1024 tokens without v heads; H 3 over G 2,
/// whose floor(H / G) = 1 maps the v heads to k heads 0 to 2, which key_dim holds; the remainder H 5 over G 2 with value
/// slack; Dv 129 with grouped heads; one token.
const RECURRENCE_PREFILL_SHAPES: [[u32; 6]; 6] = [
    [1, 1, 1, 128, 1, 1024],
    [0, 1, 4, 128, 0, 1024],
    [3, 2, 5, 384, 15, 3],
    [5, 2, 2, 384, 12, 2],
    [4, 2, 129, 256, 516, 2],
    [2, 2, 3, 256, 8, 1],
];

/// [H, G, key_dim, value_dim, Q] of Prep: grouped heads; the remainder H 5 over G 2, whose lane 4 stays untouched; no v
/// heads; fewer v heads than k heads, which write no lane; compact V past one workgroup; compact V in fewer blocks than
/// k heads, with key slack.
const RECURRENCE_PREP_SHAPES: [[u32; 5]; 6] = [
    [4, 2, 256, 6, 3],
    [5, 2, 256, 4, 2],
    [0, 2, 256, 3, 2],
    [1, 2, 256, 5, 2],
    [2, 1, 128, 300, 1],
    [3, 3, 400, 130, 2],
];

/// Update's RMS epsilon; the q and k normalizations add the kernels' own 1e-6.
const RECURRENCE_EPSILON: f32 = 1e-5;

/// Elements of [in_proj, a_log, dt_bias, norm_weight, state, out] Update reads or writes, through the last one it
/// accesses: none without v heads or rows.
fn recurrence_update_extents(shape: [u32; 5]) -> [usize; 6] {
    let [h, g, d, k, v] = shape.map(|n| n as usize);
    match h * d {
        0 => [0; 6],
        _ => [(k + g * 128).max(2 * k + v + h * d).max(2 * (k + v + h)), h, h, d, h * d * 128, h * d],
    }
}

/// Elements of [q_norm/k_norm, beta/decay, in_proj, state, out] Prefill reads or writes, through the last one: the k
/// heads up to (H - 1) / floor(H / G); none without tokens, v heads or rows.
fn recurrence_prefill_extents(shape: [u32; 6]) -> [usize; 5] {
    let [h, g, d, k, v, q] = shape.map(|n| n as usize);
    match q * h * d {
        0 => [0; 5],
        _ => [
            (q - 1) * k + ((h - 1) / (h / g) + 1) * 128,
            q * h,
            (q - 1) * 2 * (k + v + h) + 2 * k + h * d,
            h * d * 128,
            (q - 1) * v + h * d,
        ],
    }
}

/// Elements of [in_proj, a_log/dt_bias, q_norm/k_norm_out, compact_v_out, beta/decay_out] Prep reads or writes, through
/// the last one: compact V only with write_compact_v, the gate region and lanes only for G floor(H / G) > 0 lanes; none
/// without tokens.
fn recurrence_prep_extents(
    shape: [u32; 5],
    compact: bool,
) -> [usize; 5] {
    let [h, g, k, v, q] = shape.map(|n| n as usize);
    if q == 0 {
        return [0; 5];
    }
    let lanes = g * (h / g);
    let reads =
        [k + g * 128, usize::from(compact && v > 0) * (2 * k + v), usize::from(lanes > 0) * (2 * (k + v) + h + lanes)];
    let written = usize::from(lanes > 0) * ((q - 1) * h + lanes);
    [
        (q - 1) * 2 * (k + v + h) + reads.into_iter().max().unwrap(),
        lanes,
        (q - 1) * k + g * 128,
        usize::from(compact) * q * v,
        written,
    ]
}

/// Update's in_proj and [a_log, dt_bias, norm_weight, state]: finite eighths, a_log in [-2, 0) so the decays spread
/// over (0, 1).
fn recurrence_update_inputs<T: ArrayElement + Float>(shape: [u32; 5]) -> (Vec<T>, [Vec<f32>; 4]) {
    let [proj, a_log, dt_bias, weight, state, _] = recurrence_update_extents(shape);
    let a_log = conv1d_values::<f32>(a_log, 1, 0).iter().map(|x| x / 4.0 - 1.0).collect();
    (
        conv1d_values(proj, 0, 0),
        [a_log, conv1d_values(dt_bias, 2, 0), conv1d_values(weight, 3, 0), conv1d_values(state, 4, 0)],
    )
}

/// Prefill's in_proj and [q_norm, k_norm, beta, decay, state]: eighths, q scaled by 1/8, k by 1/64 so long recurrences
/// stay finite, beta and decay by 1/4; the specials at every `every`-th element when it is nonzero.
fn recurrence_prefill_inputs<T: ArrayElement + Float>(
    shape: [u32; 6],
    every: usize,
) -> (Vec<T>, [Vec<f32>; 5]) {
    let [qk, lanes, proj, state, _] = recurrence_prefill_extents(shape);
    let scaled = |len: usize, seed: usize, scale: f32| -> Vec<f32> {
        conv1d_values::<f32>(len, seed, every).iter().map(|x| x / scale).collect()
    };
    let f32s = [
        scaled(qk, 0, 8.0),
        scaled(qk, 1, 64.0),
        scaled(lanes, 2, 4.0),
        scaled(lanes, 3, 4.0),
        conv1d_values(state, 4, every),
    ];
    (conv1d_values(proj, 5, every), f32s)
}

/// Prep's in_proj, the specials at every `every`-th element when it is nonzero, and [a_log in [-2, 0), dt_bias].
fn recurrence_prep_inputs<T: ArrayElement + Float>(
    shape: [u32; 5],
    compact: bool,
    every: usize,
) -> (Vec<T>, [Vec<f32>; 2]) {
    let [proj, lanes, ..] = recurrence_prep_extents(shape, compact);
    let a_log = conv1d_values::<f32>(lanes, 1, 0).iter().map(|x| x / 4.0 - 1.0).collect();
    (conv1d_values(proj, 0, every), [a_log, conv1d_values(lanes, 2, 0)])
}

/// 1 / sqrt(Σ x² + 1e-6) of a q or k row: the square sum is exact FP32 in order on the CPU and the shader, so the shifted
/// sum is one value; NaN gives NaN, +inf gives +0, and any other value, at least 1e-6 and normal, the root's 2 ULPs.
fn recurrence_inverse(row: &[f32]) -> ((f64, f64), u8) {
    let shifted = row.iter().map(|x| x * x).sum::<f32>() + 1e-6;
    let root = 1.0 / f64::from(shifted).sqrt();
    match shifted {
        _ if shifted.is_nan() => point(f64::NAN),
        f32::INFINITY => point(0.0),
        _ => bounds::<f32>(interval(root, root, 2.0)),
    }
}

/// The normalized q (x inv) qs, qs the CPU's FP32 1 / sqrt(128), and k x inv of one head's rows as FP32 products, the
/// last rounded to U.
fn recurrence_qk<U: Float>(
    q: &[f32],
    k: &[f32],
) -> [Vec<((f64, f64), u8)>; 2] {
    let ([q_inv, k_inv], qs) = ([q, k].map(recurrence_inverse), point(f64::from(1.0 / 128f32.sqrt())));
    [
        q.iter().map(|&x| mul::<U>(mul::<f32>(point(f64::from(x)), q_inv), qs)).collect(),
        k.iter().map(|&x| mul::<U>(point(f64::from(x)), k_inv)).collect(),
    ]
}

/// [β, dt] of one v head: β = 1 / (1 + e^-β_raw), the SiLU oracle of 1 with slope β_raw; dt = e^a_log softplus(sum) with
/// sum = a_raw + dt_bias, one exact FP32 sum, e^a_log within Vulkan's exp bound, which holds the CPU's expf, and the
/// shader's flush of a subnormal result to +0.
fn recurrence_gates(
    beta_raw: f32,
    a_log: f32,
    sum: f32,
) -> [((f64, f64), u8); 2] {
    let (lo, hi) = exp(f64::from(a_log)).0;
    let e = match hi > 0.0 && lo < f64::from(f32::MIN_POSITIVE) {
        true => union(bounds::<f32>((lo, hi)), point(0.0)),
        false => bounds::<f32>((lo, hi)),
    };
    let softplus = bounds::<f32>(oracle(f64::from(sum), ActivationType::SOFTPLUS).0);
    [bounds::<f32>(silu_oracle(1.0, beta_raw).0), mul::<f32>(e, softplus)]
}

/// e^-dt by the canonical decay oracle, which iterates over dt's FP32 members: their ordinal span is asserted within the
/// test data's 2^12 before it runs, and the largest one kept in `span` for the report.
fn recurrence_decay(
    dt: ((f64, f64), u8),
    span: &mut i64,
) -> ((f64, f64), u8) {
    let ((lo, hi), _) = dt;
    if lo <= hi {
        let members = KernelFixture::ordinal(hi as f32) - KernelFixture::ordinal(lo as f32);
        assert!(members <= 1 << 12, "dt {dt:?} spans {members} FP32 values");
        *span = (*span).max(members);
    }
    decay::<f32, f32>(dt, true)
}

/// The reciprocal RMS over every member of `o`, staged as `staged_rms_bounds`: each square's finite or zero members as a
/// nonnegative term of the canonical any-order `sum_bounds` over a workgroup of 128, its `mean_bounds`, the exact FP32
/// epsilon sum and `reciprocal_root_bounds`. NaN squares add NaN, infinite ones a +inf sum and so +0.
fn recurrence_rms(
    o: &[((f64, f64), u8)],
    epsilon: f32,
) -> ((f64, f64), u8) {
    let squares = o.iter().map(|&value| mul::<f32>(value, value)).collect::<Vec<_>>();
    let (mask, zeros) = (squares.iter().fold(0, |mask, square| mask | square.1), NEG_ZERO | POS_ZERO);
    let mut set = ((f64::INFINITY, f64::NEG_INFINITY), mask & NAN);
    if mask & POS_INF != 0 {
        set = union(set, point(0.0));
    }
    if squares.iter().all(|&((lo, hi), classes)| lo <= hi || classes & zeros != 0) {
        let terms = squares.iter().map(|&((a, b), classes)| match a <= b {
            true => (f64::from(u8::from(classes & zeros == 0)) * a.max(0.0), b),
            false => (0.0, 0.0),
        });
        let (lo, hi) = sum_bounds(&terms.collect::<Vec<_>>(), 128);
        let mean = mean_bounds((lo.max(0.0), hi), f64::from(o.len() as f32));
        let shifted = [mean.0, mean.1].map(|variance| round32(variance + f64::from(epsilon)));
        let roots = [reciprocal_root_bounds(shifted[1]).0, reciprocal_root_bounds(shifted[0]).1];
        set = union(set, bounds::<f32>((roots[0], roots[1])));
    }
    set
}

/// Update's sets of every [state, out] element in buffer order, as the CPU and the shader stage them: per v head the
/// normalized q and k, kq from -0 as the CPU's iterator sum, β and the decay of `recurrence_gates`; per row sq and sk
/// from +0, retrieved = decay sk, delta = β (v - retrieved), o = decay sq + delta kq and s <- decay s + k delta; the
/// output T(((o inv) w) silu(z)) with `recurrence_rms`. Every product and sum is an FP32 set operation in the CPU's
/// order; the largest dt span goes to `span`.
fn recurrence_update_sets<T: ArrayElement + Float>(
    shape: [u32; 5],
    in_proj: &[T],
    [a_log, dt_bias, weight, state]: &[Vec<f32>; 4],
    span: &mut i64,
) -> [Vec<((f64, f64), u8)>; 2] {
    let [h, g, d, k, v] = shape.map(|n| n as usize);
    let x = |i: usize| in_proj[i].to_f32().unwrap();
    let (mut next, mut out) = (Vec::new(), Vec::new());
    for hv in 0..h * d.min(1) {
        let hk = hv / (h / g);
        let rows = [hk * 128, k + hk * 128].map(|base| (base..base + 128).map(x).collect::<Vec<_>>());
        let [qn, kn] = recurrence_qk::<f32>(&rows[0], &rows[1]);
        let kq = kn.iter().zip(&qn).fold(point(-0.0), |acc, (&a, &b)| add::<f32>(acc, mul::<f32>(a, b)));
        let gates = 2 * (k + v);
        let [beta, dt] = recurrence_gates(x(gates + hv), a_log[hv], x(gates + h + hv) + dt_bias[hv]);
        let decay = recurrence_decay(dt, span);
        let mut outputs = Vec::new();
        for i in 0..d {
            let row = &state[(hv * d + i) * 128..][..128];
            let dot = |factors: &[((f64, f64), u8)]| {
                let terms = row.iter().zip(factors);
                terms.fold(point(0.0), |acc, (&s, &factor)| add::<f32>(acc, mul::<f32>(point(f64::from(s)), factor)))
            };
            let retrieved = mul::<f32>(point(-1.0), mul::<f32>(decay, dot(&kn)));
            let delta = mul::<f32>(beta, add::<f32>(point(f64::from(x(2 * k + hv * d + i))), retrieved));
            outputs.push(add::<f32>(mul::<f32>(decay, dot(&qn)), mul::<f32>(delta, kq)));
            next.extend(
                row.iter()
                    .zip(&kn)
                    .map(|(&s, &kj)| add::<f32>(mul::<f32>(decay, point(f64::from(s))), mul::<f32>(kj, delta))),
            );
        }
        let inv = recurrence_rms(&outputs, RECURRENCE_EPSILON);
        for (i, &o) in outputs.iter().enumerate() {
            let gate = silu_set::<f32>(f64::from(x(2 * k + v + hv * d + i)));
            out.push(mul::<T>(mul::<f32>(mul::<f32>(o, inv), point(f64::from(weight[i]))), gate));
        }
    }
    [next, out]
}

/// Prep's sets and owners over [q_norm_out, k_norm_out, beta_out, decay_out], in buffer order: the normalized q and k
/// rounded to QKT, and β with the log decay -(dt) or its decay e^-dt as Update stages them for every lane below
/// G floor(H / G); the q and k slack between tokens and the remaining lanes own none. The largest dt span goes to
/// `span`.
fn recurrence_prep_sets<T: ArrayElement + Float, QKT: Float>(
    shape: [u32; 5],
    log: bool,
    in_proj: &[T],
    [a_log, dt_bias]: &[Vec<f32>; 2],
    span: &mut i64,
) -> [(Vec<((f64, f64), u8)>, Vec<Option<usize>>); 4] {
    let [_, _, qk, _, lanes] = recurrence_prep_extents(shape, false);
    let [h, g, k, v, q] = shape.map(|n| n as usize);
    let x = |i: usize| in_proj[i].to_f32().unwrap();
    let mut buffers = [qk, qk, lanes, lanes].map(|len| (Vec::new(), vec![None; len]));
    for (token, hk) in itertools::iproduct!(0..q, 0..g) {
        let row = token * 2 * (k + v + h);
        let rows = [row + hk * 128, row + k + hk * 128].map(|base| (base..base + 128).map(x).collect::<Vec<_>>());
        let [q_sets, k_sets] = recurrence_qk::<QKT>(&rows[0], &rows[1]);
        let mut entries = (0..128).map(|j| (0, token * k + hk * 128 + j, [q_sets[j], k_sets[j]])).collect::<Vec<_>>();
        let gates = row + 2 * (k + v);
        for hv in hk * (h / g)..(hk + 1) * (h / g) {
            let [beta, dt] = recurrence_gates(x(gates + hv), a_log[hv], x(gates + h + hv) + dt_bias[hv]);
            let decay = match log {
                true => mul::<f32>(point(-1.0), dt),
                false => recurrence_decay(dt, span),
            };
            entries.push((2, token * h + hv, [beta, decay]));
        }
        for (first, at, pair) in entries {
            for (index, set) in pair.into_iter().enumerate() {
                let (sets, owner) = &mut buffers[first + index];
                owner[at] = Some(sets.len());
                sets.push(set);
            }
        }
    }
    buffers
}

/// The largest relative width of the sets' finite intervals, reported with each check.
fn widest(sets: &[((f64, f64), u8)]) -> f64 {
    let finite = sets.iter().filter(|((lo, hi), _)| lo < hi);
    finite.map(|((lo, hi), _)| (hi - lo) / lo.abs().max(hi.abs())).fold(0.0, f64::max)
}

/// Every FP32 value of a set's finite interval, ascending: the enumeration of the concrete witnesses only.
fn members(((lo, hi), _): ((f64, f64), u8)) -> Vec<f32> {
    let values = std::iter::successors((lo <= hi).then_some(lo as f32), |&value| Some(value.next_up()));
    values.take_while(|&value| f64::from(value) <= hi).collect()
}

/// Whether no value is a member of both sets: a fault's exclusion from the correct oracle.
fn disjoint(
    ((a, b), m): ((f64, f64), u8),
    ((c, d), n): ((f64, f64), u8),
) -> bool {
    m & n == 0 && (a > b || c > d || b < c || d < a)
}

/// The CPU Update on a fresh context, one fresh [state, out] bundle per submission: the first bundle's results (every
/// bundle does the same work) and the wall times.
fn cpu_recurrence_update<T: ArrayElement + Float + Default>(
    shape: [u32; 5],
    head_k_dim: u32,
    in_proj: &[T],
    [a_log, dt_bias, weight, state]: &[Vec<f32>; 4],
    submissions: usize,
) -> (Vec<f32>, Vec<T>, Vec<Duration>) {
    let ([h, g, d, k, v], out_len) = (shape, recurrence_update_extents(shape)[5]);
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, T::data_type(), head_k_dim)
            .expect("CPU DeltaNetUpdate");
    let projection = cpu_buffer(&context, in_proj);
    let [a_log, dt_bias, weight] = [a_log, dt_bias, weight].map(|values| cpu_buffer(&context, values));
    let mut bundles = (0..submissions)
        .map(|_| (cpu_buffer(&context, state), cpu_buffer(&context, &vec![sentinel::<T>(); out_len])))
        .collect::<Vec<_>>();
    let mut next = bundles.iter_mut();
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let (state, out) = next.next().expect("one bundle per submission");
        kernel.encode(
            &projection,
            &a_log,
            &dt_bias,
            &weight,
            state,
            out,
            h,
            g,
            d,
            k,
            v,
            RECURRENCE_EPSILON,
            command_buffer,
        );
    });
    let (first_state, first_out) = &bundles[0];
    (
        buffer_prefix_to_vec::<Cpu, f32>(first_state, state.len()),
        buffer_prefix_to_vec::<Cpu, T>(first_out, out_len),
        times,
    )
}

/// The Vulkan Update over guarded ranges, one fresh guarded [state, out] bundle per submission (13 when `timed`): every
/// bundle's results after asserting the read-only inputs and every guard.
fn gpu_recurrence_update<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetUpdateVulkanKernel,
    shape: [u32; 5],
    in_proj: &[T],
    f32s: &[Vec<f32>; 4],
    timed: bool,
) -> (Vec<(Vec<f32>, Vec<T>)>, Option<(Duration, Duration)>) {
    let ([h, g, d, k, v], out_len) = (shape, recurrence_update_extents(shape)[5]);
    let projection = fixture.guarded(in_proj, sentinel::<T>());
    let inputs = [&f32s[0], &f32s[1], &f32s[2]].map(|values| fixture.guarded(values, sentinel::<f32>()));
    let bundles = (0..if timed {
        13
    } else {
        1
    })
        .map(|_| {
            (
                fixture.guarded(&f32s[3], sentinel::<f32>()),
                fixture.guarded(&vec![sentinel::<T>(); out_len], sentinel::<T>()),
            )
        })
        .collect::<Vec<_>>();
    let mut next = bundles.iter();
    // SAFETY: each range holds every element the shape addresses, aligned; each submission writes its own bundle.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let (state, out) = next.next().expect("one bundle per submission");
        let [a_log, dt_bias, weight] = inputs.each_ref().map(arg);
        let (projection, state, out) = (arg(&projection), arg(state), arg(out));
        kernel.encode(projection, a_log, dt_bias, weight, state, out, h, g, d, k, v, RECURRENCE_EPSILON, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&projection, sentinel::<T>(), in_proj, "Update in_proj");
        assert_inputs(&inputs, &[&f32s[0][..], &f32s[1][..], &f32s[2][..]], "Update input");
        let results = bundles.iter().map(|(state, out)| {
            (KernelFixture::read_guarded(state, sentinel()), KernelFixture::read_guarded(out, sentinel()))
        });
        (results.collect(), times)
    }
}

/// Update on the CPU in `submissions` submissions and on Vulkan, timed from 2: every state and output element of every
/// bundle a member of its set, every input and guard unchanged. Returns the CPU's and the first Vulkan bundle's states
/// and the times.
fn recurrence_update_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 5],
    in_proj: &[T],
    f32s: &[Vec<f32>; 4],
    submissions: usize,
    label: &str,
) -> ([Vec<f32>; 2], Option<(Duration, Duration)>, Vec<Duration>) {
    let kernel = DeltaNetUpdateVulkanKernel::new(&fixture.context, T::data_type(), 128).expect("Update");
    let mut span = 0;
    let [state_sets, out_sets] = recurrence_update_sets(shape, in_proj, f32s, &mut span);
    let (cpu_state, cpu_out, cpu_times) = cpu_recurrence_update(shape, 128, in_proj, f32s, submissions);
    let (mut results, times) = gpu_recurrence_update(fixture, &kernel, shape, in_proj, f32s, submissions > 1);
    results.insert(0, (cpu_state, cpu_out));
    let owner = |len: usize| (0..len).map(Some).collect::<Vec<_>>();
    for (index, (state, out)) in results.iter().enumerate() {
        let side = match index {
            0 => format!("{label} CPU"),
            _ => format!("{label} Vulkan bundle {index}"),
        };
        check(&state_sets, &owner(state.len()), &f32s[3], state, &format!("{side} state"));
        check(&out_sets, &owner(out.len()), &vec![sentinel::<T>(); out.len()], out, &format!("{side} out"));
    }
    let widths = [&state_sets, &out_sets].map(|sets| widest(sets));
    eprintln!("{label}: widest relative sets state {:.2e}, out {:.2e}; widest dt span {span}", widths[0], widths[1]);
    let mut states = results.into_iter().map(|(state, _)| state);
    ([states.next().unwrap(), states.next().unwrap()], times, cpu_times)
}

/// The CPU Prefill on a fresh context, one fresh [state, out] bundle per submission: the first bundle's results and the
/// wall times.
fn cpu_recurrence_prefill<T: ArrayElement + Float + Default>(
    shape: [u32; 6],
    head_k_dim: u32,
    in_proj: &[T],
    [q_norm, k_norm, beta, decay, state]: &[Vec<f32>; 5],
    submissions: usize,
) -> (Vec<f32>, Vec<T>, Vec<Duration>) {
    let ([h, g, d, k, v, q], out_len) = (shape, recurrence_prefill_extents(shape)[4]);
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(&context, T::data_type(), head_k_dim)
            .expect("CPU DeltaNetPrefill");
    let [q_norm, k_norm, beta, decay] = [q_norm, k_norm, beta, decay].map(|values| cpu_buffer(&context, values));
    let projection = cpu_buffer(&context, in_proj);
    let mut bundles = (0..submissions)
        .map(|_| (cpu_buffer(&context, state), cpu_buffer(&context, &vec![sentinel::<T>(); out_len])))
        .collect::<Vec<_>>();
    let mut next = bundles.iter_mut();
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let (state, out) = next.next().expect("one bundle per submission");
        let groups = d.div_ceil(16);
        kernel.encode(
            &q_norm,
            &k_norm,
            &beta,
            &decay,
            &projection,
            state,
            out,
            h,
            g,
            d,
            k,
            v,
            q,
            groups,
            command_buffer,
        );
    });
    let (first_state, first_out) = &bundles[0];
    (
        buffer_prefix_to_vec::<Cpu, f32>(first_state, state.len()),
        buffer_prefix_to_vec::<Cpu, T>(first_out, out_len),
        times,
    )
}

/// The Vulkan Prefill over guarded ranges, one fresh guarded [state, out] bundle per submission (13 when `timed`):
/// every bundle's results after asserting the read-only inputs and every guard.
fn gpu_recurrence_prefill<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetPrefillVulkanKernel,
    shape: [u32; 6],
    in_proj: &[T],
    f32s: &[Vec<f32>; 5],
    timed: bool,
) -> (Vec<(Vec<f32>, Vec<T>)>, Option<(Duration, Duration)>) {
    let ([h, g, d, k, v, q], out_len) = (shape, recurrence_prefill_extents(shape)[4]);
    let projection = fixture.guarded(in_proj, sentinel::<T>());
    let inputs = [&f32s[0], &f32s[1], &f32s[2], &f32s[3]].map(|values| fixture.guarded(values, sentinel::<f32>()));
    let bundles = (0..if timed {
        13
    } else {
        1
    })
        .map(|_| {
            (
                fixture.guarded(&f32s[4], sentinel::<f32>()),
                fixture.guarded(&vec![sentinel::<T>(); out_len], sentinel::<T>()),
            )
        })
        .collect::<Vec<_>>();
    let mut next = bundles.iter();
    // SAFETY: each range holds every element the shape addresses, aligned; each submission writes its own bundle.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let (state, out) = next.next().expect("one bundle per submission");
        let [q_norm, k_norm, beta, decay] = inputs.each_ref().map(arg);
        let (projection, state, out, groups) = (arg(&projection), arg(state), arg(out), d.div_ceil(16));
        kernel.encode(q_norm, k_norm, beta, decay, projection, state, out, h, g, d, k, v, q, groups, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&projection, sentinel::<T>(), in_proj, "Prefill in_proj");
        assert_inputs(&inputs, &[&f32s[0][..], &f32s[1][..], &f32s[2][..], &f32s[3][..]], "Prefill input");
        let results = bundles.iter().map(|(state, out)| {
            (KernelFixture::read_guarded(state, sentinel()), KernelFixture::read_guarded(out, sentinel()))
        });
        (results.collect(), times)
    }
}

/// Prefill on the CPU in `submissions` submissions and on Vulkan, timed from 2: every bundle's state and output equal to
/// the CPU's bit for bit up to NaN payloads, the slack keeping its sentinels. Returns the first Vulkan bundle and the
/// times.
fn recurrence_prefill_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 6],
    in_proj: &[T],
    f32s: &[Vec<f32>; 5],
    submissions: usize,
    label: &str,
) -> ((Vec<f32>, Vec<T>), Option<(Duration, Duration)>, Vec<Duration>) {
    let kernel = DeltaNetPrefillVulkanKernel::new(&fixture.context, T::data_type(), 128).expect("Prefill");
    let (cpu_state, cpu_out, cpu_times) = cpu_recurrence_prefill(shape, 128, in_proj, f32s, submissions);
    let (mut results, times) = gpu_recurrence_prefill(fixture, &kernel, shape, in_proj, f32s, submissions > 1);
    for (index, (state, out)) in results.iter().enumerate() {
        KernelFixture::assert_bits(&cpu_state, state, &format!("{label} bundle {index} state"));
        KernelFixture::assert_bits(&cpu_out, out, &format!("{label} bundle {index} out"));
    }
    (results.swap_remove(0), times, cpu_times)
}

/// The CPU Prep on a fresh context in `submissions` submissions: [q_norm_out, k_norm_out], compact V,
/// [beta_out, decay_out] and the wall times.
fn cpu_recurrence_prep<T: ArrayElement + Float + Default, QKT: ArrayElement + Float + Default>(
    shape: [u32; 5],
    head_k_dim: u32,
    [log, compact]: [bool; 2],
    in_proj: &[T],
    [a_log, dt_bias]: &[Vec<f32>; 2],
    submissions: usize,
) -> ([Vec<QKT>; 2], Vec<T>, [Vec<f32>; 2], Vec<Duration>) {
    let ([h, g, k, v, q], [_, _, qk, compact_len, lanes]) = (shape, recurrence_prep_extents(shape, compact));
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        T::data_type(),
        QKT::data_type(),
        head_k_dim,
        log,
        compact,
    )
    .expect("CPU DeltaNetPrefillPrep");
    let projection = cpu_buffer(&context, in_proj);
    let [a_log, dt_bias] = [a_log, dt_bias].map(|values| cpu_buffer(&context, values));
    let [mut q_out, mut k_out] = [0; 2].map(|_| cpu_buffer(&context, &vec![sentinel::<QKT>(); qk]));
    let mut compact_v = cpu_buffer(&context, &vec![sentinel::<T>(); compact_len]);
    let [mut beta, mut decay] = [0; 2].map(|_| cpu_buffer(&context, &vec![sentinel::<f32>(); lanes]));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let compact_v = compact.then_some(&mut compact_v);
        kernel.encode(
            &projection,
            &a_log,
            &dt_bias,
            &mut q_out,
            &mut k_out,
            compact_v,
            &mut beta,
            &mut decay,
            h,
            g,
            k,
            v,
            q,
            command_buffer,
        );
    });
    (
        [&q_out, &k_out].map(|buffer| buffer_prefix_to_vec::<Cpu, QKT>(buffer, qk)),
        buffer_prefix_to_vec::<Cpu, T>(&compact_v, compact_len),
        [&beta, &decay].map(|buffer| buffer_prefix_to_vec::<Cpu, f32>(buffer, lanes)),
        times,
    )
}

/// The Vulkan Prep over guarded ranges, its outputs starting as sentinels, once or in timed submissions:
/// [q_norm_out, k_norm_out], compact V and [beta_out, decay_out] after asserting the inputs and every guard.
fn gpu_recurrence_prep<T: ArrayElement + Float, QKT: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &DeltaNetPrefillPrepVulkanKernel,
    shape: [u32; 5],
    compact: bool,
    in_proj: &[T],
    f32s: &[Vec<f32>; 2],
    timed: bool,
) -> ([Vec<QKT>; 2], Vec<T>, [Vec<f32>; 2], Option<(Duration, Duration)>) {
    let ([h, g, k, v, q], [_, _, qk, compact_len, lanes]) = (shape, recurrence_prep_extents(shape, compact));
    let projection = fixture.guarded(in_proj, sentinel::<T>());
    let inputs = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    let normalized = [0; 2].map(|_| fixture.guarded(&vec![sentinel::<QKT>(); qk], sentinel::<QKT>()));
    let compact_v = fixture.guarded(&vec![sentinel::<T>(); compact_len], sentinel::<T>());
    let gates = [0; 2].map(|_| fixture.guarded(&vec![sentinel::<f32>(); lanes], sentinel::<f32>()));
    // SAFETY: each range holds every element the shape addresses, aligned; the written ranges alias nothing.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let ([a_log, dt_bias], [q_out, k_out], [beta, decay]) =
            (inputs.each_ref().map(arg), normalized.each_ref().map(arg), gates.each_ref().map(arg));
        let (projection, compact_v) = (arg(&projection), compact.then(|| arg(&compact_v)));
        kernel.encode(projection, a_log, dt_bias, q_out, k_out, compact_v, beta, decay, h, g, k, v, q, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&projection, sentinel::<T>(), in_proj, "Prep in_proj");
        assert_inputs(&inputs, &[&f32s[0][..], &f32s[1][..]], "Prep input");
        (
            normalized.each_ref().map(|buffer| KernelFixture::read_guarded(buffer, sentinel())),
            KernelFixture::read_guarded(&compact_v, sentinel()),
            gates.each_ref().map(|buffer| KernelFixture::read_guarded(buffer, sentinel())),
            times,
        )
    }
}

/// Prep on the CPU in `submissions` submissions and on Vulkan once and, from 2 submissions, timed: every owned element
/// of q_norm_out, k_norm_out, beta_out and decay_out a member of its set and every other one its sentinel, compact V the
/// raw T values of every token bit for bit, every input and guard unchanged. Returns the times.
fn recurrence_prep_check<T: ArrayElement + Float + Debug + Default, QKT: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 5],
    modes: [bool; 2],
    in_proj: &[T],
    f32s: &[Vec<f32>; 2],
    submissions: usize,
    label: &str,
) -> (Option<(Duration, Duration)>, Vec<Duration>) {
    let kernel = DeltaNetPrefillPrepVulkanKernel::new(
        &fixture.context,
        T::data_type(),
        QKT::data_type(),
        128,
        modes[0],
        modes[1],
    )
    .expect("Prep");
    let mut span = 0;
    let sets = recurrence_prep_sets::<T, QKT>(shape, modes[0], in_proj, f32s, &mut span);
    let [h, _, k, v, q] = shape.map(|n| n as usize);
    let compact: Vec<T> = match modes[1] {
        true => (0..q * v).map(|i| in_proj[i / v * 2 * (k + v + h) + 2 * k + i % v]).collect(),
        false => Vec::new(),
    };
    let (normalized, copied, gates, cpu_times) =
        cpu_recurrence_prep::<T, QKT>(shape, 128, modes, in_proj, f32s, submissions);
    let (mut results, mut times) = (vec![("CPU", normalized, copied, gates)], None);
    for timed in [false, true].into_iter().take(1 + usize::from(submissions > 1)) {
        let (normalized, copied, gates, elapsed) =
            gpu_recurrence_prep::<T, QKT>(fixture, &kernel, shape, modes[1], in_proj, f32s, timed);
        results.push(("Vulkan", normalized, copied, gates));
        times = times.or(elapsed);
    }
    for (side, normalized, copied, gates) in &results {
        for (index, name) in ["q_norm_out", "k_norm_out", "beta_out", "decay_out"].into_iter().enumerate() {
            let ((sets, owner), label) = (&sets[index], format!("{label} {side} {name}"));
            match index {
                0 | 1 => check(sets, owner, &vec![sentinel::<QKT>(); owner.len()], &normalized[index], &label),
                _ => check(sets, owner, &vec![sentinel::<f32>(); owner.len()], &gates[index - 2], &label),
            }
        }
        assert_same_bits(&compact, copied, &format!("{label} {side} compact_v_out"));
    }
    let width = sets.iter().map(|(sets, _)| widest(sets)).fold(0.0, f64::max);
    eprintln!("{label}: widest relative set {width:.2e}; widest dt span {span}");
    (times, cpu_times)
}

fn recurrence_update_oracle<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for shape in RECURRENCE_UPDATE_SHAPES {
        let (in_proj, f32s) = recurrence_update_inputs::<T>(shape);
        recurrence_update_check(fixture, shape, &in_proj, &f32s, 1, &format!("Update {:?} {shape:?}", T::data_type()));
    }
}

/// Every Update shape over exactly its addressed extents: the whole state and every output within the class-aware
/// interval oracle on the CPU and Vulkan, every input and guard unchanged.
#[uzu_test]
fn recurrence_update_matches_oracle() {
    let fixture = KernelFixture::new();
    recurrence_update_oracle::<f32>(&fixture);
    recurrence_update_oracle::<bf16>(&fixture);
    fixture.assert_clean();
}

fn recurrence_update_witness<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (shape, big, ty) = ([1, 1, 1, 128, 1], 2f32.powi(24), format!("{:?}", T::data_type()));
    let row = |head: &[f32], tail: f32| [head.to_vec(), vec![tail; 128 - head.len()]].concat();
    // in_proj [q, k, v, z, β_raw, a_raw] and [a_log, dt_bias 0, norm_weight 1, state].
    let inputs = |q: &[f32], k: &[f32], state: &[f32], scalars: [f32; 4], a_log: f32| {
        let in_proj = [q, k, &scalars[..]].concat().iter().map(|&x| T::from(x).unwrap()).collect::<Vec<T>>();
        (in_proj, [vec![a_log], vec![0.0], vec![1.0], state.to_vec()])
    };
    let (q, k, state, scalars) =
        (row(&[1.0], 0.0), row(&[1.0; 3], 0.0), row(&[big, 1.0, -big], 0.0), [0.0, 1.0, 0.0, 21.0]);
    // β = 1/2 exactly and softplus(21) = 21 above 20: state[1] = d + k_1 δ with retrieved = d sk, which only the staging
    // separates from F1 (decay inside the k dot) and F2 (retrieved without decay), enumerated over every candidate inverse,
    // e^a_log and decay of each dt.
    let (in_proj, f32s) = inputs(&q, &k, &state, scalars, 0.0);
    let ([cpu, gpu], ..) =
        recurrence_update_check(fixture, shape, &in_proj, &f32s, 1, &format!("{ty} staging witness"));
    let [q_inv, k_inv] = [&q, &k].map(|values| members(recurrence_inverse(values)));
    let es = members(bounds::<f32>(exp(0.0).0));
    let decays = es.iter().map(|&e| members(decay::<f32, f32>(point(f64::from(e * 21.0)), true))).collect::<Vec<_>>();
    let combinations = q_inv.len() * k_inv.len() * decays.iter().map(Vec::len).sum::<usize>();
    assert_eq!(combinations, 22_750, "{ty}: the witness's candidate combinations");
    let (mut correct, mut faults) = (HashSet::new(), [HashSet::new(), HashSet::new()]);
    for (_, &ck, &d) in itertools::iproduct!(&q_inv, &k_inv, decays.iter().flatten()) {
        let terms = [big, 1.0, -big];
        let sk = terms.iter().fold(0.0f32, |acc, &s| acc + s * ck);
        let decayed = terms.iter().fold(0.0f32, |acc, &s| acc + d * s * ck);
        let next = |retrieved: f32| d * 1.0 + ck * (0.5 * (0.0 - retrieved));
        correct.insert(next(d * sk).to_bits());
        faults[0].insert(next(decayed).to_bits());
        faults[1].insert(next(sk).to_bits());
    }
    assert!(faults.iter().all(|fault| fault.is_disjoint(&correct)), "{ty}: a fault's state[1] is a correct candidate");
    for (side, next) in [("CPU", &cpu), ("Vulkan", &gpu)] {
        assert!(correct.contains(&next[1].to_bits()), "{ty} {side}: state[1] {} outside the candidates", next[1]);
    }
    eprintln!(
        "{ty} staging witness: {} and {} inverses, {} e^a_log, {combinations} combinations, {} state[1] values",
        q_inv.len(),
        k_inv.len(),
        es.len(),
        correct.len()
    );
    // [name, inputs, the output's one class]: NaN q; +inf k, whose inverse +0 makes k_0 NaN; -inf v, an infinite o and so
    // a +0 inverse times it; o² past FP32 with decay e^0 (a_log -inf), a +0 inverse of finite o; Root's iterator seed
    // witness, where kq = -0 from the -0 start makes o = -0 + +0 = +0, which a +0 start would make -0.
    let classes = [
        ("q NaN", inputs(&row(&[f32::NAN], 0.0), &k, &state, scalars, 0.0), point(f64::NAN)),
        ("k +inf", inputs(&q, &row(&[f32::INFINITY, 1.0, 1.0], 0.0), &state, scalars, 0.0), point(f64::NAN)),
        ("v -inf", inputs(&q, &k, &state, [f32::NEG_INFINITY, 1.0, 0.0, 21.0], 0.0), point(f64::NAN)),
        ("o² +inf", inputs(&q, &k, &row(&[2f32.powi(70), 1.0, -big], 0.0), scalars, f32::NEG_INFINITY), point(0.0)),
        (
            "kq -0",
            inputs(
                &row(&[-65536.0], -0.0),
                &row(&[], 0.0),
                &row(&[f32::MIN_POSITIVE], 0.0),
                [-0.0, 1.0, 0.0, 21.0],
                0.0,
            ),
            point(0.0),
        ),
    ];
    for (name, (in_proj, f32s), expected) in classes {
        let [_, out] = recurrence_update_sets(shape, &in_proj, &f32s, &mut 0);
        assert_eq!(out, [expected], "{ty} {name}: oracle");
        recurrence_update_check(fixture, shape, &in_proj, &f32s, 1, &format!("{ty} {name} witness"));
    }
}

/// Update witnesses, F32 first: Method 2's enumeration of 22,750 candidate combinations of the staging witness, whose
/// state[1] the CPU and Vulkan must take while faults F1 and F2 fall outside; NaN, infinities, a +0 inverse after an
/// overflowing square sum and the signed zero of kq's -0 start, each a single output class of the oracle.
#[uzu_test]
fn update_witnesses() {
    let fixture = KernelFixture::new();
    recurrence_update_witness::<f32>(&fixture);
    recurrence_update_witness::<bf16>(&fixture);
    fixture.assert_clean();
}

fn recurrence_prefill_cpu<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for shape in RECURRENCE_PREFILL_SHAPES {
        let (in_proj, f32s) = recurrence_prefill_inputs::<T>(
            shape,
            if shape[5] > 64 {
                0
            } else {
                11
            },
        );
        recurrence_prefill_check(
            fixture,
            shape,
            &in_proj,
            &f32s,
            1,
            &format!("Prefill {:?} {shape:?}", T::data_type()),
        );
    }
}

/// Every Prefill shape over exactly its addressed extents, with the specials on the short ones: state and output equal
/// to the CPU's bit for bit up to NaN payloads, every input, slack and guard unchanged.
#[uzu_test]
fn prefill_matches_cpu() {
    let fixture = KernelFixture::new();
    recurrence_prefill_cpu::<f32>(&fixture);
    recurrence_prefill_cpu::<bf16>(&fixture);
    fixture.assert_clean();
}

fn recurrence_prefill_witness<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (big, ty) = (2f32.powi(24), format!("{:?}", T::data_type()));
    let row = |head: &[f32]| [head.to_vec(), vec![0.0; 128 - head.len()]].concat();
    let f32s = [row(&[1.0]), row(&[1.0; 3]), vec![0.5], vec![0.75], row(&[big, 1.0, -big])];
    let label = format!("{ty} Prefill witness");
    let ((state, out), ..) =
        recurrence_prefill_check(fixture, [1, 1, 1, 128, 1, 1], &[T::zero(); 257], &f32s, 1, &label);
    // kv = (0.75 2^24 + 0.75) - 0.75 2^24 = 1 after rounding, so delta = -1/2 and the state 0.75 2^24 (to even), 0.25,
    // -0.75 2^24 and +0, and out its first element. F3 (decay after the dot, kv = 0) would leave state[1] 0.75 and F4
    // (o from the old state) give 2^24.
    assert_same_bits(&row(&[0.75 * big, 0.25, -0.75 * big]), &state, &format!("{label} state"));
    assert_same_bits(&[T::from(0.75 * big).unwrap()], &out, &format!("{label} out"));
    assert!(state[1] != 0.75 && out[0] != T::from(big).unwrap(), "{label}: a fault's value");
}

/// Prefill's exact witness for both storage types on the CPU and Vulkan, separating F3 and F4.
#[uzu_test]
fn prefill_witnesses() {
    let fixture = KernelFixture::new();
    recurrence_prefill_witness::<f32>(&fixture);
    recurrence_prefill_witness::<bf16>(&fixture);
    fixture.assert_clean();
}

fn recurrence_prep_oracle<T: ArrayElement + Float + Debug + Default, QKT: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture
) {
    for shape in RECURRENCE_PREP_SHAPES {
        for modes in [[false, false], [false, true], [true, false], [true, true]] {
            let (in_proj, f32s) = recurrence_prep_inputs::<T>(shape, modes[1], 13);
            let types = format!("{:?}/{:?}", T::data_type(), QKT::data_type());
            let label = format!("Prep {types} {shape:?} log {} compact {}", modes[0], modes[1]);
            recurrence_prep_check::<T, QKT>(fixture, shape, modes, &in_proj, &f32s, 1, &label);
        }
    }
}

/// Every Prep shape with the specials in in_proj, for both T and QKT and the four mode combinations: q, k, beta and the
/// (log) decay within the oracle on the CPU and Vulkan, untouched lanes and slack, every input and guard unchanged,
/// compact V bit for bit.
#[uzu_test]
fn prep_matches_oracle() {
    let fixture = KernelFixture::new();
    recurrence_prep_oracle::<f32, f32>(&fixture);
    recurrence_prep_oracle::<f32, bf16>(&fixture);
    recurrence_prep_oracle::<bf16, f32>(&fixture);
    recurrence_prep_oracle::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}

fn recurrence_prep_witness<T: ArrayElement + Float + Debug + Default, QKT: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture
) {
    let (shape, label) = ([1, 1, 128, 1, 1], format!("{:?}/{:?} Prep witness", T::data_type(), QKT::data_type()));
    let row = [vec![1.0], vec![0.0; 127]].concat();
    let in_proj =
        [&row[..], &row[..], &[0.0, 0.0, 0.0, 21.0]].concat().iter().map(|&x| T::from(x).unwrap()).collect::<Vec<T>>();
    let f32s = [vec![0.0], vec![0.0]];
    let sets = [true, false].map(|log| recurrence_prep_sets::<T, QKT>(shape, log, &in_proj, &f32s, &mut 0));
    let (query, decays) = (sets[0][0].0[0], [sets[0][3].0[0], sets[1][3].0[0]]);
    // F5 drops the scale, leaving QKT(1 inv), about 1, against about 0.0884; F6 swaps the modes, about -21 against about
    // 7.6e-10.
    let unscaled = mul::<QKT>(point(1.0), recurrence_inverse(&row));
    assert!(disjoint(query, unscaled) && disjoint(decays[0], decays[1]), "{label}: {query:?} {unscaled:?} {decays:?}");
    eprintln!("{label}: q_norm_out[0] {query:?}, log decay {:?}, decay {:?}", decays[0], decays[1]);
    for log in [true, false] {
        recurrence_prep_check::<T, QKT>(fixture, shape, [log, true], &in_proj, &f32s, 1, &format!("{label} log {log}"));
    }
}

/// Prep witnesses for the four type pairs, F32 first: the CPU and Vulkan within the q scale's and each decay mode's
/// sets, which exclude F5 and F6.
#[uzu_test]
fn prep_witnesses() {
    let fixture = KernelFixture::new();
    recurrence_prep_witness::<f32, f32>(&fixture);
    recurrence_prep_witness::<f32, bf16>(&fixture);
    recurrence_prep_witness::<bf16, f32>(&fixture);
    recurrence_prep_witness::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}

/// Prep's compact V presence, the generated binding's invariant, failing before any precondition, also without work;
/// every precondition with its message, HEAD_K_DIM's first; nothing recorded or changed. Then the CPU's own failures,
/// each on a fresh context: its encode-time variant panic for HEAD_K_DIM 129, its divisions by G = 0 or by
/// floor(H / G) = 0, also without tokens, and with debug assertions Update's H % G check; no CPU run for the ownership
/// guards, which the CPU would complete. Only F32 and BF16 exist.
#[uzu_test]
fn recurrence_presence_and_preconditions() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let buffers = [0; 8].map(|_| fixture.guarded(&[sentinel::<f32>(); 64], sentinel::<f32>()));
    let [a, b, c, d, e, f, g, h] = buffers.each_ref().map(arg);
    let mut encoding = fixture.encoding();
    for (compact, q) in [(true, 2), (true, 0), (false, 2), (false, 0)] {
        let prep = DeltaNetPrefillPrepVulkanKernel::new(context, DataType::F32, DataType::F32, 129, false, compact)
            .expect("Prep");
        let wrong = (!compact).then(|| f.clone());
        // SAFETY: the presence check panics before anything is recorded.
        let message = panics(|| unsafe {
            let (a, b, c, d, e, g, h) = (a.clone(), b.clone(), c.clone(), d.clone(), e.clone(), g.clone(), h.clone());
            prep.encode(a, b, c, d, e, wrong, g, h, 2, 0, 128, 4, q, &mut encoding)
        });
        let presence = "DeltaNetPrefillPrep: argument 'compact_v_out' must be present exactly when write_compact_v";
        let expected = format!("assertion `left == right` failed: {presence}\n  left: {}\n right: {compact}", !compact);
        assert_eq!(message, expected, "Prep compact {compact}, Q {q}");
    }
    let violated = |kernel: &str, text: &str| format!("{kernel}: precondition {text} violated");
    let (dims, groups, positive) = (
        "HEAD_K_DIM == 128",
        "num_v_heads == 0 || num_k_heads > 0 && num_v_heads.is_multiple_of(num_k_heads)",
        "num_k_heads > 0",
    );
    let enough = "num_v_heads == 0 || num_v_heads >= num_k_heads";
    let rows = "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim";
    let keys = "suffix_len <= 1 || num_k_heads <= key_dim / 128";
    for (dim, [hv, hk], text) in [(129, [2, 0], dims), (128, [2, 0], groups), (128, [3, 2], groups)] {
        let update = DeltaNetUpdateVulkanKernel::new(context, DataType::F32, dim).expect("Update");
        // SAFETY: the precondition panics before anything is recorded.
        let actual = panics(|| unsafe {
            let (a, b, c, d, e, f) = (a.clone(), b.clone(), c.clone(), d.clone(), e.clone(), f.clone());
            update.encode(a, b, c, d, e, f, hv, hk, 1, 128, 4, 1e-5, &mut encoding)
        });
        assert_eq!(actual, violated("DeltaNetUpdate", text), "Update HEAD_K_DIM {dim}, H {hv}, G {hk}");
    }
    let prefills = [
        (129, [0, 0, 1, 4, 0], dims),
        (128, [0, 0, 1, 4, 0], positive),
        (128, [1, 2, 1, 4, 0], enough),
        (128, [2, 1, 4, 7, 2], rows),
    ];
    for (dim, [hv, hk, dv, v, q], text) in prefills {
        let prefill = DeltaNetPrefillVulkanKernel::new(context, DataType::F32, dim).expect("Prefill");
        // SAFETY: as above.
        let actual = panics(|| unsafe {
            let (a, b, c, d, e, f, g) = (a.clone(), b.clone(), c.clone(), d.clone(), e.clone(), f.clone(), g.clone());
            prefill.encode(a, b, c, d, e, f, g, hv, hk, dv, 128, v, q, 1, &mut encoding)
        });
        assert_eq!(actual, violated("DeltaNetPrefill", text), "Prefill HEAD_K_DIM {dim}, H {hv}, G {hk}, Q {q}");
    }
    for (dim, [hv, hk, k, q], text) in
        [(129, [0, 0, 128, 0], dims), (128, [0, 0, 128, 0], positive), (128, [2, 2, 255, 2], keys)]
    {
        let prep = DeltaNetPrefillPrepVulkanKernel::new(context, DataType::F32, DataType::F32, dim, true, false)
            .expect("Prep");
        // SAFETY: as above.
        let actual = panics(|| unsafe {
            let (a, b, c, d, e, g, h) = (a.clone(), b.clone(), c.clone(), d.clone(), e.clone(), g.clone(), h.clone());
            prep.encode(a, b, c, d, e, None, g, h, hv, hk, k, 4, q, &mut encoding)
        });
        assert_eq!(actual, violated("DeltaNetPrefillPrep", text), "Prep HEAD_K_DIM {dim}, H {hv}, G {hk}, Q {q}");
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for buffer in &buffers {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &[sentinel::<f32>(); 64], "buffer");
        }
    }
    let unsupported = |variant: String| format!("not implemented: variant doesn't exist: {variant}");
    let update = |shape: [u32; 5], dim: u32| {
        panics(|| drop(cpu_recurrence_update::<f32>(shape, dim, &[], &Default::default(), 1)))
    };
    assert_eq!(update([2, 0, 1, 128, 4], 128), CPU_FAILURE, "CPU Update G 0");
    if cfg!(debug_assertions) {
        assert_eq!(update([3, 2, 1, 128, 4], 128), CPU_FAILURE, "CPU Update H % G");
    }
    assert_eq!(update([1, 1, 1, 128, 4], 129), unsupported(format!("{:?}", (DataType::F32, 129))));
    let prefill = |shape: [u32; 6], dim: u32| {
        panics(|| drop(cpu_recurrence_prefill::<f32>(shape, dim, &[], &Default::default(), 1)))
    };
    assert_eq!(prefill([0, 0, 1, 128, 4, 0], 128), CPU_FAILURE, "CPU Prefill G 0");
    assert_eq!(prefill([1, 2, 1, 128, 4, 0], 128), CPU_FAILURE, "CPU Prefill floor(H / G) 0");
    assert_eq!(prefill([1, 1, 1, 128, 4, 1], 129), unsupported(format!("{:?}", (DataType::F32, 129))));
    let prep = |shape: [u32; 5], dim: u32| {
        panics(|| drop(cpu_recurrence_prep::<f32, f32>(shape, dim, [false, false], &[], &Default::default(), 1)))
    };
    assert_eq!(prep([0, 0, 128, 4, 0], 128), CPU_FAILURE, "CPU Prep G 0");
    assert_eq!(prep([1, 1, 128, 4, 1], 129), unsupported(format!("{:?}", (DataType::F32, DataType::F32, 129))));
    let f16 = DataType::F16;
    let variant = |kernel: &str, error: Option<Error>| match error {
        Some(Error::KernelVariant {
            kernel: name,
            ..
        }) => assert_eq!(name, kernel),
        _ => panic!("{kernel}: F16 accepted"),
    };
    variant("DeltaNetUpdate", DeltaNetUpdateVulkanKernel::new(context, f16, 128).err());
    variant("DeltaNetPrefill", DeltaNetPrefillVulkanKernel::new(context, f16, 128).err());
    for [t, qkt] in [[f16, DataType::F32], [DataType::F32, f16]] {
        variant("DeltaNetPrefillPrep", DeltaNetPrefillPrepVulkanKernel::new(context, t, qkt, 128, false, false).err());
    }
    fixture.assert_clean();
}

/// No work at u32::MAX scalars and empty ranges, past the preconditions: Update without v heads (also with G = 0, which
/// the CPU completes too) or rows; Prefill without tokens, v heads or rows; Prep without tokens, with and without
/// compact V. Nothing is recorded or changed.
#[uzu_test]
fn recurrence_zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (context, empty, m) = (&fixture.context, fixture.guarded::<f32>(&[], sentinel()), u32::MAX);
    let e = || arg(&empty);
    let update = DeltaNetUpdateVulkanKernel::new(context, DataType::F32, 128).expect("Update");
    let prefill = DeltaNetPrefillVulkanKernel::new(context, DataType::F32, 128).expect("Prefill");
    let mut encoding = fixture.encoding();
    // SAFETY: without work nothing is indexed or recorded.
    unsafe {
        for [h, g, d] in [[0, m, m], [0, 0, m], [m, 1, 0]] {
            update.encode(e(), e(), e(), e(), e(), e(), h, g, d, m, m, 1e-5, &mut encoding);
        }
        for [h, g, d, q] in [[m, 1, m, 0], [0, m, m, m], [m, 1, 0, m]] {
            prefill.encode(e(), e(), e(), e(), e(), e(), e(), h, g, d, m, m, q, m, &mut encoding);
        }
        for compact in [false, true] {
            let prep =
                DeltaNetPrefillPrepVulkanKernel::new(context, DataType::F32, DataType::F32, 128, compact, compact)
                    .expect("Prep");
            prep.encode(e(), e(), e(), e(), e(), compact.then(e), e(), e(), m, m, m, m, 0, &mut encoding);
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty") };
    let (state, out, _) = cpu_recurrence_update::<f32>([0, 0, 4, 128, 4], 128, &[], &Default::default(), 1);
    assert!(state.is_empty() && out.is_empty(), "CPU Update without v heads and G 0");
    fixture.assert_clean();
}

fn recurrence_measure<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    lengths: &[u32],
) {
    let ([h, g, d, k, v], size, ty) =
        ([48u32, 16, 128, 2048, 6144], size_of::<T>() as u64, format!("{:?}", T::data_type()));
    let [h64, d64, v64] = [h, d, v].map(u64::from);
    let print = |kernel: &str, q: u32, bytes: u64, (times, mut cpu): (Option<(Duration, Duration)>, Vec<Duration>)| {
        let (gpu, wall) = times.expect("timed");
        cpu.drain(..3);
        cpu.sort();
        let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
        eprintln!(
            "DeltaNet{kernel} {ty} Q {q}: {bytes} B logical, provisional ({rate:.1} GB/s effective); GPU {gpu:?}, wall {wall:?}; CPU wall {:?}",
            cpu[cpu.len() / 2]
        );
    };
    let shape = [h, g, d, k, v];
    let (in_proj, f32s) = recurrence_update_inputs::<T>(shape);
    let (_, times, cpu) = recurrence_update_check(fixture, shape, &in_proj, &f32s, 13, &format!("{ty} Update"));
    print("Update", 1, h64 * (768 * size + 1024) + h64 * d64 * (2048 + 4 * size + 4), (times, cpu));
    for &q in lengths {
        let (shape, q64, lanes) = ([h, g, k, v, q], u64::from(q), 48 * (2 * size + 16));
        let (in_proj, f32s) = recurrence_prep_inputs::<T>(shape, false, 0);
        let flat = recurrence_prep_check::<T, f32>(
            fixture,
            shape,
            [false, false],
            &in_proj,
            &f32s,
            13,
            &format!("{ty} flat Prep Q {q}"),
        );
        print("PrefillPrep flat", q, q64 * (16 * (512 * size + 1024) + lanes), flat);
        let (in_proj, f32s) = recurrence_prep_inputs::<T>(shape, true, 0);
        let tree = recurrence_prep_check::<T, T>(
            fixture,
            shape,
            [true, true],
            &in_proj,
            &f32s,
            13,
            &format!("{ty} tree Prep Q {q}"),
        );
        print("PrefillPrep tree", q, q64 * (16 * 768 * size + lanes + 2 * v64 * size), tree);
        let shape = [h, g, d, k, v, q];
        let (in_proj, f32s) = recurrence_prefill_inputs::<T>(shape, 0);
        let (_, times, cpu) =
            recurrence_prefill_check(fixture, shape, &in_proj, &f32s, 13, &format!("{ty} Prefill Q {q}"));
        print("Prefill", q, q64 * h64 * d64 * (3080 + 2 * size), (times, cpu));
    }
}

/// Run alone, without sync validation: `... delta_net_test::recurrence_throughput -- --ignored --nocapture`. Synthetic
/// inputs at the common test's Qwen3.5-labelled shapes (48 v heads, 16 k heads of 128, Dv 128, so key_dim 2048,
/// value_dim 6144 and total_proj_dim 16480), not verified model files: Update for one token, flat Prep (FP32 q and k,
/// decays), tree Prep (T q and k, log decays, compact V) and Prefill for 64 and 1024 tokens, in two rounds of opposite
/// order. Prints the GPU and wall medians of 10 Vulkan submissions after 3 warm-up ones and the CPU kernels' wall
/// medians. Update and Prefill write their state, so every submission gets its own fresh bundle, each checked after
/// completion against the oracle or the CPU; Prep's outputs are checked once and after timing. The bytes are the logical
/// loads and stores of the shaders' loops, provisional until reviewed, not measured bandwidth.
#[uzu_test]
#[ignore]
fn recurrence_throughput() {
    let fixture = KernelFixture::new();
    for (round, lengths) in [[64, 1024], [1024, 64]].iter().enumerate() {
        eprintln!("DeltaNet recurrence throughput round {round}");
        recurrence_measure::<f32>(&fixture, lengths);
        recurrence_measure::<bf16>(&fixture, lengths);
    }
    fixture.assert_clean();
}
