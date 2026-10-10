use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem::size_of,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    arg, assert_same_bits, conv1d_values as values, cpu_buffer, cpu_submissions, exp, interval,
    kernel_fixture::KernelFixture, oracle,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, gpu_types::ActivationType, kernel::SSDUpdateKernel},
        cpu::Cpu,
        vulkan::{SSDUpdateVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

// A set of values is `((lo, hi), classes)`: its finite nonzero members lie in [lo, hi], none when lo > hi, and its other
// members are the classes below. It is one bit pattern only as one finite value or one class other than NaN, so a zero's
// sign is never inferred from +0 == -0.
pub const NAN: u8 = 1;
pub const NEG_INF: u8 = 2;
pub const POS_INF: u8 = 4;
pub const NEG_ZERO: u8 = 8;
pub const POS_ZERO: u8 = 16;
const NONE: (f64, f64) = (f64::INFINITY, f64::NEG_INFINITY);
/// The smallest positive FP32 value.
const TINY: f64 = f32::from_bits(1) as f64;

/// The stage `update` keeps in FP32 instead of rounding it to T, a staging that witnesses tell apart; NARROWED keeps none.
const NARROWED: usize = 0;
const DT: usize = 1;
const DECAY: usize = 2;
const GATE: usize = 3;

/// B 2, H 3, Dh 2, group_size 2, N 4 with gaps in x, dt_raw, B and C and between the state rows, and ignored last
/// strides; DENSE is another layout of the shape within PADDED's spans.
const PADDED_DIMS: [u32; 5] = [2, 3, 2, 2, 4];
const PADDED: [u32; 12] = [16, 5, 2, 4, 1, 12, 5, 7, 48, 15, 6, 9];
const DENSE: [u32; 12] = [12, 2, 1, 3, 1, 10, 4, 0, 40, 12, 4, 0];
/// One head per witness, each with its own x, z, dt_raw, B, C, d and state element.
const BATCHED: [u32; 12] = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0];
/// dt_raw values with proven or characterized decays: around softplus's 20, at exp's subnormals and past 2^-150, the
/// classes and -0.
const DT_RAW: [f32; 10] = [-104.0, 0.0, 21.0, 32.0, 90.0, 104.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0];

/// The set of one value; each negative class is its positive one shifted right once.
pub fn point(value: f64) -> ((f64, f64), u8) {
    let negative = u8::from(value.is_sign_negative());
    match value {
        _ if value.is_nan() => (NONE, NAN),
        _ if value.is_infinite() => (NONE, POS_INF >> negative),
        _ if value == 0.0 => (NONE, POS_ZERO >> negative),
        _ => ((value, value), 0),
    }
}

pub fn union(
    ((a, b), m): ((f64, f64), u8),
    ((c, d), n): ((f64, f64), u8),
) -> ((f64, f64), u8) {
    ((a.min(c), b.max(d)), m | n)
}

/// The negated members, whose classes swap signs.
fn negate(((lo, hi), mask): ((f64, f64), u8)) -> ((f64, f64), u8) {
    ((-hi, -lo), mask & NAN | (mask & (NEG_INF | NEG_ZERO)) << 1 | (mask & (POS_INF | POS_ZERO)) >> 1)
}

/// The reals [a, b], nonzero and of one sign, each rounded to FP32 and then to U. That is monotonic, so the rounded
/// endpoints bound it: one rounding to zero or infinity adds that class, and only finite results bound finite members.
pub fn round<U: Float>((a, b): (f64, f64)) -> ((f64, f64), u8) {
    if b < 0.0 {
        return negate(round::<U>((-b, -a)));
    }
    let to_u = |value: f64| U::from(value as f32).unwrap().to_f64().unwrap();
    let (lo, hi) = (to_u(a), to_u(b));
    let mask = (u8::from(lo == 0.0) * POS_ZERO) | (u8::from(hi.is_infinite()) * POS_INF);
    let smallest = U::min_positive_value().to_f64().unwrap() * U::epsilon().to_f64().unwrap();
    match lo.is_infinite() || hi == 0.0 {
        true => (NONE, mask),
        false => ((lo.max(smallest), hi.min(U::max_value().to_f64().unwrap())), mask),
    }
}

/// The classes of a set, each alone, and its finite members of each sign.
fn parts(((lo, hi), mask): ((f64, f64), u8)) -> Vec<((f64, f64), u8)> {
    let classes = [NAN, NEG_INF, POS_INF, NEG_ZERO, POS_ZERO].into_iter().filter(|class| mask & class != 0);
    let mut parts = classes.map(|class| (NONE, class)).collect::<Vec<_>>();
    if lo < 0.0 {
        parts.push(((lo, hi.min(-TINY)), 0));
    }
    if hi > 0.0 {
        parts.push(((lo.max(TINY), hi), 0));
    }
    parts
}

/// The union of `operation` over every pair of parts.
fn combine(
    a: ((f64, f64), u8),
    b: ((f64, f64), u8),
    operation: impl Fn(((f64, f64), u8), ((f64, f64), u8)) -> ((f64, f64), u8),
) -> ((f64, f64), u8) {
    let pairs = parts(a).into_iter().flat_map(|p| parts(b).into_iter().map(move |q| (p, q)));
    pairs.fold((NONE, 0), |set, (p, q)| union(set, operation(p, q)))
}

/// T(a b) of every pair of members, rounded from FP32: classes by IEEE rules, finite products between the exact corner
/// products of each sign.
pub fn mul<T: Float>(
    a: ((f64, f64), u8),
    b: ((f64, f64), u8),
) -> ((f64, f64), u8) {
    combine(a, b, |p, q| {
        let classes = p.1 | q.1;
        let (zero, infinite) = (classes & (NEG_ZERO | POS_ZERO) != 0, classes & (NEG_INF | POS_INF) != 0);
        let is_negative = |((_, hi), mask): ((f64, f64), u8)| mask & (NEG_INF | NEG_ZERO) != 0 || mask == 0 && hi < 0.0;
        let negative = u8::from(is_negative(p) != is_negative(q));
        match () {
            _ if classes & NAN != 0 || zero && infinite => (NONE, NAN),
            _ if zero => (NONE, POS_ZERO >> negative),
            _ if infinite => (NONE, POS_INF >> negative),
            _ => {
                let corners = [p.0.0 * q.0.0, p.0.0 * q.0.1, p.0.1 * q.0.0, p.0.1 * q.0.1];
                let low = corners.into_iter().fold(f64::INFINITY, f64::min);
                round::<T>((low, corners.into_iter().fold(f64::NEG_INFINITY, f64::max)))
            },
        }
    })
}

/// T(a + b) of every pair of members of T, rounded from FP32: classes by IEEE rules, finite sums between the endpoint
/// sums, where opposite signs may cancel to +0 and other sums are at least T's smallest positive value.
pub fn add<T: Float>(
    a: ((f64, f64), u8),
    b: ((f64, f64), u8),
) -> ((f64, f64), u8) {
    let smallest = T::min_positive_value().to_f64().unwrap() * T::epsilon().to_f64().unwrap();
    combine(a, b, |p, q| {
        let classes = p.1 | q.1;
        match () {
            _ if classes & NAN != 0 || classes & (NEG_INF | POS_INF) == NEG_INF | POS_INF => (NONE, NAN),
            _ if classes & (NEG_INF | POS_INF) != 0 => (NONE, classes & (NEG_INF | POS_INF)),
            _ if p.1 & q.1 & NEG_ZERO != 0 => (NONE, NEG_ZERO),
            _ if p.1 != 0 && q.1 != 0 => (NONE, POS_ZERO),
            _ if p.1 != 0 => q,
            _ if q.1 != 0 => p,
            _ => by_sign::<T>((p.0.0 + q.0.0, p.0.1 + q.0.1), smallest, POS_ZERO),
        }
    })
}

/// The reals [lo, hi] rounded to U by sign, where 0 is the classes `zeros` and other values are at least `least`.
fn by_sign<U: Float>(
    (lo, hi): (f64, f64),
    least: f64,
    zeros: u8,
) -> ((f64, f64), u8) {
    let mut set = (NONE, u8::from(lo <= 0.0 && hi >= 0.0) * zeros);
    if lo <= -least {
        set = union(set, round::<U>((lo, hi.min(-least))));
    }
    if hi >= least {
        set = union(set, round::<U>((lo.max(least), hi)));
    }
    set
}

/// An oracle's bounds on an FP32 result rounded to U: NaN bounds give NaN, and bounds holding 0 both zeros.
pub fn bounds<U: Float>((lo, hi): (f64, f64)) -> ((f64, f64), u8) {
    match lo.is_nan() || hi.is_nan() {
        true => (NONE, NAN),
        false => by_sign::<U>((lo, hi), TINY, NEG_ZERO | POS_ZERO),
    }
}

/// The one bit pattern of a set, if it holds exactly one.
pub fn single(((lo, hi), mask): ((f64, f64), u8)) -> Option<f64> {
    let classes = [(NEG_INF, f64::NEG_INFINITY), (POS_INF, f64::INFINITY), (NEG_ZERO, -0.0), (POS_ZERO, 0.0)];
    match mask {
        0 => (lo.to_bits() == hi.to_bits()).then_some(lo),
        _ if lo <= hi => None,
        _ => classes.into_iter().find(|&(class, _)| class == mask).map(|(_, value)| value),
    }
}

/// exp(-t) rounded to V over every member t of dt, a set of U values, which softplus never makes negative: NaN propagates
/// and +inf gives +0 (the shader's branch, the CPU's expf(-inf)). Zeros and finite t give, on Vulkan, Exp's bound, whose
/// subnormal results may flush to the zero of their sign; on the CPU 1 ULP, glibc's expf as characterized, not a Rust
/// guarantee.
pub fn decay<U: ArrayElement + Float, V: Float>(
    dt: ((f64, f64), u8),
    shader: bool,
) -> ((f64, f64), u8) {
    let ((lo, hi), mask) = dt;
    assert!(mask & NEG_INF == 0 && lo >= 0.0, "negative softplus {dt:?}");
    let mut set = (NONE, mask & NAN | (u8::from(mask & POS_INF != 0) * POS_ZERO));
    let zero = (mask & (NEG_ZERO | POS_ZERO) != 0).then_some(0.0);
    // Positive U values step by their bits.
    let next = |t: f64| (KernelFixture::ordinal(U::from(t).unwrap()) + 1).to_ne_bytes();
    let next = |t| bytemuck::pod_read_unaligned::<U>(&next(t)[..size_of::<U>()]).to_f64().unwrap();
    let finite = std::iter::successors((lo <= hi).then_some(lo), |&t| Some(next(t)).filter(|&t| t <= hi));
    let normal = f64::from(f32::MIN_POSITIVE);
    for t in zero.into_iter().chain(finite) {
        let e = (-t).exp();
        let (a, b) = [interval(e, e, 1.0), exp(-t).0][usize::from(shader)];
        let flushed = (u8::from(shader && b > 0.0 && a < normal) * POS_ZERO)
            | (u8::from(shader && a < 0.0 && b > -normal) * NEG_ZERO);
        set = union(set, union(bounds::<V>((a, b)), (NONE, flushed)));
    }
    set
}

/// Sets of next_state and y of one (batch, head, element) as the shader or CPU stages it, `wide` kept in FP32: in order
/// next = T(T(s decay) + T(b x)), acc from +0 adds T(next c), then T(d x); y = T(acc gate).
fn update<T: ArrayElement + Float>(
    shader: bool,
    wide: usize,
    [x, dt_raw, d, z]: [f64; 4],
    bc: &[(f64, f64)],
    state: &[((f64, f64), u8)],
) -> (Vec<((f64, f64), u8)>, ((f64, f64), u8)) {
    let (x, dt, gate) = (point(x), oracle(dt_raw, ActivationType::SOFTPLUS).0, oracle(z, ActivationType::SILU).0);
    let decay = match wide {
        DT => decay::<f32, T>(bounds::<f32>(dt), shader),
        DECAY => decay::<T, f32>(bounds::<T>(dt), shader),
        _ => decay::<T, T>(bounds::<T>(dt), shader),
    };
    let gate = [bounds::<T>(gate), bounds::<f32>(gate)][usize::from(wide == GATE)];
    let mut acc = (NONE, POS_ZERO);
    let next = bc
        .iter()
        .zip(state)
        .map(|(&(b, c), &s)| {
            let next = add::<T>(mul::<T>(s, decay), mul::<T>(point(b), x));
            acc = add::<T>(acc, mul::<T>(next, point(c)));
            next
        })
        .collect::<Vec<_>>();
    (next, mul::<T>(add::<T>(acc, mul::<T>(point(d), x)), gate))
}

fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

/// [x; dt; B and C; state] strides as the kernels take them.
fn split(s: &[u32; 12]) -> ([u32; 3], [u32; 2], [u32; 3], [u32; 4]) {
    (s[..3].try_into().unwrap(), s[3..5].try_into().unwrap(), s[5..8].try_into().unwrap(), s[8..].try_into().unwrap())
}

/// Contiguous strides of [B, H, Dh, group_size, N], as Mamba2's.
fn contiguous([_, heads, elements, group, n]: [u32; 5]) -> [u32; 12] {
    let groups = heads.div_ceil(group.max(1));
    [heads * elements, elements, 1, heads, 1, groups * n, n, 1, heads * elements * n, elements * n, n, 1]
}

/// [x (z, y), dt_raw, B and C row, state row, head] of every (batch, head, element) in order, after asserting that the y
/// elements and state rows written are distinct.
fn items(
    dims: [u32; 5],
    strides: [u32; 12],
) -> Vec<[usize; 5]> {
    let ([batches, heads, elements, group, n], s) = (dims.map(u64::from), strides.map(u64::from));
    let item = move |b, h, e| {
        [
            b * s[0] + h * s[1] + e * s[2],
            b * s[3] + h * s[4],
            b * s[5] + h / group * s[6],
            b * s[8] + h * s[9] + e * s[10],
            h,
        ]
    };
    let items = (0..batches)
        .flat_map(|b| (0..heads).flat_map(move |h| (0..elements).map(move |e| item(b, h, e).map(|i| i as usize))))
        .collect::<Vec<_>>();
    let n = n as usize;
    let ys = items.iter().map(|item| item[0]).collect::<HashSet<_>>();
    let states = items.iter().flat_map(|item| item[3]..item[3] + n).collect::<HashSet<_>>();
    assert_eq!((ys.len(), states.len()), (items.len(), items.len() * n), "{dims:?} {strides:?} writes alias");
    items
}

/// Valid spans [x (z, y), dt_raw, B and C, d, state]: one past the last element read, 0 without work.
fn spans(
    dims: [u32; 5],
    strides: [u32; 12],
) -> [usize; 5] {
    let (items, n) = (items(dims, strides), dims[4] as usize);
    let end = |index: usize, width: usize| match width {
        0 => 0,
        _ => items.iter().map(|item| item[index] + width).max().unwrap_or(0),
    };
    [end(0, 1), end(1, 1), end(2, n), end(4, 1), end(3, n)]
}

/// [x, dt_raw, b, c, d, z, state] over the valid spans: finite eighths, the specials in every fifth x, dt_raw and z and
/// every third state element, and DT_RAW in every other dt_raw.
fn inputs<T: ArrayElement + Float>(
    dims: [u32; 5],
    strides: [u32; 12],
) -> [Vec<T>; 7] {
    let [x, dt, bc, d, state] = spans(dims, strides);
    let mut dt_raw = values::<T>(dt, 1, 5);
    for (index, value) in dt_raw.iter_mut().enumerate().skip(1).step_by(2) {
        *value = T::from(DT_RAW[index / 2 % DT_RAW.len()]).unwrap();
    }
    [values(x, 0, 5), dt_raw, values(bc, 2, 0), values(bc, 3, 0), values(d, 4, 0), values(x, 5, 5), values(state, 6, 3)]
}

/// [B, H, Dh, group_size, N] with strides: contiguous with N 0 to 128, group tails, batches, ignored last strides 7;
/// PADDED; interleaved heads and elements; dt_raw, B and C broadcast over batches; u32::MAX strides of unit extents.
fn cases() -> Vec<([u32; 5], [u32; 12])> {
    let shapes =
        [[1, 1, 1, 1, 0], [1, 2, 129, 1, 1], [2, 3, 5, 2, 63], [1, 4, 3, 4, 64], [3, 5, 2, 3, 65], [1, 6, 7, 1, 128]];
    let mut cases = shapes.map(|dims| (dims, contiguous(dims))).to_vec();
    let (mut ignored, m) = (contiguous([2, 3, 5, 2, 63]), u32::MAX);
    (ignored[7], ignored[11]) = (7, 7);
    cases.extend([
        ([2, 3, 5, 2, 63], ignored),
        (PADDED_DIMS, PADDED),
        ([1, 3, 2, 1, 1], [0, 2, 3, 0, 1, 0, 1, 0, 0, 2, 3, 0]),
        ([2, 2, 3, 1, 4], [6, 3, 1, 0, 0, 0, 4, 1, 24, 12, 4, 1]),
        ([1, 1, 3, 1, 2], [m, m, 1, m, m, m, m, 1, m, m, 2, 1]),
    ]);
    cases
}

/// Initial [y, next_state]: sentinels, next_state the state in place.
fn initial<T: Float>(
    inputs: &[Vec<T>; 7],
    in_place: bool,
) -> [Vec<T>; 2] {
    [
        vec![sentinel(); inputs[0].len()],
        inputs[6].iter().map(|&state| [sentinel(), state][usize::from(in_place)]).collect(),
    ]
}

/// The CPU kernel over [x, dt_raw, b, c, d, z, state] in `submissions` timed submissions: [y, next_state] from `initial`
/// and each submission's wall time.
fn cpu_update<T: ArrayElement + Float + Default>(
    dims: [u32; 5],
    strides: [u32; 12],
    inputs: &[Vec<T>; 7],
    in_place: bool,
    submissions: usize,
) -> ([Vec<T>; 2], Vec<Duration>) {
    let [b_size, h_size, dh_size, group_size, state_size] = dims;
    let (x_strides, dt_strides, cb_strides, state_strides) = split(&strides);
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::SSDUpdateKernel::new(&context, T::data_type(), in_place)
        .expect("CPU SSDUpdate");
    let [x, dt_raw, b, c, d, z, state] = inputs.each_ref().map(|values| cpu_buffer(&context, values));
    let [mut y, mut next_state] = initial(inputs, in_place).map(|values| cpu_buffer(&context, &values));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        kernel.encode(
            &x,
            &dt_raw,
            &b,
            &c,
            &d,
            &z,
            (!in_place).then_some(&state),
            &mut y,
            &mut next_state,
            group_size,
            state_size,
            &x_strides,
            &dt_strides,
            &cb_strides,
            &state_strides,
            b_size,
            h_size,
            dh_size,
            command_buffer,
        );
    });
    let [y_len, state_len] = [inputs[0].len(), inputs[6].len()];
    ([buffer_prefix_to_vec::<Cpu, T>(&y, y_len), buffer_prefix_to_vec::<Cpu, T>(&next_state, state_len)], times)
}

/// Guarded ranges of [x, dt_raw, b, c, d, z, state] and of [y, next_state] from `initial`.
fn gpu_buffers<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    inputs: &[Vec<T>; 7],
    in_place: bool,
) -> [(Arc<VkBuffer>, Range<u64>); 9] {
    let fill = sentinel::<T>();
    let [x, dt_raw, b, c, d, z, state] = inputs.each_ref().map(|values| fixture.guarded(values, fill));
    let [y, next_state] = initial(inputs, in_place).map(|values| fixture.guarded(&values, fill));
    [x, dt_raw, b, c, d, z, state, y, next_state]
}

/// Records one dispatch over `gpu_buffers`, the state only out of place.
///
/// # Safety
/// The ranges hold the valid spans of `dims` and the strides, and written ranges alias nothing.
unsafe fn encode(
    kernel: &SSDUpdateVulkanKernel,
    buffers: &[(Arc<VkBuffer>, Range<u64>); 9],
    [batches, heads, width, group, n]: [u32; 5],
    (xs, dts, cbs, ss): (&[u32; 3], &[u32; 2], &[u32; 3], &[u32; 4]),
    in_place: bool,
    encoding: &mut VkCommandBufferEncoding,
) {
    let [x, dt, b, c, d, z, state, y, next] = buffers.each_ref().map(arg);
    let state = (!in_place).then_some(state);
    unsafe {
        kernel.encode(x, dt, b, c, d, z, state, y, next, group, n, xs, dts, cbs, ss, batches, heads, width, encoding)
    }
}

/// [y, next_state] of `gpu_buffers`, after asserting the read-only inputs and every guard.
///
/// # Safety
/// Every command buffer using the buffers has completed.
unsafe fn gpu_results<T: ArrayElement + Float>(
    buffers: &[(Arc<VkBuffer>, Range<u64>); 9],
    inputs: &[Vec<T>; 7],
) -> [Vec<T>; 2] {
    let fill = sentinel::<T>();
    for ((name, buffer), values) in ["x", "dt_raw", "b", "c", "d", "z", "state"].into_iter().zip(buffers).zip(inputs) {
        unsafe { KernelFixture::assert_unchanged(buffer, fill, values, name) };
    }
    [7, 8].map(|index| unsafe { KernelFixture::read_guarded(&buffers[index], fill) })
}

/// The Vulkan counterpart of `cpu_update`: `dispatches` dispatches in one command buffer or, when `timed`, one in each of
/// `median_times`' submissions.
fn gpu_update<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &SSDUpdateVulkanKernel,
    dims: [u32; 5],
    strides: [u32; 12],
    inputs: &[Vec<T>; 7],
    in_place: bool,
    dispatches: usize,
    timed: bool,
) -> ([Vec<T>; 2], Option<(Duration, Duration)>) {
    let (buffers, (x, dt, cb, state)) = (gpu_buffers(fixture, inputs, in_place), split(&strides));
    // SAFETY: each range holds exactly its valid span, and `items` asserts every written element distinct.
    let mut record = |encoding: &mut VkCommandBufferEncoding| {
        for _ in 0..dispatches {
            unsafe { encode(kernel, &buffers, dims, (&x, &dt, &cb, &state), in_place, encoding) }
        }
    };
    let times = timed.then(|| fixture.median_times(&mut record));
    if !timed {
        let mut encoding = fixture.encoding();
        record(&mut encoding);
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    (unsafe { gpu_results(&buffers, inputs) }, times)
}

/// Checks [y, next_state] of the CPU and Vulkan after `steps` updates (in place chained) against their own sets: owned
/// elements are members, all others keep their initial bits.
fn check<T: ArrayElement + Float + Debug>(
    dims: [u32; 5],
    strides: [u32; 12],
    inputs: &[Vec<T>; 7],
    in_place: bool,
    steps: usize,
    cpu: &[Vec<T>; 2],
    gpu: &[Vec<T>; 2],
    label: &str,
) {
    let (items, n) = (items(dims, strides), dims[4] as usize);
    let value = |input: usize, index: usize| inputs[input][index].to_f64().unwrap();
    let mut violations = 0;
    for (shader, results) in [(true, gpu), (false, cpu)] {
        let mut state = inputs[6].iter().map(|value| point(value.to_f64().unwrap())).collect::<Vec<_>>();
        let mut owned = [HashMap::new(), HashMap::new()];
        for _ in 0..steps {
            let mut next = state.clone();
            for &[x, dt, row, s, h] in &items {
                let bc = (row..row + n).map(|i| (value(2, i), value(3, i))).collect::<Vec<_>>();
                let input = [value(0, x), value(1, dt), value(4, h), value(5, x)];
                let (rows, y) = update::<T>(shader, NARROWED, input, &bc, &state[s..s + n]);
                next[s..s + n].copy_from_slice(&rows);
                owned[0].insert(x, y);
                owned[1].extend((s..s + n).zip(rows));
            }
            if in_place {
                state = next;
            }
        }
        for (k, (results, initial)) in results.iter().zip(initial(inputs, in_place)).enumerate() {
            assert_eq!(results.len(), initial.len(), "{label}: length");
            for (index, (&actual, initial)) in results.iter().zip(initial).enumerate() {
                let valid = match (owned[k].get(&index), point(actual.to_f64().unwrap())) {
                    (None, _) => bytemuck::bytes_of(&actual) == bytemuck::bytes_of(&initial),
                    (Some(&((lo, hi), _)), ((value, _), 0)) => lo <= value && value <= hi,
                    (Some(&(_, mask)), (_, class)) => mask & class != 0,
                };
                violations += usize::from(!valid);
                if !valid && violations <= 5 {
                    let (side, name) = (["CPU", "Vulkan"][usize::from(shader)], ["y", "next_state"][k]);
                    eprintln!("{label}: {side} {name} {index}: {actual:?} outside {:?}", owned[k].get(&index));
                }
            }
        }
    }
    assert_eq!(violations, 0, "{label}: {violations} elements outside the oracle");
}

/// Every case, in and out of place, against the oracle.
#[uzu_test]
fn matches_cpu_all_types() {
    fn matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        for in_place in [false, true] {
            let kernel = SSDUpdateVulkanKernel::new(&fixture.context, T::data_type(), in_place).expect("SSDUpdate");
            for (dims, strides) in cases() {
                let label = format!("{:?} {dims:?} {strides:?} in place {in_place}", T::data_type());
                let inputs = inputs::<T>(dims, strides);
                let (cpu, _) = cpu_update(dims, strides, &inputs, in_place, 1);
                let (gpu, _) = gpu_update(fixture, &kernel, dims, strides, &inputs, in_place, 1, false);
                check(dims, strides, &inputs, in_place, 1, &cpu, &gpu, &label);
            }
        }
    }
    let fixture = KernelFixture::new();
    matches::<f32>(&fixture);
    matches::<f16>(&fixture);
    matches::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Hand-specified class boundaries: +inf with zeros, -inf and both signs; single values; cancellation; signed underflow;
/// overflow throughout or in part; bounds holding 0; the decay's class branch (CPU only).
#[uzu_test]
fn oracle_boundaries() {
    let (inf, finite) = (point(f64::INFINITY), ((-3.0, 2.0), 0));
    assert_eq!(mul::<f32>(inf, union(point(0.0), ((1.0, 2.0), 0))), (NONE, NAN | POS_INF));
    assert_eq!(add::<f32>(inf, union(point(f64::NEG_INFINITY), finite)), (NONE, NAN | POS_INF));
    assert_eq!(mul::<f32>(inf, finite), (NONE, NEG_INF | POS_INF));
    assert_eq!(single((NONE, NEG_ZERO | POS_ZERO)), None);
    assert_eq!(single((NONE, NAN)), None);
    assert_eq!(single(point(-0.0)).map(f64::to_bits), Some((-0.0f64).to_bits()));
    assert_eq!(single(point(1.5)), Some(1.5));
    assert_eq!(add::<f32>(((1.0, 2.0), 0), ((-2.0, -1.0), 0)), ((-1.0, 1.0), POS_ZERO));
    assert_eq!(add::<f32>(point(1.0), point(-1.0)), (NONE, POS_ZERO));
    assert_eq!(mul::<f32>(point(TINY), point(-0.5)), (NONE, NEG_ZERO));
    assert_eq!(mul::<f16>(point(2f64.powi(-24)), point(-0.25)), (NONE, NEG_ZERO));
    assert_eq!(mul::<f16>(((256.0, 300.0), 0), point(512.0)), (NONE, POS_INF));
    assert_eq!(mul::<f16>(((64.0, 256.0), 0), point(512.0)), ((32768.0, 65504.0), POS_INF));
    assert_eq!(bounds::<f32>((-1.0, 1.0)), ((-1.0, 1.0), NEG_ZERO | POS_ZERO));
    assert_eq!(decay::<f32, f32>(inf, true), (NONE, POS_ZERO));
    assert_eq!(decay::<f32, f32>(point(f64::NAN), false), (NONE, NAN));
}

/// One element of `b.len()` state elements in place from [x, dt_raw, d, z]: the CPU and Vulkan each give exactly `y` and
/// `next` (any NaN for a NaN).
fn witness<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    name: &str,
    [x, dt_raw, d, z]: [T; 4],
    b: &[T],
    c: &[T],
    state: &[T],
    y: T,
    next: &[T],
) {
    let dims = [1, 1, 1, 1, b.len() as u32];
    let inputs = [vec![x], vec![dt_raw], b.to_vec(), c.to_vec(), vec![d], vec![z], state.to_vec()];
    let kernel = SSDUpdateVulkanKernel::new(&fixture.context, T::data_type(), true).expect("SSDUpdate");
    let (cpu, _) = cpu_update(dims, contiguous(dims), &inputs, true, 1);
    let (gpu, _) = gpu_update(fixture, &kernel, dims, contiguous(dims), &inputs, true, 1, false);
    for (side, [ys, states]) in [("CPU", cpu), ("Vulkan", gpu)] {
        KernelFixture::assert_bits(&[y], &ys, &format!("{name}: {side} y"));
        KernelFixture::assert_bits(next, &states, &format!("{name}: {side} next_state"));
    }
}

/// Independent FP32 results with dt_raw = +inf, whose decay is exactly +0, and z = 2^-27, whose gate is exactly 2^-28:
/// d x added after the fold, the fold ascending and unfused, the accumulator from +0.
#[uzu_test]
fn order_witnesses() {
    let fixture = KernelFixture::new();
    let p = |e: i32| 2f32.powi(e);
    let cases: [(&str, f32, &[f32], &[f32], &[f32], f32); 5] = [
        ("d x after the fold: first gives 2^-28", -p(24), &[p(24), 1.0], &[1.0; 2], &[1.0; 2], 0.0),
        ("ascending fold: descending gives (2^24 + 2) 2^-28", 0.0, &[p(24), 1.0, 1.0], &[1.0; 3], &[0.0; 3], p(-4)),
        ("+0 start without state", -0.0, &[], &[], &[], 0.0),
        ("+0 start before a -0 term", -0.0, &[-0.0], &[1.0], &[-1.0], 0.0),
        ("unfused: an FMA gives 2^-52", 0.0, &[-(1.0 + p(-11)), 1.0 + p(-12)], &[1.0, 1.0 + p(-12)], &[0.0; 2], 0.0),
    ];
    for (name, d, b, c, state, y) in cases {
        witness(&fixture, name, [1.0, f32::INFINITY, d, p(-27)], b, c, state, y, b);
    }
    fixture.assert_clean();
}

/// With dt_raw = -104 (a decay of exactly 1) and z = 32 (a gate of exactly 32), 16-bit types round each product and sum:
/// unrounded, the state update would keep 2^-20 (F16) or 2^-14 (BF16) and the fold 1 + 2^-10 or 1 + 2^-7.
fn half_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (zero, one, eps, v) = (T::zero(), T::one(), T::epsilon(), |value: f32| T::from(value).unwrap());
    let inputs = |x| [x, v(-104.0), zero, v(32.0)];
    let x = one + eps;
    witness(fixture, "state update", inputs(x), &[x], &[one], &[-(x + eps)], zero, &[zero]);
    let b = [one, eps / v(2.0), eps / v(2.0)];
    witness(fixture, "fold", inputs(one), &b, &[one; 3], &[zero; 3], v(32.0), &b);
}

#[uzu_test]
fn half_narrowing_witnesses() {
    let fixture = KernelFixture::new();
    half_witnesses::<f16>(&fixture);
    half_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Classes in every type: a NaN input; an infinite state times a decay of exactly zero (dt_raw +inf for F32, 32 for F16
/// and 104 for BF16, whose Exp results round to zeros); a product overflowing; opposite infinities in the fold.
fn class_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let v = |value: f32| T::from(value).unwrap();
    let zero_decay = v(match T::data_type() {
        DataType::F32 => f32::INFINITY,
        DataType::F16 => 32.0,
        _ => 104.0,
    });
    let (zero, one, inf, nan, max) = (v(0.0), v(1.0), T::infinity(), T::nan(), T::max_value());
    witness(fixture, "NaN x", [nan, zero_decay, zero, one], &[one], &[one], &[one], nan, &[nan]);
    witness(fixture, "inf state, zero decay", [one, zero_decay, zero, one], &[one], &[one], &[inf], nan, &[nan]);
    let two = [v(2.0), zero_decay, zero, one];
    witness(fixture, "overflow", two, &[max], &[one], &[zero], inf, &[inf]);
    witness(fixture, "inf - inf", two, &[max, -max], &[one; 2], &[zero; 2], nan, &[inf, -inf]);
}

/// The classes, an F32 subnormal state kept through the fold, and an F16 subnormal state stored.
#[uzu_test]
fn class_and_subnormal_witnesses() {
    let fixture = KernelFixture::new();
    class_witnesses::<f32>(&fixture);
    class_witnesses::<f16>(&fixture);
    class_witnesses::<bf16>(&fixture);
    let p = |e: i32| 2f32.powi(e);
    let state = [p(-75), f32::INFINITY, 0.0, p(-27)];
    witness(&fixture, "F32 subnormal", state, &[p(-74)], &[p(60)], &[0.0], p(-117), &[f32::from_bits(1)]);
    let h = |value: f32| f16::from_f32(value);
    let state = [h(p(-12)), h(-104.0), f16::ZERO, h(32.0)];
    witness(&fixture, "F16 subnormal", state, &[h(p(-12))], &[f16::ONE], &[f16::ZERO], h(p(-19)), &[f16::from_bits(1)]);
    fixture.assert_clean();
}

/// Up to 256 witnesses that `stage` is rounded to T, from 16-bit (dt_raw, s, z), x = c = 1, b = d = +0: y's shader and
/// CPU sets are one same value, the shader's with `stage` in FP32 another. Returns the batched case and y.
fn narrowing_witnesses<T: ArrayElement + Float>(stage: usize) -> ([u32; 5], [Vec<T>; 7], Vec<T>) {
    let v = |value: f32| T::from(value).unwrap();
    let grid = |low: f32, high: f32| {
        let values = (0..=u16::MAX).map(|bits| bytemuck::pod_read_unaligned::<T>(&bits.to_ne_bytes()));
        values.filter(move |value| (low..high).contains(&value.to_f32().unwrap().abs()))
    };
    let candidates: Vec<[T; 3]> = match stage {
        DT => grid(0.125, 8.0).map(|dt| [dt, v(1.0), v(32.0)]).collect(),
        DECAY => grid(0.125, 8.0).flat_map(|dt| [1.25, 1.5, 1.75, 3.0].map(|s| [dt, v(s), v(32.0)])).collect(),
        _ => grid(1.0, 4.0)
            .filter(|s| s.is_sign_positive())
            .flat_map(|s| [0.5, 1.0, 2.0, 3.0].map(|z| [v(-104.0), s, v(z)]))
            .collect(),
    };
    let y = |shader: bool, wide: usize, [dt, s, z]: [T; 3]| {
        let f = |value: T| value.to_f64().unwrap();
        single(update::<T>(shader, wide, [1.0, f(dt), 0.0, f(z)], &[(0.0, 1.0)], &[point(f(s))]).1)
    };
    let bits = |value: Option<f64>| value.map(f64::to_bits);
    let witnesses = candidates
        .into_iter()
        .filter(|&candidate| {
            let expected = bits(y(true, NARROWED, candidate));
            expected.is_some()
                && bits(y(false, NARROWED, candidate)) == expected
                && bits(y(true, stage, candidate)).is_some_and(|wide| Some(wide) != expected)
        })
        .take(256)
        .collect::<Vec<_>>();
    let (count, column) = (witnesses.len(), |k: usize| witnesses.iter().map(|witness| witness[k]).collect::<Vec<_>>());
    let (ones, zeros) = (vec![v(1.0); count], vec![v(0.0); count]);
    let inputs = [ones.clone(), column(0), zeros.clone(), ones, zeros, column(2), column(1)];
    let expected = witnesses.iter().map(|&witness| T::from(y(true, NARROWED, witness).unwrap()).unwrap()).collect();
    ([1, count as u32, 1, 1, 1], inputs, expected)
}

/// The bounded searches find witnesses that dt, the decay and the gate are each rounded to T in both 16-bit types, and
/// the CPU kernel gives each witness's value (CPU only).
#[uzu_test]
fn narrowing_witnesses_exist() {
    fn count<T: ArrayElement + Float + Debug + Default>() {
        for (stage, name) in [(DT, "dt"), (DECAY, "decay"), (GATE, "gate")] {
            let (dims, inputs, expected) = narrowing_witnesses::<T>(stage);
            eprintln!("{:?} {name}: {} narrowing witnesses", T::data_type(), expected.len());
            assert!(!expected.is_empty(), "{:?} {name}: the bounded grid holds no witness", T::data_type());
            let ([y, _], _) = cpu_update(dims, BATCHED, &inputs, true, 1);
            assert_same_bits(&expected, &y, &format!("{:?} {name} CPU", T::data_type()));
        }
    }
    count::<f16>();
    count::<bf16>();
}

/// Vulkan gives every narrowing witness's value.
#[uzu_test]
fn narrowing_witnesses_match_vulkan() {
    fn matches<T: ArrayElement + Float + Debug>(fixture: &KernelFixture) {
        let kernel = SSDUpdateVulkanKernel::new(&fixture.context, T::data_type(), true).expect("SSDUpdate");
        for stage in [DT, DECAY, GATE] {
            let (dims, inputs, expected) = narrowing_witnesses::<T>(stage);
            let ([y, _], _) = gpu_update(fixture, &kernel, dims, BATCHED, &inputs, true, 1, false);
            assert_same_bits(&expected, &y, &format!("{:?} stage {stage} Vulkan", T::data_type()));
        }
    }
    let fixture = KernelFixture::new();
    matches::<f16>(&fixture);
    matches::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Two in-place dispatches in one command buffer continue one state, as two CPU submissions do.
#[uzu_test]
fn chained_dispatches_match_cpu() {
    let fixture = KernelFixture::new();
    let kernel = SSDUpdateVulkanKernel::new(&fixture.context, DataType::F32, true).expect("SSDUpdate");
    let inputs = inputs::<f32>(PADDED_DIMS, PADDED);
    let (cpu, _) = cpu_update(PADDED_DIMS, PADDED, &inputs, true, 2);
    let (gpu, _) = gpu_update(&fixture, &kernel, PADDED_DIMS, PADDED, &inputs, true, 2, false);
    check(PADDED_DIMS, PADDED, &inputs, true, 2, &cpu, &gpu, "two chained dispatches");
    fixture.assert_clean();
}

/// Uploads snapshot the stride arrays: two out-of-place dispatches recorded from the same four arrays, holding PADDED and
/// then DENSE, which change again before submission, each match the CPU with its saved layout.
#[uzu_test]
fn uploads_snapshot_strides() {
    let fixture = KernelFixture::new();
    let kernel = SSDUpdateVulkanKernel::new(&fixture.context, DataType::F32, false).expect("SSDUpdate");
    let inputs = inputs::<f32>(PADDED_DIMS, PADDED);
    let buffers = [PADDED, DENSE].map(|_| gpu_buffers(&fixture, &inputs, false));
    let (mut x, mut dt, mut cb, mut state) = split(&PADDED);
    let mut encoding = fixture.encoding();
    // SAFETY: both layouts index within PADDED's spans, and each dispatch writes its own buffers.
    unsafe { encode(&kernel, &buffers[0], PADDED_DIMS, (&x, &dt, &cb, &state), false, &mut encoding) };
    (x, dt, cb, state) = split(&DENSE);
    unsafe { encode(&kernel, &buffers[1], PADDED_DIMS, (&x, &dt, &cb, &state), false, &mut encoding) };
    for strides in [&mut x[..], &mut dt[..], &mut cb[..], &mut state[..]] {
        strides.fill(0);
    }
    KernelFixture::complete(encoding);
    for (layout, buffers) in [PADDED, DENSE].into_iter().zip(&buffers) {
        let (cpu, _) = cpu_update(PADDED_DIMS, layout, &inputs, false, 1);
        // SAFETY: the command buffer has completed.
        let gpu = unsafe { gpu_results(buffers, &inputs) };
        check(PADDED_DIMS, layout, &inputs, false, 1, &cpu, &gpu, &format!("uploaded {layout:?}"));
    }
    fixture.assert_clean();
}

/// No work at u32::MAX extents, state size and strides, also with group_size 0: nothing is recorded or changed.
#[uzu_test]
fn zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let fill = sentinel::<f32>();
    let empty = fixture.guarded::<f32>(&[], fill);
    let buffers = [(); 9].map(|_| empty.clone());
    let kernel = SSDUpdateVulkanKernel::new(&fixture.context, DataType::F32, true).expect("SSDUpdate");
    let m = u32::MAX;
    let (x, dt, cb, state) = split(&[m; 12]);
    let mut encoding = fixture.encoding();
    for dims in [[0, m, m, 1, m], [m, 0, m, 0, m], [m, m, 0, 0, m]] {
        // SAFETY: without work nothing is indexed or recorded.
        unsafe { encode(&kernel, &buffers, dims, (&x, &dt, &cb, &state), true, &mut encoding) };
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, fill, &[], "empty") };
    fixture.assert_clean();
}

/// group_size 0 with work fails the encode precondition before anything is recorded; the CPU kernel's division panics.
#[uzu_test]
fn rejects_group_size_zero() {
    let fixture = KernelFixture::new();
    let kernel = SSDUpdateVulkanKernel::new(&fixture.context, DataType::F32, true).expect("SSDUpdate");
    let (dims, strides, inputs) = ([1, 1, 1, 0, 1], [1; 12], [(); 7].map(|_| vec![1.0f32]));
    let (buffers, (x, dt, cb, state)) = (gpu_buffers(&fixture, &inputs, true), split(&strides));
    let mut encoding = fixture.encoding();
    // SAFETY: the precondition panics before anything is recorded.
    let gpu = catch_unwind(AssertUnwindSafe(|| unsafe {
        encode(&kernel, &buffers, dims, (&x, &dt, &cb, &state), true, &mut encoding)
    }));
    let message = gpu.expect_err("group_size 0 encoded").downcast::<String>().expect("panic message");
    assert!(message.contains("precondition group_size > 0"), "{message}");
    let cpu = catch_unwind(AssertUnwindSafe(|| cpu_update(dims, strides, &inputs, true, 1)));
    assert!(cpu.is_err(), "the CPU kernel accepted group_size 0");
    KernelFixture::complete(encoding);
    fixture.assert_clean();
}

/// Run alone, without sync validation: `... ssd_update_test::throughput -- --ignored --nocapture`. Times one decode step
/// at the illustrative shape [B 1, H 128, Dh 64, group_size 16, N 128], not tied to a model configuration, contiguous,
/// from nonzero eighths: GPU and wall medians of 10 Vulkan submissions after 3 warm-up ones and the CPU kernel's wall
/// median. The buffers after all 13 submissions must be in the oracle iterated as often. F32 is Mamba2's inner type;
/// F16 and BF16 measure the kernel in other dtypes.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Debug + Default>(
        fixture: &KernelFixture,
        reverse: bool,
    ) {
        let dims = [1, 128, 64, 16, 128];
        let strides = contiguous(dims);
        let [x, dt, bc, d, state] = spans(dims, strides);
        let lens = [x, dt, bc, bc, d, x, state];
        let inputs =
            std::array::from_fn(|i| values::<T>(lens[i], i, 0).iter().map(|&v| v + T::from(0.0625).unwrap()).collect());
        for in_place in [[false, true], [true, false]][usize::from(reverse)] {
            let kernel = SSDUpdateVulkanKernel::new(&fixture.context, T::data_type(), in_place).expect("SSDUpdate");
            let label = format!("SSDUpdate {:?} {dims:?} in place {in_place}", T::data_type());
            let (cpu, cpu_times) = cpu_update(dims, strides, &inputs, in_place, 13);
            let (gpu, times) = gpu_update(fixture, &kernel, dims, strides, &inputs, in_place, 1, true);
            check(dims, strides, &inputs, in_place, 13, &cpu, &gpu, &label);
            let ((gpu, wall), mut cpu) = (times.expect("timed"), cpu_times[3..].to_vec());
            cpu.sort();
            eprintln!("{label}: GPU {gpu:?}, wall {wall:?}; CPU wall {:?}", cpu[cpu.len() / 2]);
        }
    }
    let fixture = KernelFixture::new();
    // Two rounds in opposite orders, both reported: round 0 runs F32, F16, BF16, each out of place then in place; round 1
    // runs BF16, F16, F32, each in place then out of place. No clock or thermal state is assumed for either.
    eprintln!("SSDUpdate throughput round 0");
    measure::<f32>(&fixture, false);
    measure::<f16>(&fixture, false);
    measure::<bf16>(&fixture, false);
    eprintln!("SSDUpdate throughput round 1");
    measure::<bf16>(&fixture, true);
    measure::<f16>(&fixture, true);
    measure::<f32>(&fixture, true);
    fixture.assert_clean();
}
