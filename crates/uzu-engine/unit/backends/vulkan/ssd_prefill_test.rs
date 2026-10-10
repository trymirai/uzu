use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    mem::size_of,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Duration,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    NAN, NEG_INF, NEG_ZERO, POS_INF, POS_ZERO, add, arg, assert_same_bits, bounds, conv1d_values as values, cpu_buffer,
    cpu_submissions, decay, kernel_fixture::KernelFixture, mul, oracle, point, round, single, union,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::ActivationType,
            kernel::{SSDPrefill64Kernel, SSDPrefillKernel},
        },
        cpu::Cpu,
        vulkan::{SSDPrefill64VulkanKernel, SSDPrefillVulkanKernel, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// SSDPrefill64's generated encode precondition, from its source attribute.
const GUARD: &str =
    "SSDPrefill64: precondition suffix_len == 0 || num_heads == 0 || head_dim == 0 || state_size == 64 violated";

fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

/// Contiguous [x; dt; B and C; state] strides of [Q, H, Dh, N, group_size], whose last group may be partial.
fn contiguous([_, heads, head_dim, n, group]: [u32; 5]) -> [u32; 11] {
    let groups = heads.div_ceil(group.max(1));
    [heads * head_dim, head_dim, 1, heads, 1, groups * n, n, 1, head_dim * n, n, 1]
}

/// [x; dt; B and C; state] strides as the kernels take them.
fn split(s: &[u32; 11]) -> ([u32; 3], [u32; 2], [u32; 3], [u32; 3]) {
    (s[..3].try_into().unwrap(), s[3..5].try_into().unwrap(), s[5..8].try_into().unwrap(), s[8..].try_into().unwrap())
}

/// The token ranges of `tokens` dispatched one after another, cut at `splits`.
fn segments(
    tokens: u32,
    splits: &[u32],
) -> Vec<Range<u32>> {
    let cuts = [&[0], splits, &[tokens]].concat();
    cuts.windows(2).map(|pair| pair[0]..pair[1]).collect()
}

/// Every (head, element) with work, in order, as (head, its state row, [x (z, y), dt_raw, B and C row] of each token),
/// after asserting that the y and state elements written are distinct.
fn items(
    dims: [u32; 5],
    strides: [u32; 11],
) -> Vec<(usize, usize, Vec<[usize; 3]>)> {
    let ([tokens, heads, elements, n, group], s) = (dims.map(|v| v as usize), strides.map(|v| v as usize));
    let items = (0..heads * usize::from(tokens > 0))
        .flat_map(|h| {
            (0..elements).map(move |e| {
                let token = |t: usize| {
                    [t * s[0] + h * s[1] + e * s[2], t * s[3] + h * s[4], t * s[5] + h / group.max(1) * s[6]]
                };
                (h, h * s[8] + e * s[9], (0..tokens).map(token).collect::<Vec<_>>())
            })
        })
        .collect::<Vec<_>>();
    let ys = items.iter().flat_map(|(_, _, tokens)| tokens.iter().map(|token| token[0])).collect::<HashSet<_>>();
    let states = items.iter().flat_map(|&(_, row, _)| (0..n).map(move |i| row + i * s[10])).collect::<HashSet<_>>();
    assert_eq!((ys.len(), states.len()), (items.len() * tokens, items.len() * n), "{dims:?} {strides:?} writes alias");
    items
}

/// [x, dt_raw, b, c, d, z, state] of `fill(array, len)` over the valid spans: one past the last element addressed, none
/// where nothing is (dt_raw, B, C and the state with N 0).
fn inputs<T: ArrayElement + Float>(
    dims: [u32; 5],
    strides: [u32; 11],
    fill: impl Fn(usize, usize) -> Vec<T>,
) -> [Vec<T>; 7] {
    let (items, inner, s) = (items(dims, strides), dims[3].saturating_sub(1) as usize, strides.map(|v| v as usize));
    let last = |values: &mut dyn Iterator<Item = usize>| values.max().map_or(0, |i| i + 1);
    let tokens = || items.iter().flat_map(|(_, _, tokens)| tokens).filter(|_| dims[3] > 0);
    let x = last(&mut items.iter().flat_map(|(_, _, tokens)| tokens).map(|token| token[0]));
    let bc = last(&mut tokens().map(|token| token[2] + inner * s[7]));
    let state = last(&mut items.iter().filter(|_| dims[3] > 0).map(|item| item.1 + inner * s[10]));
    let lens =
        [x, last(&mut tokens().map(|token| token[1])), bc, bc, last(&mut items.iter().map(|item| item.0)), x, state];
    std::array::from_fn(|array| fill(array, lens[array]))
}

/// The executed CPU SSDPrefill, or SSDPrefill64 when `special64`, in `submissions` timed submissions per range of
/// `segments` on the same buffers, each from the previous state and its arrays from its first token (the token strides
/// outermost): [y from sentinels, state] and each submission's wall time.
fn cpu_prefill<T: ArrayElement + Float + Default>(
    dims: [u32; 5],
    strides: [u32; 11],
    inputs: &[Vec<T>; 7],
    special64: bool,
    splits: &[u32],
    submissions: usize,
) -> ([Vec<T>; 2], Vec<Duration>) {
    let ([_, heads, dh, n, group], (xs, dts, cbs, ss)) = (dims, split(&strides));
    let context = create_context::<Cpu>();
    let generic = (!special64).then(|| {
        <<Cpu as Backend>::Kernels as Kernels>::SSDPrefillKernel::new(&context, T::data_type(), n)
            .expect("CPU SSDPrefill")
    });
    let special = special64.then(|| {
        <<Cpu as Backend>::Kernels as Kernels>::SSDPrefill64Kernel::new(&context, T::data_type())
            .expect("CPU SSDPrefill64")
    });
    let (mut outputs, mut times) = ([vec![sentinel::<T>(); inputs[0].len()], inputs[6].clone()], Vec::new());
    for tokens in segments(dims[0], splits) {
        let starts = [xs[0], dts[0], cbs[0], cbs[0], 0, xs[0]].map(|stride| tokens.start as usize * stride as usize);
        let at = |values: &[T], start: usize| values[start.min(values.len())..].to_vec();
        let [x, dt, b, c, d, z] = std::array::from_fn(|k| cpu_buffer(&context, &at(&inputs[k], starts[k])));
        let (mut state, mut y) = (cpu_buffer(&context, &outputs[1]), cpu_buffer(&context, &at(&outputs[0], starts[0])));
        let q = tokens.len() as u32;
        times.extend(cpu_submissions(&context, submissions, |e| {
            let (s, o) = (&mut state, &mut y);
            if let Some(kernel) = &generic {
                kernel.encode(&x, &dt, &b, &c, &d, &z, s, o, q, group, &xs, &dts, &cbs, &ss, heads, dh, e);
            } else if let Some(kernel) = &special {
                kernel.encode(&x, &dt, &b, &c, &d, &z, s, o, q, group, n, &xs, &dts, &cbs, &ss, heads, dh, e);
            }
        }));
        let start = starts[0].min(outputs[0].len());
        outputs[0][start..].copy_from_slice(&buffer_prefix_to_vec::<Cpu, T>(&y, inputs[0].len() - start));
        outputs[1] = buffer_prefix_to_vec::<Cpu, T>(&state, outputs[1].len());
    }
    (outputs, times)
}

/// The Vulkan counterpart of `cpu_prefill`: one dispatch per range in one command buffer, over guarded ranges from the
/// range's first token, or when `timed` that command buffer in each of `median_times`' submissions on the same buffers.
/// Returns [y, state] after asserting the read-only inputs and every guard unchanged, and the GPU and wall medians.
fn gpu_prefill<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    dims: [u32; 5],
    strides: [u32; 11],
    inputs: &[Vec<T>; 7],
    special64: bool,
    splits: &[u32],
    timed: bool,
) -> ([Vec<T>; 2], Option<(Duration, Duration)>) {
    let ([_, heads, dh, n, group], (xs, dts, cbs, ss)) = (dims, split(&strides));
    let (fill, size) = (sentinel::<T>(), size_of::<T>() as u64);
    let ys = vec![fill; inputs[0].len()];
    let buffers: [_; 8] = std::array::from_fn(|k| fixture.guarded(inputs.get(k).unwrap_or(&ys), fill));
    let generic =
        (!special64).then(|| SSDPrefillVulkanKernel::new(&fixture.context, T::data_type(), n).expect("SSDPrefill"));
    let special =
        special64.then(|| SSDPrefill64VulkanKernel::new(&fixture.context, T::data_type()).expect("SSDPrefill64"));
    let mut record = |e: &mut VkCommandBufferEncoding| {
        for tokens in segments(dims[0], splits) {
            let starts = [xs[0], dts[0], cbs[0], cbs[0], 0, xs[0], 0, xs[0]];
            let [x, dt, b, c, d, z, state, y] = std::array::from_fn(|k| {
                let (buffer, range) = &buffers[k];
                (
                    buffer,
                    (range.start + u64::from(tokens.start) * u64::from(starts[k]) * size).min(range.end)..range.end,
                )
            });
            let q = tokens.len() as u32;
            // SAFETY: `inputs` sized each range to its valid span, asserting the writes distinct, and each dispatch's
            // ranges start at its first token.
            unsafe {
                if let Some(kernel) = &generic {
                    kernel.encode(x, dt, b, c, d, z, state, y, q, group, &xs, &dts, &cbs, &ss, heads, dh, e);
                } else if let Some(kernel) = &special {
                    kernel.encode(x, dt, b, c, d, z, state, y, q, group, n, &xs, &dts, &cbs, &ss, heads, dh, e);
                }
            }
        }
    };
    let times = timed.then(|| fixture.median_times(&mut record));
    if !timed {
        let mut encoding = fixture.encoding();
        record(&mut encoding);
        KernelFixture::complete(encoding);
    }
    for (k, name) in ["x", "dt_raw", "b", "c", "d", "z"].into_iter().enumerate() {
        // SAFETY: the command buffer has completed.
        unsafe { KernelFixture::assert_unchanged(&buffers[k], fill, &inputs[k], name) };
    }
    // SAFETY: every command buffer has completed.
    ([7, 6].map(|k| unsafe { KernelFixture::read_guarded(&buffers[k], fill) }), times)
}

/// Sets of y for each token and of the final state row of one (head, element) from the row's sets, as the shader or the
/// CPU stages them: per token, in FP32, decay = exp(-softplus(dt_raw)), s_i = s_i decay + B_i x for i ascending and the
/// dot from +0 over s_i C_i, then y = T((dot + d x) gate) with the gate SiLU rounded to T; the row is stored in T.
/// `stage` stages one value as the contract does not: "dt" or "decay" rounded to T, "gate" kept in FP32.
fn prefill<T: ArrayElement + Float>(
    shader: bool,
    stage: &str,
    d: f64,
    tokens: &[([f64; 3], Vec<(f64, f64)>)],
    mut row: Vec<((f64, f64), u8)>,
) -> (Vec<((f64, f64), u8)>, Vec<((f64, f64), u8)>) {
    let ys = tokens
        .iter()
        .map(|([x, dt_raw, z], bc)| {
            let (x, gate) = (point(*x), oracle(*z, ActivationType::SILU).0);
            let gate = [bounds::<T>(gate), bounds::<f32>(gate)][usize::from(stage == "gate")];
            let mut dot = point(0.0);
            if !bc.is_empty() {
                let dt = oracle(*dt_raw, ActivationType::SOFTPLUS).0;
                let decay = match stage {
                    "dt" => decay::<T, f32>(bounds::<T>(dt), shader),
                    "decay" => decay::<f32, T>(bounds::<f32>(dt), shader),
                    _ => decay::<f32, f32>(bounds::<f32>(dt), shader),
                };
                for (s, &(b, c)) in row.iter_mut().zip(bc) {
                    *s = add::<f32>(mul::<f32>(*s, decay), mul::<f32>(point(b), x));
                    dot = add::<f32>(dot, mul::<f32>(*s, point(c)));
                }
            }
            mul::<T>(add::<f32>(dot, mul::<f32>(point(d), x)), gate)
        })
        .collect();
    (ys, row.into_iter().map(|s| mul::<T>(s, point(1.0))).collect())
}

/// [y, state] of the CPU and of Vulkan SSDPrefill, each dispatched once over the ranges of `splits`, after `check`.
fn run<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    dims: [u32; 5],
    strides: [u32; 11],
    inputs: &[Vec<T>; 7],
    splits: &[u32],
    label: &str,
) -> [[Vec<T>; 2]; 2] {
    let cpu = cpu_prefill(dims, strides, inputs, false, splits, 1).0;
    let outputs = [cpu, gpu_prefill(fixture, dims, strides, inputs, false, splits, false).0];
    check(dims, strides, inputs, splits, 1, &outputs, label);
    outputs
}

/// Checks [y, state] of the CPU and Vulkan after `submissions` repetitions of the ranges of `splits`, each from the
/// previous one's state, against their own oracle with the state stored in T between ranges: written elements are
/// members (a set holds a value when their union is the set) of the last repetition's sets, all others keep their
/// initial bits. Returns the largest relative width of a finite set an element was checked against.
fn check<T: ArrayElement + Float + Debug>(
    dims: [u32; 5],
    strides: [u32; 11],
    inputs: &[Vec<T>; 7],
    splits: &[u32],
    submissions: usize,
    outputs: &[[Vec<T>; 2]; 2],
    label: &str,
) -> f64 {
    assert!(submissions == 1 || splits.is_empty(), "{label}: repeated submissions dispatch one range");
    let (items, n, s) = (items(dims, strides), dims[3] as usize, strides.map(|v| v as usize));
    let value = |array: usize, index: usize| inputs[array][index].to_f64().unwrap();
    let (mut violations, mut width) = (0, 0f64);
    for (shader, results) in [(false, &outputs[0]), (true, &outputs[1])] {
        let mut owned = [HashMap::new(), HashMap::new()];
        for (h, row, tokens) in &items {
            let mut state = (0..n).map(|i| point(value(6, row + i * s[10]))).collect::<Vec<_>>();
            for range in (0..submissions).flat_map(|_| segments(dims[0], splits)) {
                let tokens = &tokens[range.start as usize..range.end as usize];
                let arguments = tokens
                    .iter()
                    .map(|&[x, dt, cb]| {
                        let dt = inputs[1].get(dt).map_or(f64::NAN, |dt| dt.to_f64().unwrap());
                        let bc = (0..n).map(|i| (value(2, cb + i * s[7]), value(3, cb + i * s[7]))).collect();
                        ([value(0, x), dt, value(5, x)], bc)
                    })
                    .collect::<Vec<_>>();
                let (ys, next) = prefill::<T>(shader, "", value(4, *h), &arguments, state);
                owned[0].extend(tokens.iter().map(|token| token[0]).zip(ys));
                state = next;
            }
            owned[1].extend((0..n).map(|i| row + i * s[10]).zip(state));
        }
        let initial = [vec![sentinel::<T>(); inputs[0].len()], inputs[6].clone()];
        for (k, (results, initial)) in results.iter().zip(initial).enumerate() {
            assert_eq!(results.len(), initial.len(), "{label}: length");
            for (index, (&actual, initial)) in results.iter().zip(initial).enumerate() {
                let valid = match owned[k].get(&index) {
                    None => bytemuck::bytes_of(&actual) == bytemuck::bytes_of(&initial),
                    Some(&set) => {
                        let ((lo, hi), _) = set;
                        width = width.max(if lo <= hi {
                            (hi - lo) / lo.abs().max(hi.abs())
                        } else {
                            0.0
                        });
                        union(set, point(actual.to_f64().unwrap())) == set
                    },
                };
                violations += usize::from(!valid);
                if !valid && violations <= 5 {
                    let (side, name) = (["CPU", "Vulkan"][usize::from(shader)], ["y", "state"][k]);
                    eprintln!("{label}: {side} {name} {index}: {actual:?} outside {:?}", owned[k].get(&index));
                }
            }
        }
    }
    assert_eq!(violations, 0, "{label}: {violations} elements outside the oracle");
    width
}

/// One contiguous case over the ranges of `splits`, array k's element i `arrays[k][i % len]`: the CPU and Vulkan each
/// give exactly `expected` [y, state] (any NaN for a NaN) and lie in their oracles.
fn witness<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    dims: [u32; 5],
    arrays: [&[f64]; 7],
    splits: &[u32],
    expected: [&[f64]; 2],
    label: &str,
) {
    let (strides, label) = (contiguous(dims), format!("{:?} {label}", T::data_type()));
    let fill = |k: usize, len| (0..len).map(|i| T::from(arrays[k][i % arrays[k].len()]).unwrap()).collect();
    let outputs = run(fixture, dims, strides, &inputs::<T>(dims, strides, fill), splits, &label);
    let expected = expected.concat().iter().map(|&value| T::from(value).unwrap()).collect::<Vec<_>>();
    for (side, outputs) in ["CPU", "Vulkan"].into_iter().zip(&outputs) {
        KernelFixture::assert_bits(&expected, &outputs.concat(), &format!("{label}: {side} [y, state]"));
    }
}

/// [Q, H, Dh, N, group_size] with strides: contiguous with N 0 to 4096, Dh tails across 64, partial last groups and
/// group_size 0; N 0 beside u32::MAX strides of the unaddressed dt_raw, B, C and state; the CPU kernel's padded,
/// token-broadcast and interleaved layouts; u32::MAX strides of unit extents; padded N 64.
fn cases() -> Vec<([u32; 5], [u32; 11])> {
    let shapes = [
        [3, 2, 3, 0, 1],
        [2, 1, 2, 1, 1],
        [2, 3, 65, 63, 2],
        [2, 2, 64, 64, 0],
        [1, 3, 63, 65, 2],
        [2, 2, 2, 128, 2],
        [1, 1, 2, 256, 1],
        [2, 1, 1, 1024, 1],
        [1, 1, 1, 4096, 1],
        [3, 2, 129, 4, 1],
    ];
    let m = u32::MAX;
    let mut cases = shapes.map(|dims| (dims, contiguous(dims))).to_vec();
    cases.extend([
        ([2, 2, 2, 0, 1], [4, 2, 1, m, m, m, m, m, m, m, m]),
        ([3, 3, 2, 4, 2], [16, 5, 2, 4, 1, 20, 9, 2, 30, 13, 3]),
        ([2, 2, 2, 4, 1], [8, 4, 1, 0, 1, 0, 4, 1, 8, 4, 1]),
        ([2, 2, 3, 2, 2], [8, 1, 2, 2, 1, 2, 2, 1, 1, 2, 6]),
        ([1, 1, 3, 2, 1], [m, m, 1, m, m, m, m, 1, m, 2, 1]),
        ([2, 3, 2, 64, 2], [16, 5, 2, 4, 1, 300, 140, 2, 300, 140, 2]),
    ]);
    cases
}

/// Every case against the oracle from finite eighths, with the specials in every fifth element up to N 128; at N 64
/// SSDPrefill64 gives SSDPrefill's bits on each side.
#[uzu_test]
fn matches_cpu_all_types() {
    fn matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        for (dims, strides) in cases() {
            let label = format!("{:?} {dims:?} {strides:?}", T::data_type());
            let every = [5, 0][usize::from(dims[3] > 128)];
            let inputs = inputs::<T>(dims, strides, |k, len| values(len, k, every));
            let generic = run(fixture, dims, strides, &inputs, &[], &label);
            if dims[3] == 64 {
                let special = [
                    cpu_prefill(dims, strides, &inputs, true, &[], 1).0,
                    gpu_prefill(fixture, dims, strides, &inputs, true, &[], false).0,
                ];
                for ((side, generic), special) in ["CPU", "Vulkan"].into_iter().zip(&generic).zip(special) {
                    assert_same_bits(&generic.concat(), &special.concat(), &format!("{label}: {side} SSDPrefill64"));
                }
            }
        }
    }
    let fixture = KernelFixture::new();
    matches::<f32>(&fixture);
    matches::<f16>(&fixture);
    matches::<bf16>(&fixture);
    fixture.assert_clean();
}

/// group_size 0 reads B and C as group_size 1 does.
#[uzu_test]
fn group_size_zero_is_one() {
    let fixture = KernelFixture::new();
    let strides = contiguous([3, 3, 2, 4, 1]);
    let inputs = inputs::<f32>([3, 3, 2, 4, 1], strides, |k, len| values(len, k, 0));
    let [zero, one] =
        [0, 1].map(|group| gpu_prefill(&fixture, [3, 3, 2, 4, group], strides, &inputs, false, &[], false).0);
    KernelFixture::assert_bits(&one.concat(), &zero.concat(), "[y, state]");
    fixture.assert_clean();
}

/// No work (Q, H or Dh 0) at u32::MAX extents and strides, also with group_size 0, and SSDPrefill64 at the irrelevant
/// state sizes 63 and u32::MAX: nothing is recorded or changed.
#[uzu_test]
fn zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (fill, m) = (sentinel::<f32>(), u32::MAX);
    let empty = fixture.guarded::<f32>(&[], fill);
    let generic = SSDPrefillVulkanKernel::new(&fixture.context, DataType::F32, 7).expect("SSDPrefill");
    let special = SSDPrefill64VulkanKernel::new(&fixture.context, DataType::F32).expect("SSDPrefill64");
    let (e, s3, s2) = (|| arg(&empty), &[m; 3], &[m; 2]);
    let mut encoding = fixture.encoding();
    for [q, h, dh, group] in [[0, m, m, 1], [m, 0, m, 0], [m, m, 0, 2]] {
        let w = &mut encoding;
        // SAFETY: without work nothing is indexed or recorded.
        unsafe {
            generic.encode(e(), e(), e(), e(), e(), e(), e(), e(), q, group, s3, s2, s3, s3, h, dh, w);
            for n in [63, m] {
                special.encode(e(), e(), e(), e(), e(), e(), e(), e(), q, group, n, s3, s2, s3, s3, h, dh, w);
            }
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, fill, &[], "empty") };
    fixture.assert_clean();
}

/// SSDPrefill64 with work at N 63 and 65 fails its exact encode precondition before anything is recorded or written;
/// the CPU kernel's assert panics on its command buffer's thread, surfacing as the failed wait.
#[uzu_test]
fn prefill64_rejects_other_state_sizes() {
    let fixture = KernelFixture::new();
    let kernel = SSDPrefill64VulkanKernel::new(&fixture.context, DataType::F32).expect("SSDPrefill64");
    let fill = sentinel::<f32>();
    for n in [63, 65] {
        let (dims, strides) = ([1, 1, 1, n, 1], contiguous([1, 1, 1, n, 1]));
        let (xs, dts, cbs, ss) = split(&strides);
        let inputs = inputs::<f32>(dims, strides, |k, len| values(len, k, 0));
        let y = vec![fill; inputs[0].len()];
        let buffers: [_; 8] = std::array::from_fn(|k| fixture.guarded(inputs.get(k).unwrap_or(&y), fill));
        let [x, dt, b, c, d, z, state, out] = buffers.each_ref().map(arg);
        let mut encoding = fixture.encoding();
        // SAFETY: the precondition panics before anything is recorded.
        let gpu = catch_unwind(AssertUnwindSafe(|| unsafe {
            kernel.encode(x, dt, b, c, d, z, state, out, 1, 1, n, &xs, &dts, &cbs, &ss, 1, 1, &mut encoding)
        }));
        let message = gpu.expect_err("SSDPrefill64 encoded").downcast::<String>().expect("panic message");
        assert_eq!(*message, GUARD, "N {n}");
        KernelFixture::complete(encoding);
        for (k, payload) in inputs.iter().chain([&y]).enumerate() {
            // SAFETY: the command buffer has completed.
            unsafe { KernelFixture::assert_unchanged(&buffers[k], fill, payload, &format!("N {n} buffer {k}")) };
        }
        let cpu = catch_unwind(AssertUnwindSafe(|| cpu_prefill(dims, strides, &inputs, true, &[], 1)));
        let message = cpu.expect_err("CPU SSDPrefill64 accepted").downcast::<String>().expect("panic message");
        assert_eq!(*message, "called `Result::unwrap()` on an `Err` value: CommandBufferExecutionFailed(RecvError)");
    }
    fixture.assert_clean();
}

/// The oracle's own staging: an F16 state of 1 + 2^-10 retained and its y; the signed zeros of s decay + B x at decay
/// +0; an F16 store overflowing; infinite and NaN state classes; a state set of an infinity and -0.
#[uzu_test]
fn oracle_boundaries() {
    let p = |e: i32| 2f64.powi(e);
    let unit = |x: f64, b: f64, state: ((f64, f64), u8)| {
        prefill::<f16>(true, "", 0.0, &[([x, -104.0, 32.0], vec![(b, 1.0)])], vec![state])
    };
    let tokens = vec![([1.0, -104.0, 32.0], vec![(p(-11), 1.0)]); 2];
    let (ys, state) = prefill::<f16>(true, "", 0.0, &tokens, vec![point(1.0)]);
    assert_eq!((ys[1], single(state[0])), (round::<f16>((32.03125, 32.03125)), Some(1.0 + p(-10))));
    let zero = |b: f64, s: f64| {
        prefill::<f32>(true, "", 0.0, &[([1.0, f64::INFINITY, 0.0], vec![(b, 1.0)])], vec![point(s)]).1[0]
    };
    let zeros = [zero(-0.0, -1.0), zero(-0.0, 1.0), zero(0.0, -1.0)];
    assert_eq!(zeros, [NEG_ZERO, POS_ZERO, POS_ZERO].map(|class| (point(f64::NAN).0, class)));
    assert_eq!(unit(1.0, 60000.0, point(60000.0)).1[0].1, POS_INF);
    assert_eq!(unit(f64::NEG_INFINITY, 1.0, point(1.0)).1[0].1, NEG_INF);
    let (_, state) = unit(1.0, 1.0, union(point(f64::INFINITY), point(-0.0)));
    assert_eq!(state[0], union(point(1.0), point(f64::INFINITY)));
    assert_eq!(unit(f64::NAN, 1.0, point(1.0)).1[0].1, NAN);
}

/// The CPU's retention witnesses: Q 2 with decay 1, x = C = state = 1, d 0, gate 32 and B half of 1's spacing in T: kept
/// in FP32 the state ends one T step above 1 and y1 = 32 (1 + 2B); stored in T between two dispatches it stays 1.
#[uzu_test]
fn retention_witnesses() {
    fn retained<T: ArrayElement + Float + Debug + Default>(
        fixture: &KernelFixture,
        b: f64,
        [one, above_one, y_32, y_above]: [u16; 4],
    ) {
        let bits = |bits: u16| bytemuck::pod_read_unaligned::<T>(&bits.to_ne_bytes()).to_f64().unwrap();
        let arrays: [&[f64]; 7] = [&[1.0], &[-104.0], &[b], &[1.0], &[0.0], &[32.0], &[1.0]];
        let ([y_32, y_above], [one, above_one]) = ([y_32, y_above].map(bits), [one, above_one].map(bits));
        witness::<T>(fixture, [2, 1, 1, 1, 1], arrays, &[], [&[y_32, y_above], &[above_one]], "retained");
        witness::<T>(fixture, [2, 1, 1, 1, 1], arrays, &[1], [&[y_32, y_32], &[one]], "stored between dispatches");
    }
    let fixture = KernelFixture::new();
    retained::<f16>(&fixture, 2f64.powi(-11), [0x3c00, 0x3c01, 0x5000, 0x5001]);
    retained::<bf16>(&fixture, 2f64.powi(-8), [0x3f80, 0x3f81, 0x4200, 0x4201]);
    fixture.assert_clean();
}

/// Up to 64 witnesses per stage from the CPU's bounded 16-bit grids, x = C = 1, B = d = 0: dt_raw in ±[1/8, 8) with s 1
/// (dt) or s in {5/4, 3/2, 7/4, 3} (decay), z 32; s in [1, 4), z in {1/2, 1, 2, 3}, dt_raw -104 (gate). Each y set is
/// one same value on both sides, which the set staging `stage` as the contract does not excludes on both.
#[uzu_test]
fn staging_witnesses() {
    fn staged<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let v = |value: f64| T::from(value).unwrap();
        let grid = |low: f64, high: f64| {
            let values = (0..=u16::MAX).map(|bits| bytemuck::pod_read_unaligned::<T>(&bits.to_ne_bytes()));
            values.filter(move |value| (low..high).contains(&value.to_f64().unwrap().abs()))
        };
        let y = |shader: bool, stage: &str, [dt, s, z]: [T; 3]| {
            let f = |value: T| value.to_f64().unwrap();
            prefill::<T>(shader, stage, 0.0, &[([1.0, f(dt), f(z)], vec![(0.0, 1.0)])], vec![point(f(s))]).0[0]
        };
        for stage in ["dt", "decay", "gate"] {
            let candidates: Vec<[T; 3]> = match stage {
                "dt" => grid(0.125, 8.0).map(|dt| [dt, v(1.0), v(32.0)]).collect(),
                "decay" => {
                    grid(0.125, 8.0).flat_map(|dt| [1.25, 1.5, 1.75, 3.0].map(|s| [dt, v(s), v(32.0)])).collect()
                },
                _ => grid(1.0, 4.0)
                    .filter(|s| s.is_sign_positive())
                    .flat_map(|s| [0.5, 1.0, 2.0, 3.0].map(|z| [v(-104.0), s, v(z)]))
                    .collect(),
            };
            let witnesses = candidates
                .into_iter()
                .filter_map(|case| {
                    let expected = single(y(true, "", case))?;
                    let same = single(y(false, "", case)).map(f64::to_bits) == Some(expected.to_bits());
                    let staged = [true, false].map(|shader| y(shader, stage, case));
                    let excluded = staged.into_iter().all(|set| union(set, point(expected)) != set);
                    (same && excluded).then_some((case, expected))
                })
                .take(64)
                .collect::<Vec<_>>();
            let label = format!("{:?} {stage}: {} witnesses", T::data_type(), witnesses.len());
            eprintln!("{label}");
            assert!(!witnesses.is_empty(), "{label}");
            let column = |k: usize| witnesses.iter().map(|(case, _)| case[k]).collect::<Vec<_>>();
            let dims = [1, witnesses.len() as u32, 1, 1, 1];
            let strides = contiguous(dims);
            let inputs = inputs::<T>(dims, strides, |k, len| match k {
                1 => column(0),
                5 => column(2),
                6 => column(1),
                _ => vec![v([1.0, 0.0, 0.0, 1.0, 0.0][k]); len],
            });
            let outputs = run(fixture, dims, strides, &inputs, &[], &label);
            let expected = witnesses.iter().map(|&(_, y)| v(y)).collect::<Vec<_>>();
            for (side, [y, _]) in ["CPU", "Vulkan"].into_iter().zip(&outputs) {
                KernelFixture::assert_bits(&expected, y, &format!("{label}: {side} y"));
            }
        }
    }
    let fixture = KernelFixture::new();
    staged::<f16>(&fixture);
    staged::<bf16>(&fixture);
    fixture.assert_clean();
}

/// SSDUpdate's and the CPU's F32 order witnesses at Q 1, x = 1, dt_raw +inf (decay +0, so each state becomes B) and z
/// 2^-27 (gate 2^-28): d x after the dot (first: 2^-28), ascending (descending: (2^24 + 2) 2^-28), the +0 start (N 0
/// and a -0 term), unfused (an FMA: 2^-52).
#[uzu_test]
fn order_witnesses() {
    let fixture = KernelFixture::new();
    let p = |e: i32| 2f64.powi(e);
    let cases: [(&str, f64, &[f64], &[f64], &[f64], f64); 5] = [
        ("d x after the dot", -p(24), &[p(24), 1.0], &[1.0; 2], &[1.0; 2], 0.0),
        ("ascending dot", 0.0, &[p(24), 1.0, 1.0], &[1.0; 3], &[0.0; 3], p(-4)),
        ("+0 start without state", -0.0, &[], &[], &[], 0.0),
        ("+0 start before a -0 term", -0.0, &[-0.0], &[1.0], &[-1.0], 0.0),
        ("unfused", 0.0, &[-(1.0 + p(-11)), 1.0 + p(-12)], &[1.0, 1.0 + p(-12)], &[0.0; 2], 0.0),
    ];
    for (name, d, b, c, state, y) in cases {
        let arrays: [&[f64]; 7] = [&[1.0], &[f64::INFINITY], b, c, &[d], &[p(-27)], state];
        witness::<f32>(&fixture, [1, 1, 1, b.len() as u32, 1], arrays, &[], [&[y], b], name);
    }
    fixture.assert_clean();
}

/// The state update s decay + B x is two rounded products and a sum; fusing either product keeps its rounding error.
/// With each side's own decay at dt_raw -1 (its state from s = 1, x = 0), s = 1 + 2^-23 and B = -(s decay) cancel to +0
/// unless s decay is fused; at decay 1 (dt_raw -104), s = -(1 + 2^-11) and x = B = 1 + 2^-12 cancel to +0 unless B x
/// is fused. With C 1, d 0 and gate 32, y and the state are +0; each fused alternative is asserted nonzero.
#[uzu_test]
fn recurrence_fma_witnesses() {
    let fixture = KernelFixture::new();
    let p = |e: i32| 2f32.powi(e);
    let dims = [1, 1, 1, 1, 1];
    let unit = |values: [f32; 7], label: &str| {
        let inputs = inputs::<f32>(dims, contiguous(dims), |k, len| vec![values[k]; len]);
        run(&fixture, dims, contiguous(dims), &inputs, &[], label)
    };
    let decays = unit([0.0, -1.0, 0.0, 1.0, 0.0, 32.0, 1.0], "decay").map(|[_, state]| state[0]);
    let (s, b) = (1.0 + p(-23), 1.0 + p(-12));
    for (side, decay) in decays.into_iter().enumerate() {
        // (dt_raw, s, x, B, the next state with one product fused)
        let cases = [
            (-1.0, s, 1.0, -(s * decay), s.mul_add(decay, -(s * decay))),
            (-104.0, -(1.0 + p(-11)), b, b, b.mul_add(b, -(1.0 + p(-11)))),
        ];
        for (dt_raw, s, x, b, fused) in cases {
            let label =
                format!("{} dt_raw {dt_raw}: decay {decay:e}, fused next state {fused:e}", ["CPU", "Vulkan"][side]);
            assert!(fused != 0.0, "{label}: the fused alternative does not differ");
            let outputs = unit([x, dt_raw, b, 1.0, 0.0, 32.0, s], &label);
            let [y, state] = &outputs[side];
            KernelFixture::assert_bits(&[0.0], y, &format!("{label}: y"));
            KernelFixture::assert_bits(&[0.0], state, &format!("{label}: state"));
        }
    }
    fixture.assert_clean();
}

/// The CPU's class witnesses in every type. One infinity at a time, each its own head, with x = B = C = state = 1, decay
/// 1 (dt_raw -104), d 0 and gate 32 (state 2, y 64): x reaches the state and y is NaN (d x = 0 inf); B and the state
/// reach both; C, d and z only y, SiLU(-inf) NaN; dt_raw +inf decays to exactly 0 and -inf to 1; an infinite state times
/// that zero is NaN. Then a NaN reaches exactly its dependents in [Q 2, H 1, Dh 1, N 2].
fn class_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let base = [1.0, -104.0, 1.0, 1.0, 0.0, 32.0, 1.0];
    let cases: [(&[(usize, f64)], f64, f64); 15] = [
        (&[(0, inf)], nan, inf),
        (&[(0, -inf)], nan, -inf),
        (&[(2, inf)], inf, inf),
        (&[(2, -inf)], -inf, -inf),
        (&[(3, inf)], inf, 2.0),
        (&[(3, -inf)], -inf, 2.0),
        (&[(4, inf)], inf, 2.0),
        (&[(4, -inf)], -inf, 2.0),
        (&[(5, inf)], inf, 2.0),
        (&[(5, -inf)], nan, 2.0),
        (&[(6, inf)], inf, inf),
        (&[(6, -inf)], -inf, -inf),
        (&[(1, inf)], 32.0, 1.0),
        (&[(1, -inf)], 64.0, 2.0),
        (&[(1, inf), (6, inf)], nan, nan),
    ];
    let columns: [Vec<f64>; 7] = std::array::from_fn(|k| {
        cases
            .iter()
            .map(|(set, _, _)| set.iter().find(|&&(array, _)| array == k).map_or(base[k], |&(_, v)| v))
            .collect()
    });
    let [ys, states] = [1, 2].map(|k| cases.iter().map(|case| [case.1, case.2][k - 1]).collect::<Vec<_>>());
    witness::<T>(fixture, [1, 15, 1, 1, 1], columns.each_ref().map(Vec::as_slice), &[], [&ys, &states], "infinities");

    let dims = [2, 1, 1, 2, 1];
    // (array, element, y NaN, state NaN): z, C and d reach only y; x, dt_raw, B and the state the state and y onwards.
    let cases = [
        (5, 0, [true, false], [false, false]),
        (3, 0, [true, false], [false, false]),
        (4, 0, [true, true], [false, false]),
        (0, 0, [true, true], [true, true]),
        (1, 1, [false, true], [true, true]),
        (2, 3, [false, true], [false, true]),
        (6, 0, [true, true], [true, false]),
    ];
    for (array, element, y_nan, state_nan) in cases {
        let mut inputs = inputs::<T>(dims, contiguous(dims), |k, len| values(len, k, 0));
        inputs[array][element] = T::nan();
        let label = format!("{:?} NaN in array {array} element {element}", T::data_type());
        let outputs = run(fixture, dims, contiguous(dims), &inputs, &[], &label);
        for (side, [y, state]) in ["CPU", "Vulkan"].into_iter().zip(&outputs) {
            let nans = |values: &[T]| values.iter().map(|value| value.is_nan()).collect::<Vec<_>>();
            assert_eq!((nans(y), nans(state)), (y_nan.to_vec(), state_nan.to_vec()), "{label}: {side}");
        }
    }
}

/// The classes in every type; T narrowing only at the stores: an F16 state passing 65504 in FP32 (60000 + 60000 -
/// 60000) ends 60000, infinite when stored between two dispatches; an F32 subnormal state 2^-140 survives two tokens.
#[uzu_test]
fn class_and_narrowing_witnesses() {
    let fixture = KernelFixture::new();
    class_witnesses::<f32>(&fixture);
    class_witnesses::<f16>(&fixture);
    class_witnesses::<bf16>(&fixture);
    let (dims, inf) = ([2, 1, 1, 1, 1], f64::INFINITY);
    let arrays: [&[f64]; 7] = [&[1.0], &[-104.0], &[60000.0, -60000.0], &[2f64.powi(-10)], &[0.0], &[32.0], &[60000.0]];
    witness::<f16>(&fixture, dims, arrays, &[], [&[3750.0, 1875.0], &[60000.0]], "F16 state past 65504");
    witness::<f16>(&fixture, dims, arrays, &[1], [&[3750.0, inf], &[inf]], "F16 state stored between dispatches");
    let tiny = f64::from(f32::from_bits(1 << 9));
    let arrays: [&[f64]; 7] = [&[1.0], &[-104.0], &[0.0], &[2f64.powi(100)], &[0.0], &[32.0], &[tiny]];
    witness::<f32>(&fixture, dims, arrays, &[], [&[2f64.powi(-35); 2], &[tiny]], "F32 subnormal state");
    fixture.assert_clean();
}

/// Two dispatches (tokens 0..2, 2..5) in one command buffer continue one state, stored in T at the boundary: in the
/// oracle in every type, and with F32 one dispatch's bits on each side.
#[uzu_test]
fn chained_dispatches() {
    fn chained<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let (dims, label) = ([5, 3, 4, 8, 3], format!("{:?} chained", T::data_type()));
        let strides = contiguous(dims);
        let inputs = inputs::<T>(dims, strides, |k, len| values(len, k, 0));
        let splits: [&[u32]; 2] = [&[], &[2]];
        let [one, two] = splits.map(|splits| run(fixture, dims, strides, &inputs, splits, &label));
        if matches!(T::data_type(), DataType::F32) {
            for ((side, one), two) in ["CPU", "Vulkan"].into_iter().zip(&one).zip(&two) {
                assert_same_bits(&one[0], &two[0], &format!("{label}: {side} y"));
                assert_same_bits(&one[1], &two[1], &format!("{label}: {side} state"));
            }
        }
    }
    let fixture = KernelFixture::new();
    chained::<f32>(&fixture);
    chained::<f16>(&fixture);
    chained::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Run alone: `... ssd_prefill_test::throughput -- --ignored --nocapture`. Times illustrative shapes [Q, H, Dh, N,
/// group_size], not tied to a model configuration, 13 submissions each on the same full buffers: Vulkan GPU and wall
/// medians of the last 10 (`median_times`) and the CPU kernel's wall median of its last 10. Inputs replicate one compact
/// [Q, H, 1, N, group_size] pattern of finite eighths / 16 (dt_raw 0.5 + |·|) across Dh: on each side every y and
/// state element must equal its head's element 0 bit for bit and be finite, and those elements 0 lie in the oracle
/// after 13 submissions; at N 64 SSDPrefill64 gives SSDPrefill's bits. Two rounds in exactly opposite orders, both
/// reported; no clock or thermal state is assumed.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Debug + Default>(
        fixture: &KernelFixture,
        round: usize,
        variants: &[([u32; 5], bool)],
    ) {
        let mut previous: Option<([u32; 5], [[Vec<T>; 2]; 2])> = None;
        for &(dims, special64) in variants {
            let (compact, strides) = ([dims[0], dims[1], 1, dims[3], dims[4]], contiguous(dims));
            let [dh, n] = [dims[2], dims[3]].map(|extent| extent as usize);
            let pattern = inputs::<T>(compact, contiguous(compact), |k, len| {
                let scaled = values::<T>(len, k, 0).into_iter().map(|value| value.to_f64().unwrap() / 16.0);
                scaled
                    .map(|value| {
                        T::from(if k == 1 {
                            0.5 + value.abs()
                        } else {
                            value
                        })
                        .unwrap()
                    })
                    .collect()
            });
            // Full x and z [Q, H, Dh] and state [H, Dh, N] indices to the compact ones; dt_raw, B, C and d keep theirs.
            let full = inputs::<T>(dims, strides, |k, len| {
                let compact_index = |i: usize| match k {
                    0 | 5 => i / dh,
                    6 => i / (dh * n) * n + i % n,
                    _ => i,
                };
                (0..len).map(|i| pattern[k][compact_index(i)]).collect()
            });
            let (cpu, cpu_times) = cpu_prefill(dims, strides, &full, special64, &[], 13);
            let (gpu, times) = gpu_prefill(fixture, dims, strides, &full, special64, &[], true);
            let (variant, outputs) = (["generic", "literal64"][usize::from(special64)], [cpu, gpu]);
            let label = format!("SSDPrefill throughput round {round} {:?} {variant} {dims:?}", T::data_type());
            // Each y [Q, H, Dh] and state [H, Dh, N] element's channel 0 of its head.
            let first = |k: usize, i: usize| {
                if k == 0 {
                    i - i % dh
                } else {
                    i / (dh * n) * dh * n + i % n
                }
            };
            for (side, [y, state]) in ["CPU", "Vulkan"].into_iter().zip(&outputs) {
                for (k, values) in [y, state].into_iter().enumerate() {
                    let name = format!("{label}: {side} {}", ["y", "state"][k]);
                    assert_same_bits(
                        &(0..values.len()).map(|i| values[first(k, i)]).collect::<Vec<_>>(),
                        values,
                        &name,
                    );
                    assert!(values.iter().all(|value| value.is_finite()), "{name}: not finite");
                }
            }
            let compacted = outputs.each_ref().map(|[y, state]| {
                [
                    (0..y.len() / dh).map(|c| y[c * dh]).collect(),
                    (0..state.len() / dh).map(|c| state[c / n * dh * n + c % n]).collect(),
                ]
            });
            let width = check(compact, contiguous(compact), &pattern, &[], 13, &compacted, &label);
            if let Some((previous, other)) = &previous
                && *previous == dims
            {
                for (side, (other, outputs)) in ["CPU", "Vulkan"].into_iter().zip(other.iter().zip(&outputs)) {
                    assert_same_bits(
                        &other.concat(),
                        &outputs.concat(),
                        &format!("{label}: {side} generic and literal64"),
                    );
                }
            }
            let ((gpu, wall), mut cpu) = (times.expect("timed"), cpu_times[3..].to_vec());
            cpu.sort();
            let [y, state] = [outputs[0][0].len(), outputs[0][1].len()];
            eprintln!(
                "{label}: GPU {gpu:?}, wall {wall:?}; CPU wall {:?}; each side: {y} y and {state} state elements equal \
                 across {dh} channels per head, {} y and {} state channel-0 elements in the oracle after 13 submissions \
                 (largest relative set width {width:e})",
                cpu[cpu.len() / 2],
                compacted[0][0].len(),
                compacted[0][1].len()
            );
            previous = Some((dims, outputs));
        }
    }
    let variants = [
        ([1, 128, 64, 128, 16], false),
        ([32, 64, 64, 128, 8], false),
        ([128, 16, 64, 64, 4], false),
        ([128, 16, 64, 64, 4], true),
        ([128, 8, 64, 256, 4], false),
        ([128, 2, 64, 1024, 2], false),
        ([32, 1, 64, 4096, 1], false),
    ];
    let reversed = variants.iter().rev().copied().collect::<Vec<_>>();
    let fixture = KernelFixture::new();
    eprintln!("SSDPrefill throughput round 0");
    measure::<f32>(&fixture, 0, &variants);
    measure::<f16>(&fixture, 0, &variants);
    measure::<bf16>(&fixture, 0, &variants);
    eprintln!("SSDPrefill throughput round 1");
    measure::<bf16>(&fixture, 1, &reversed);
    measure::<f16>(&fixture, 1, &reversed);
    measure::<f32>(&fixture, 1, &reversed);
    fixture.assert_clean();
}
