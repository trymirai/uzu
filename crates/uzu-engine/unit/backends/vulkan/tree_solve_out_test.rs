use std::{fmt::Debug, ops::Range, slice, sync::Arc, time::Duration};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    CPU_FAILURE, NAN, NEG_ZERO, POS_ZERO, add, arg, assert_inputs, assert_same_bits, bounds, conv1d_values, cpu_buffer,
    cpu_submissions, cpu_tree_gram, cpu_tree_prefix, decay_set, delta_net_check, kernel_fixture::KernelFixture, mul,
    negate, panics, point, sentinel, single, tree_gram_inputs,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            kernel::{BuildTreeOutKernel, TreeUpdateSolveKernel},
        },
        cpu::Cpu,
        vulkan::{BuildTreeOutVulkanKernel, Error, TreeUpdateSolveVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// [B, N, H, Dv] of TreeUpdateSolve: one block; a one-token second block over two heads; three blocks at Dv 128; two
/// batches of four blocks with a dv tail past 64; one element; 65600 heads past 65535 groups.
const SOLVE_SHAPES: [[u32; 4]; 6] =
    [[1, 16, 1, 16], [1, 17, 2, 20], [1, 33, 4, 128], [2, 49, 3, 65], [1, 1, 1, 1], [1, 1, 65600, 1]];

/// [B, N, Hqk, Hv, Dk, Dv] of BuildTreeOut: one block; a dv tail past 64; two batches at the ratio 2; the 3 over 2
/// ratio, whose v head 2 reads q head 2; 65600 v heads past 65535 groups.
const OUT_SHAPES: [[u32; 6]; 5] =
    [[1, 16, 1, 1, 4, 3], [1, 17, 1, 2, 5, 70], [2, 33, 2, 4, 8, 6], [1, 40, 2, 3, 3, 2], [1, 1, 1, 65600, 1, 1]];

/// Every [use_mxu, transposed_h0], none of which changes BuildTreeOut.
const FLAGS: [[bool; 2]; 4] = [[false, false], [false, true], [true, false], [true, true]];

/// Poisons to NaN every A and a_inv entry TreeUpdateSolve never reads: A tile rows past the block's tokens, columns
/// from the block's first token on, later pairs, and a_inv rows or columns past the block's tokens.
fn unread(
    n: usize,
    a: &mut [f32],
    inverse: &mut [f32],
) {
    let (blocks, pairs) = (n.div_ceil(16), n.div_ceil(16).div_ceil(2));
    let (rows, block) = (|block: usize| 16.min(n - 16 * block), |i: usize| i / 512 / pairs % blocks);
    let tile = |i: usize| i % 512 / 32 >= rows(block(i)) || i / 512 % pairs * 32 + i % 32 >= 16 * block(i);
    a.iter_mut().enumerate().filter(|&(i, _)| tile(i)).for_each(|(_, x)| *x = f32::NAN);
    let past = |i: usize| (i / 16 % 16).max(i % 16) >= rows(i / 256 % blocks);
    inverse.iter_mut().enumerate().filter(|&(i, _)| past(i)).for_each(|(_, x)| *x = f32::NAN);
}

/// Input data, never an oracle: tree_gram_inputs' q, k / 16 and h0 or, `positive`, shifted into (0, 8) as q / 64,
/// k / 64 and h0 / 16, so every dot is positive; beta in (0, 1/2); the CPU Prefix of log_decay = -beta on the trie;
/// the CPU Gram's A, qkd, a_inv and kh0 from them, with the A and a_inv entries Solve never reads NaN.
fn tree_inputs<T: ArrayElement + Float + Default>(
    shape: [u32; 6],
    kind: &str,
    slots: &[i32],
    positive: bool,
) -> (Vec<T>, [Vec<f32>; 7]) {
    let ([b, n, _, hv, ..], shift) = (shape, [0.0, 4.0][usize::from(positive)]);
    let ([q, k], nodes, [_, beta, h0]) = tree_gram_inputs::<T>(shape, kind, slots);
    let by = [[1.0, 0.0625, 1.0], [0.015625, 0.015625, 0.0625]][usize::from(positive)];
    let map =
        |x: &[T], by: f32| x.iter().map(|x| T::from((x.to_f32().unwrap() + shift) * by).unwrap()).collect::<Vec<T>>();
    let (q, k, h0) = (map(&q, by[0]), map(&k, by[1]), h0.iter().map(|x| (x + shift) * by[2]).collect());
    let beta = beta.iter().map(|x| (x + 4.0) / 16.0).collect::<Vec<f32>>();
    let log_decay = beta.iter().map(|x| -x).collect::<Vec<_>>();
    let f32s = [cpu_tree_prefix([b, n, hv], bytemuck::cast_slice(&nodes[..]), &log_decay, 1).0, beta, h0];
    let gram = cpu_tree_gram(shape, [false, !slots.is_empty()], 0.375, (&[q.clone(), k], &nodes, &f32s, slots), 1);
    let ([mut a, qkd, mut inverse, kh0], [prefix, beta, h0]) = (gram.0, f32s);
    unread(n as usize, &mut a, &mut inverse);
    (q, [prefix, beta, h0, a, qkd, inverse, kh0])
}

/// The CPU Prefix's prefix (variant 0), the same negated at odd lanes so e sees prefixes of both signs (1), or the
/// exact-class prefix (2): -inf, so e = +0, at every token but each column's last, which alternates +inf and NaN.
fn prefix_variant(
    prefix: Vec<f32>,
    [n, heads]: [u32; 2],
    variant: usize,
) -> Vec<f32> {
    let last = |i: usize| i / heads as usize % n as usize + 1 == n as usize;
    let special = |i: usize| [f32::NEG_INFINITY, [f32::INFINITY, f32::NAN][i % 2]][usize::from(last(i))];
    prefix.iter().enumerate().map(|(i, &p)| [p, [p, -p][i % 2], special(i)][variant]).collect()
}

/// Both ends of an exp set, its finite range or its single class: a set mixing the two has no endpoint replay.
fn ends(set: ((f64, f64), u8)) -> [f32; 2] {
    match set {
        ((lo, hi), 0) => [lo as f32, hi as f32],
        (_, NAN) => [f32::NAN; 2],
        _ => [single(set).expect("an exp set with both finite members and classes") as f32; 2],
    }
}

/// Each u element's set, in the CPU's order on this side's own returned earlier u: per token rhs = beta (v - e kh),
/// minus a u over the earlier tokens ascending, then the a_inv sum from +0 over the block. `allowance` carries e's set
/// through the set algebra; otherwise the monotone endpoint replay carries both ends of e and sums at the corners each
/// a_inv sign selects: a point where both ends agree for all the block's tokens, else bounds. In a block where a
/// finite e set reaches some rhs through a nonzero kh, every replayed operation must be finite; elsewhere e kh is
/// exact, so both ends agree, IEEE classes included.
fn solve_sets(
    [b, n, h, dv]: [usize; 4],
    v: &[f32],
    [prefix, beta, a, inverse, kh0]: &[Vec<f32>; 5],
    slots: &[i32],
    u: &[f32],
    [shader, allowance]: [bool; 2],
    strict: bool,
) -> Vec<((f64, f64), u8)> {
    let (blocks, pairs) = (n.div_ceil(16), n.div_ceil(16).div_ceil(2));
    let mut sets = vec![point(0.0); u.len()];
    for (batch, head, d, block) in itertools::iproduct!(0..b, 0..h, 0..dv, 0..blocks) {
        let (group, rows, slot) = (batch * h + head, 16.min(n - 16 * block), slots.get(batch).copied().unwrap_or(-1));
        let tile = (group * blocks + block) * pairs * 512;
        let product =
            |lt: usize, prev: usize| a[tile + prev / 32 * 512 + lt * 32 + prev % 32] * u[(group * n + prev) * dv + d];
        let w = |lt: usize, lp: usize| inverse[((group * blocks + block) * 16 + lt) * 16 + lp];
        let at = |lt: usize| (group * n + 16 * block + lt) * dv + d;
        let tokens = (0..rows).map(|lt| {
            let lane = (batch * n + 16 * block + lt) * h + head;
            let kh = match slot >= 0 {
                true => kh0[lane * dv + d],
                false => 0.0,
            };
            (decay_set(prefix[lane], shader), beta[lane], v[lane * dv + d], kh)
        });
        let tokens = tokens.collect::<Vec<_>>();
        if allowance {
            let allowed = |lt: usize| {
                let (e, beta, v, kh) = tokens[lt];
                let term = negate(mul::<f32>(e, point(kh.into())));
                let rhs = mul::<f32>(point(beta.into()), add::<f32>(point(v.into()), term));
                (0..16 * block).fold(rhs, |x, prev| add::<f32>(x, negate(point(product(lt, prev).into()))))
            };
            let acc = (0..rows).map(allowed).collect::<Vec<_>>();
            for lt in 0..rows {
                let terms = (0..rows).map(|lp| mul::<f32>(point(w(lt, lp).into()), acc[lp]));
                sets[at(lt)] = terms.fold(point(0.0), add::<f32>);
            }
            continue;
        }
        let finite = strict || tokens.iter().any(|&((_, mask), _, _, kh)| mask == 0 && kh != 0.0);
        let step = |x: f32| {
            assert!(!finite || x.is_finite(), "TreeUpdateSolve finite endpoint replay step {x}");
            x
        };
        let replay = |lt: usize| {
            let (e, beta, v, kh) = tokens[lt];
            let rhs = ends(e).map(|e| step(beta * step(v - step(step(e) * kh))));
            (0..16 * block).fold(rhs, |x, prev| x.map(|x| step(x - step(product(lt, prev)))))
        };
        let acc = (0..rows).map(replay).collect::<Vec<_>>();
        let exact = acc.iter().all(|[x, y]| x.to_bits() == y.to_bits());
        for lt in 0..rows {
            let corner = |high: bool| {
                let pick = |[x, y]: [f32; 2], w: f32| [x.min(y), x.max(y)][usize::from((w >= 0.0) == high)];
                (0..rows).fold(0.0f32, |s, lp| step(s + step(w(lt, lp) * pick(acc[lp], w(lt, lp)))))
            };
            let (lo, hi) = (corner(false), corner(true));
            sets[at(lt)] = [bounds::<f32>((lo.into(), hi.into())), point(lo.into())][usize::from(exact)];
        }
    }
    sets
}

/// `count` CPU submissions, one at a time, each into a fresh u of sentinels read once it completes.
fn cpu_solve<T: ArrayElement + Float + Default>(
    [b, n, h, dv]: [u32; 4],
    bv: u32,
    (v, f32s, slots): (&[T], &[Vec<f32>; 5], &[i32]),
    count: usize,
) -> Vec<(Vec<f32>, Duration)> {
    let (context, use_h0) = (create_context::<Cpu>(), !slots.is_empty());
    let new = <<Cpu as Backend>::Kernels as Kernels>::TreeUpdateSolveKernel::new;
    let kernel = new(&context, T::data_type(), bv, use_h0).expect("CPU TreeUpdateSolve");
    let (values, index) = (cpu_buffer(&context, v), cpu_buffer(&context, bytemuck::cast_slice::<i32, u32>(slots)));
    let [prefix, beta, a, inverse, kh0] = f32s.each_ref().map(|values| cpu_buffer(&context, values));
    let sample = |_| {
        let mut u = cpu_buffer(&context, &vec![sentinel::<f32>(); v.len()]);
        let time = cpu_submissions(&context, 1, |command_buffer| {
            let (kh0, index) = (use_h0.then_some(&kh0), use_h0.then_some(&index));
            kernel.encode(kh0, &values, &prefix, &beta, &a, &inverse, index, &mut u, b, n, h, dv, command_buffer);
        });
        (buffer_prefix_to_vec::<Cpu, f32>(&u, v.len()), time[0])
    };
    (0..count).map(sample).collect()
}

/// `count` Vulkan submissions, each of what `record` records into a fresh guarded output of `len` sentinels: once
/// each completes and its GPU and wall times are taken, `unchanged` asserts the read-only inputs and the output is read
/// with its guards asserted.
fn gpu_samples<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    (len, count): (usize, usize),
    record: impl Fn(&mut VkCommandBufferEncoding, (&Arc<VkBuffer>, Range<u64>)),
    unchanged: impl Fn(),
) -> Vec<(Vec<T>, (Duration, Duration))> {
    let sample = |_| {
        let output = fixture.guarded(&vec![sentinel::<T>(); len], sentinel::<T>());
        let (start, mut encoding) = (std::time::Instant::now(), fixture.encoding());
        record(&mut encoding, arg(&output));
        let times = (KernelFixture::complete(encoding).gpu_execution_time(), start.elapsed());
        unchanged();
        // SAFETY: the only command buffer using the output has completed.
        (unsafe { KernelFixture::read_guarded(&output, sentinel::<T>()) }, times)
    };
    (0..count).map(sample).collect()
}

/// A variant's first CPU and Vulkan outputs, then `timed` more of each, from runs of `cpu` and `gpu` with that many
/// submissions. While nothing is `checked`, `validate` checks the first outputs before any timed run, and they become
/// the checked ones. Every output of a side must equal its checked one bit for bit, so it lies in the same oracle sets
/// (for Solve its own earlier u, the oracle's points, are the same). Returns the timed runs' times.
fn verify<U: ArrayElement + Debug>(
    checked: &mut Option<[Vec<U>; 2]>,
    timed: usize,
    cpu: impl Fn(usize) -> Vec<(Vec<U>, Duration)>,
    gpu: impl Fn(usize) -> Vec<(Vec<U>, (Duration, Duration))>,
    validate: impl Fn(&[Vec<U>; 2]),
    label: &str,
) -> (Vec<Duration>, Vec<(Duration, Duration)>) {
    let first = [cpu(1).remove(0).0, gpu(1).remove(0).0];
    let checked = checked.get_or_insert_with(|| {
        validate(&first);
        first.clone()
    });
    let (cpu, gpu) = (cpu(timed), gpu(timed));
    let later = cpu.iter().map(|(output, _)| (0, output)).chain(gpu.iter().map(|(output, _)| (1, output)));
    for (side, output) in first.iter().enumerate().chain(later) {
        assert_same_bits(&checked[side], output, &format!("{label}: {} output", ["CPU", "Vulkan"][side]));
    }
    (cpu.into_iter().map(|(_, time)| time).collect(), gpu.into_iter().map(|(_, time)| time).collect())
}

/// TreeUpdateSolve at each BV through `verify`, on the CPU and on guarded Vulkan buffers whose read-only inputs and
/// guards are asserted after every completion: the first BV's first u of each side checked against its monotone
/// endpoint replay sets and, at N <= 33 and Dv <= 20, its allowance sets, both conditional on its own earlier u.
fn solve_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape @ [b, n, h, dv]: [u32; 4],
    inputs @ (v, f32s, slots): (&[T], &[Vec<f32>; 5], &[i32]),
    bvs: &[u32],
    timed: usize,
    label: &str,
) -> ([Vec<f32>; 2], (Vec<Duration>, Vec<(Duration, Duration)>)) {
    let (use_h0, word, small) = (!slots.is_empty(), sentinel::<f32>().to_bits(), n <= 33 && dv <= 20);
    let (stored, index) = (fixture.guarded(v, sentinel::<T>()), fixture.guarded(bytemuck::cast_slice(slots), word));
    let guarded = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    // SAFETY: `gpu_samples` calls it only once the command buffer using these buffers has completed.
    let unchanged = || unsafe {
        assert_inputs(slice::from_ref(&stored), &[v], &format!("{label} v"));
        assert_inputs(&guarded, &f32s.each_ref().map(Vec::as_slice), &format!("{label} f32 inputs"));
        KernelFixture::assert_unchanged(&index, word, bytemuck::cast_slice(slots), &format!("{label} h0_idx"));
    };
    let (dims, values) = (shape.map(|extent| extent as usize), v.iter().map(|x| x.to_f32().unwrap()));
    let values = values.collect::<Vec<_>>();
    let validate = |outputs: &[Vec<f32>; 2]| {
        for (side, (name, u)) in ["CPU", "Vulkan"].into_iter().zip(outputs).enumerate() {
            let owner = (0..u.len()).map(Some).collect::<Vec<_>>();
            let modes = [(false, "monotone endpoint replay"), (true, "allowance sets")];
            for (allowance, oracle) in modes.into_iter().filter(|&(allowance, _)| small || !allowance) {
                let sets = solve_sets(dims, &values, f32s, slots, u, [side == 1, allowance], timed != 0);
                let case = format!("{label} {name} u, conditional on its own earlier u: {oracle}");
                delta_net_check(&sets, &owner, u, u, &case);
            }
        }
    };
    let (mut checked, mut times) = (None, Default::default());
    for &bv in bvs {
        let kernel =
            TreeUpdateSolveVulkanKernel::new(&fixture.context, T::data_type(), bv, use_h0).expect("TreeUpdateSolve");
        // SAFETY: each range holds every element the shape addresses, aligned; u aliases nothing.
        let record = |encoding: &mut VkCommandBufferEncoding, u: (&Arc<VkBuffer>, Range<u64>)| unsafe {
            let [prefix, beta, a, inverse, kh0] = guarded.each_ref().map(arg);
            let (kh0, slot) = (use_h0.then_some(kh0), use_h0.then(|| arg(&index)));
            kernel.encode(kh0, arg(&stored), prefix, beta, a, inverse, slot, u, b, n, h, dv, encoding)
        };
        let cpu = |count| cpu_solve(shape, bv, inputs, count);
        let gpu = |count| gpu_samples::<f32>(fixture, (v.len(), count), &record, &unchanged);
        times = verify(&mut checked, timed, cpu, gpu, validate, &format!("{label} BV {bv} u"));
    }
    (checked.expect("at least one BV"), times)
}

/// Each o element's set: from +0, for a nonnegative h0 slot (e scale) times the ascending q h0 dot, then every qkd u
/// product over col ascending, all in FP32, and finally stored as O. `allowance` carries e's set through the set
/// algebra; otherwise the monotone endpoint replay of the h0 term's two ends gives a point where they agree, else
/// bounds. Where a finite e set reaches the output through a nonzero dot, every replayed operation must be finite, the
/// dot's own included, as a nonfinite product or partial sum leaves the ascending dot nonfinite; elsewhere the term is
/// exact.
fn out_sets<O: Float>(
    [b, n, hqk, hv, dk, dv]: [usize; 6],
    q: &[f32],
    [prefix, qkd, u, h0]: &[Vec<f32>; 4],
    slots: &[i32],
    scale: f32,
    [shader, allowance]: [bool; 2],
    strict: bool,
) -> Vec<((f64, f64), u8)> {
    let set = |(batch, row, head, d): (usize, usize, usize, usize)| {
        let (lane, group) = ((batch * n + row) * hv + head, batch * hv + head);
        let product = |col: usize| qkd[(group * n + row) * n + col] * u[(group * n + col) * dv + d];
        let term = slots.get(batch).copied().filter(|&slot| slot >= 0).map(|slot| {
            let q = &q[(lane / hv * hqk + head / (hv / hqk)) * dk..][..dk];
            let state = &h0[((slot as usize * hv + head) * dv + d) * dk..][..dk];
            (decay_set(prefix[lane], shader), q.iter().zip(state).fold(0.0f32, |s, (x, y)| s + x * y))
        });
        if allowance {
            let start = term.map_or(point(0.0), |(e, dot)| {
                add::<f32>(point(0.0), mul::<f32>(mul::<f32>(e, point(scale.into())), point(dot.into())))
            });
            // O only after the whole FP32 chain: the set product with 1, exact in FP32, rounds every member to O and
            // keeps every class and zero sign.
            return mul::<O>((0..n).fold(start, |s, col| add::<f32>(s, point(product(col).into()))), point(1.0));
        }
        let finite = strict || term.is_some_and(|((_, mask), dot)| mask == 0 && dot != 0.0);
        let step = |x: f32| {
            assert!(!finite || x.is_finite(), "BuildTreeOut finite endpoint replay step {x}");
            x
        };
        let starts =
            term.map_or([0.0; 2], |(e, dot)| ends(e).map(|e| step(0.0 + step(step(step(e) * scale) * step(dot)))));
        let [x, y] = starts.map(|x| (0..n).fold(x, |s, col| step(s + step(product(col)))));
        if x.to_bits() == y.to_bits() {
            return point(O::from(x).unwrap().to_f64().unwrap());
        }
        assert!(finite, "BuildTreeOut endpoint replay [{x}, {y}] without a finite e");
        bounds::<O>((x.min(y).into(), x.max(y).into()))
    };
    itertools::iproduct!(0..b, 0..n, 0..hv, 0..dv).map(set).collect()
}

/// `count` CPU submissions, one at a time, each into a fresh o of sentinels read once it completes.
fn cpu_out<Q: ArrayElement + Float + Default, O: ArrayElement + Float + Default>(
    [b, n, hqk, hv, dk, dv]: [u32; 6],
    [mxu, transposed]: [bool; 2],
    (q, f32s, slots): (&[Q], &[Vec<f32>; 4], &[i32]),
    scale: f32,
    count: usize,
) -> Vec<(Vec<O>, Duration)> {
    let (context, use_h0, len) = (create_context::<Cpu>(), !slots.is_empty(), f32s[0].len() * dv as usize);
    let new = <<Cpu as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new;
    let kernel = new(&context, Q::data_type(), O::data_type(), mxu, transposed, use_h0).expect("CPU BuildTreeOut");
    let (stored, index) = (cpu_buffer(&context, q), cpu_buffer(&context, bytemuck::cast_slice::<i32, u32>(slots)));
    let [prefix, qkd, u, h0] = f32s.each_ref().map(|values| cpu_buffer(&context, values));
    let sample = |_| {
        let mut o = cpu_buffer(&context, &vec![sentinel::<O>(); len]);
        let time = cpu_submissions(&context, 1, |command_buffer| {
            let (h0, index) = (use_h0.then_some(&h0), use_h0.then_some(&index));
            kernel.encode(&stored, &prefix, &qkd, &u, h0, index, &mut o, scale, b, n, hqk, hv, dk, dv, command_buffer);
        });
        (buffer_prefix_to_vec::<Cpu, O>(&o, len), time[0])
    };
    (0..count).map(sample).collect()
}

/// BuildTreeOut under each [use_mxu, transposed_h0] through `verify`, on the CPU and on guarded Vulkan buffers whose
/// read-only inputs and guards are asserted after every completion: the first flags' first o of each side checked
/// against its monotone endpoint replay set and, at N <= 33 and Dv <= 20, its allowance set, both output-independent.
fn out_check<Q: ArrayElement + Float + Debug + Default, O: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape @ [b, n, hqk, hv, dk, dv]: [u32; 6],
    inputs @ (q, f32s, slots): (&[Q], &[Vec<f32>; 4], &[i32]),
    scale: f32,
    flags: &[[bool; 2]],
    timed: usize,
    label: &str,
) -> ([Vec<O>; 2], (Vec<Duration>, Vec<(Duration, Duration)>)) {
    let (use_h0, word, small) = (!slots.is_empty(), sentinel::<f32>().to_bits(), n <= 33 && dv <= 20);
    let (stored, index) = (fixture.guarded(q, sentinel::<Q>()), fixture.guarded(bytemuck::cast_slice(slots), word));
    let guarded = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    // SAFETY: `gpu_samples` calls it only once the command buffer using these buffers has completed.
    let unchanged = || unsafe {
        assert_inputs(slice::from_ref(&stored), &[q], &format!("{label} q"));
        assert_inputs(&guarded, &f32s.each_ref().map(Vec::as_slice), &format!("{label} f32 inputs"));
        KernelFixture::assert_unchanged(&index, word, bytemuck::cast_slice(slots), &format!("{label} h0_indices"));
    };
    let (dims, values) = (shape.map(|extent| extent as usize), q.iter().map(|x| x.to_f32().unwrap()));
    let values = values.collect::<Vec<_>>();
    let sets = |shader, allowance| out_sets::<O>(dims, &values, f32s, slots, scale, [shader, allowance], timed != 0);
    let oracles = [false, true].map(|shader| [Some(sets(shader, false)), small.then(|| sets(shader, true))]);
    let validate = |outputs: &[Vec<O>; 2]| {
        for (side, (name, o)) in ["CPU", "Vulkan"].into_iter().zip(outputs).enumerate() {
            let owner = (0..o.len()).map(Some).collect::<Vec<_>>();
            for (sets, oracle) in oracles[side].iter().flatten().zip(["monotone endpoint replay", "allowance sets"]) {
                delta_net_check(sets, &owner, o, o, &format!("{label} {name} o: {oracle}"));
            }
        }
    };
    let (len, mut checked, mut times) = (f32s[0].len() * dv as usize, None, Default::default());
    for &flag in flags {
        let [mxu, transposed] = flag;
        let kernel =
            BuildTreeOutVulkanKernel::new(&fixture.context, Q::data_type(), O::data_type(), mxu, transposed, use_h0)
                .expect("BuildTreeOut");
        // SAFETY: each range holds every element the shape addresses, aligned; o aliases nothing.
        let record = |encoding: &mut VkCommandBufferEncoding, o: (&Arc<VkBuffer>, Range<u64>)| unsafe {
            let [prefix, qkd, u, h0] = guarded.each_ref().map(arg);
            let (h0, slot) = (use_h0.then_some(h0), use_h0.then(|| arg(&index)));
            kernel.encode(arg(&stored), prefix, qkd, u, h0, slot, o, scale, b, n, hqk, hv, dk, dv, encoding)
        };
        let cpu = |count| cpu_out::<Q, O>(shape, flag, inputs, scale, count);
        let gpu = |count| gpu_samples::<O>(fixture, (len, count), &record, &unchanged);
        times = verify(&mut checked, timed, cpu, gpu, validate, &format!("{label} {flag:?} o"));
    }
    (checked.expect("at least one flag pair"), times)
}

fn solve_matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let ty = T::data_type();
    for (index, shape @ [b, n, h, dv]) in SOLVE_SHAPES.into_iter().enumerate() {
        let kind = ["chain", "star", "binary", "random"][index % 4];
        for (slots, variant) in itertools::iproduct!([&[0, -1][..b as usize], &[]], 0..3) {
            let (_, [prefix, beta, _, mut a, _, mut inverse, kh0]) =
                tree_inputs::<T>([b, n, 1, h, 4, dv], kind, slots, false);
            for (values, seed) in [(&mut a, 7), (&mut inverse, 8)].into_iter().filter(|_| index == 3) {
                let synthetic = conv1d_values::<f32>(values.len(), seed, 0);
                values.iter_mut().zip(synthetic).filter(|(x, _)| !x.is_nan()).for_each(|(x, y)| *x = y / 64.0);
            }
            let prefix = prefix_variant(prefix, [n, h], variant);
            let v = conv1d_values::<T>(prefix.len() * dv as usize, 6, 0);
            let label = format!("TreeUpdateSolve {ty:?} {shape:?} {kind} slots {slots:?} prefix variant {variant}");
            solve_check(fixture, shape, (&v, &[prefix, beta, a, inverse, kh0], slots), &[16, 32], 0, &label);
        }
    }
}

/// Every shape and tree with slots [0, -1] (one per batch) and none, through `solve_check` at BV 16 and 32 for each
/// `prefix_variant`: finite e sets of both prefix signs (exact where no slot reads kh0) and exact classes. Shape 3's A
/// and a_inv entries Solve reads are synthetic; in every case the entries it never reads are NaN.
#[uzu_test]
fn tree_update_solve_matches_oracles() {
    let fixture = KernelFixture::new();
    solve_matches::<f32>(&fixture);
    solve_matches::<bf16>(&fixture);
    fixture.assert_clean();
}

fn out_matches<Q: ArrayElement + Float + Debug + Default, O: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture
) {
    let types = [Q::data_type(), O::data_type()];
    for (index, shape @ [b, n, _, hv, _, dv]) in OUT_SHAPES.into_iter().enumerate() {
        let kind = ["chain", "star", "binary", "random"][index % 4];
        for (slots, variant) in itertools::iproduct!([&[0, -1][..b as usize], &[]], 0..3) {
            let (q, [prefix, _, h0, _, qkd, ..]) = tree_inputs::<Q>(shape, kind, slots, false);
            let prefix = prefix_variant(prefix, [n, hv], variant);
            let u = conv1d_values::<f32>(prefix.len() * dv as usize, 6, 0);
            let label = format!("BuildTreeOut {types:?} {shape:?} {kind} slots {slots:?} prefix variant {variant}");
            out_check::<Q, O>(fixture, shape, (&q, &[prefix, qkd, u, h0], slots), -0.625, &FLAGS, 0, &label);
        }
    }
}

/// Every shape and tree with slots [0, -1] and none, each `prefix_variant`, the CPU Gram's qkd, for all four QKT and
/// OutputT pairs, through `out_check` under every use_mxu and transposed_h0.
#[uzu_test]
fn build_tree_out_matches_oracles() {
    let fixture = KernelFixture::new();
    out_matches::<f32, f32>(&fixture);
    out_matches::<f32, bf16>(&fixture);
    out_matches::<bf16, f32>(&fixture);
    out_matches::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}

fn solve_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let ty = format!("{:?}", T::data_type());
    let run = |name: &str, [n, h]: [usize; 2], [v, prefix, beta]: [Vec<f32>; 3], entries: &[(usize, usize, f32)]| {
        let (blocks, pairs) = (n.div_ceil(16), n.div_ceil(16).div_ceil(2));
        let (mut a, mut inverse) = (vec![0.0; h * blocks * pairs * 512], vec![0.0; h * blocks * 256]);
        for &(token, prev, value) in entries {
            a[(token / 16 * pairs + prev / 32) * 512 + token % 16 * 32 + prev % 32] = value;
        }
        inverse.chunks_mut(256).for_each(|block| (0..16).for_each(|i| block[i * 17] = 1.0));
        unread(n, &mut a, &mut inverse);
        let v = v.iter().map(|&x| T::from(x).unwrap()).collect::<Vec<T>>();
        let (f32s, label) = ([prefix, beta, a, inverse, vec![2.0; n * h]], format!("{ty} TreeUpdateSolve {name}"));
        solve_check(fixture, [1, n as u32, h as u32, 1], (&v, &f32s, &[-1]), &[16, 32], 0, &label).0
    };
    // One column under a negative slot: tokens 3, 4 and 5 hold u = 2^24, 1, -2^24, which token 16 subtracts at a = 1,
    // +0 ascending and -1 descending; token 17 subtracts the one product 1, -1 (adding it gives +1); token 0 holds
    // u0 = 1 + 2^-17 and token 18 has rhs = round((1 + 2^-7)(1 + 2^-17)), minus a u0 at a = 1 + 2^-7: +0 unfused, the
    // exact residual -2^-24 fused.
    let (big, step, fine) = (2f32.powi(24), 1.0 + 2f32.powi(-7), 1.0 + 2f32.powi(-17));
    let (mut v, mut beta) = (vec![0.0; 19], vec![1.0; 19]);
    (v[0], v[3], v[4], v[5], v[18], beta[0], beta[18]) = (1.0, big, 1.0, -big, step, fine, fine);
    let entries = [(16, 3, 1.0), (16, 4, 1.0), (16, 5, 1.0), (17, 4, 1.0), (18, 0, step)];
    for u in run("order, sign and fusion", [19, 1], [v, vec![0.0; 19], beta], &entries) {
        assert_eq!([u[16], u[17], u[18]].map(f32::to_bits), [0, (-1.0f32).to_bits(), 0], "{ty} order, sign and fusion");
    }
    // Four heads of two tokens under a negative slot: a NaN prefix still gives NaN, e kh being formed; v = -0 gives
    // beta (-0), which the a_inv sum from +0 makes +0; v = inf meets the upper a_inv +0: NaN; 2^-65 2^-65 = 2^-130.
    let (s, inf) = (2f32.powi(-65), f32::INFINITY);
    let (v, beta) = (vec![1.0, -0.0, 1.0, s, 1.0, -0.0, inf, 1.0], vec![1.0, 1.0, 1.0, s, 1.0, 1.0, 1.0, 1.0]);
    for u in run("classes", [2, 4], [v, [vec![f32::NAN], vec![0.0; 7]].concat(), beta], &[]) {
        let expected = [0, 0, inf.to_bits(), (s * s).to_bits(), 1.0f32.to_bits()];
        let exact = [u[2], u[3], u[5], u[6], u[7]].map(f32::to_bits) == expected && !u[6].is_normal();
        assert!(exact && [u[0], u[1], u[4]].iter().all(|x| x.is_nan()), "{ty} classes: {u:?}");
    }
}

fn out_witnesses<Q: ArrayElement + Float + Debug + Default, O: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture
) {
    let ty = format!("{:?} {:?}", Q::data_type(), O::data_type());
    let run = |name: &str, shape: [u32; 6], [q, prefix, qkd, u, h0]: [Vec<f32>; 5], slots: &[i32], scale: f32| {
        let (q, label) = (q.iter().map(|&x| Q::from(x).unwrap()), format!("{ty} BuildTreeOut {name}"));
        let q = q.collect::<Vec<Q>>();
        let outputs = out_check::<Q, O>(fixture, shape, (&q, &[prefix, qkd, u, h0], slots), scale, &FLAGS, 0, &label).0;
        outputs.map(|o| o.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<_>>())
    };
    for shader in [false, true] {
        let ((lo, hi), classes) = decay_set(0.0, shader);
        assert!(classes == 0 && 0.99 < lo && hi < 1.01, "shader {shader}: e^0 in [{lo}, {hi}]");
    }
    // Row 0: the q h0 dot 2^24 + 1 - 2^24 is +0 ascending, so the term (e (-1)) (+0) is -0 and o = +0; descending it
    // is 1, a nonzero term. Row 1: qkd u products 2^24, 1, -2^24 give +0 ascending, 1 descending. Row 2: qkd +0 times
    // u inf is NaN; 1 + 2^-8 and 1 + 3 2^-8 round to even, to 1 and 1 + 2^-6 in bf16 (toward zero 1 + 2^-7). Row 3:
    // the subnormal product 2^-65 2^-65.
    let (big, s, ties) = (2f32.powi(24), 2f32.powi(-65), [1.0 + 2f32.powi(-8), 1.0 + 3.0 * 2f32.powi(-8)]);
    let qkd = [[0.0; 4], [big, 1.0, -big, 0.0], [1.0, 0.0, 0.0, 0.0], [s, 0.0, 0.0, 0.0]].concat();
    let rest = [1.0, 1.0, 0.0, 0.0, 0.0];
    let u = [[1.0, 1.0, ties[0], ties[1], s], [1.0, f32::INFINITY, 0.0, 0.0, 0.0], rest, rest].concat();
    let inputs = [[vec![big, 1.0, -big], vec![0.0; 9]].concat(), vec![0.0; 4], qkd, u, vec![1.0; 15]];
    let rounded = [ties, [1.0, 1.0 + 2f32.powi(-6)]][usize::from(O::data_type() == DataType::BF16)];
    for o in run("order, zeros and rounding", [1, 4, 1, 1, 3, 5], inputs, &[0], -1.0) {
        let expected = [0, 0, rounded[0].to_bits(), rounded[1].to_bits(), (s * s).to_bits()];
        assert!([o[0], o[5], o[12], o[13], o[19]].map(f32::to_bits) == expected && o[11].is_nan(), "{ty}: {o:?}");
    }
    // The -0 accumulator start, first in the set algebra for every e: (e (-1)) (+0) is exactly -0; from +0 the sum is
    // +0 and stays +0 after the -0 chain product; from -0 it would be -0 both times, a disjoint single class.
    let only = |((lo, hi), mask): ((f64, f64), u8), class: u8| mask == class && lo > hi;
    for shader in [false, true] {
        let term = mul::<f32>(mul::<f32>(decay_set(0.0, shader), point(-1.0)), point(0.0));
        let product = mul::<f32>(point(-0.0), point(1.0));
        for (seed, class) in [(0.0, POS_ZERO), (-0.0, NEG_ZERO)] {
            let start = add::<f32>(point(seed), term);
            assert!(only(term, NEG_ZERO) && only(start, class) && only(add::<f32>(start, product), class), "{ty} -0");
        }
    }
    for o in run("+0 start", [1; 6], [vec![0.0], vec![0.0], vec![-0.0], vec![1.0], vec![1.0]], &[0], -1.0) {
        assert_eq!(o[0].to_bits(), 0, "{ty} +0 start");
    }
    // Dk 0 still writes every output: without h0 the qkd u chain 3 at any scale; with slot 0 the empty dot's
    // (e scale) (+0) is +0 at prefix 0 and scale -1, and NaN where e or scale is infinite or NaN: prefix +inf and NaN
    // at scale -1, and the finite prefixes 0, -1/2 and 1/2 at scale +inf and NaN.
    let (inf, nan, finite) = (f32::INFINITY, f32::NAN, [0.0, -0.5, 0.5]);
    let cases = [(-1.0, [0.0, inf, nan], [3.0, nan, nan]), (inf, finite, [nan; 3]), (nan, finite, [nan; 3])];
    for (scale, prefix, enabled) in cases {
        for (slots, expected) in [(&[][..], [3.0; 3]), (&[0][..], enabled)] {
            let inputs = [vec![], prefix.to_vec(), vec![1.0; 9], vec![1.0; 3], vec![]];
            for o in run(&format!("Dk 0 scale {scale}"), [1, 3, 1, 1, 0, 1], inputs, slots, scale) {
                KernelFixture::assert_bits(&expected, &o, &format!("{ty} Dk 0 scale {scale} slots {slots:?}"));
            }
        }
    }
}

/// Exact TreeUpdateSolve witnesses derived here, on both sides at both BVs and inside both oracles.
#[uzu_test]
fn tree_update_solve_witnesses() {
    let fixture = KernelFixture::new();
    solve_witnesses::<f32>(&fixture);
    solve_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Exact BuildTreeOut witnesses with FP32 output, for both QKT, on both sides under every flag and inside both oracles.
#[uzu_test]
fn build_tree_out_f32_output_witnesses() {
    let fixture = KernelFixture::new();
    out_witnesses::<f32, f32>(&fixture);
    out_witnesses::<bf16, f32>(&fixture);
    fixture.assert_clean();
}

/// The same witnesses with BF16 output, on their own so that a fault in the BF16 storage rounding fails only here.
#[uzu_test]
fn build_tree_out_bf16_output_witnesses() {
    let fixture = KernelFixture::new();
    out_witnesses::<f32, bf16>(&fixture);
    out_witnesses::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}

/// Optional presence fails before recording, also without work, and so does Out's head grouping, which the CPU fails
/// executing its queued command; the same encoding then records valid work. BV 64 is KernelPrecondition and F16 in
/// either Out type, like in Solve, KernelVariant, where the CPU panics at encode.
#[uzu_test]
fn tree_solve_out_presence_and_rejects() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let buffers = [0; 12].map(|_| fixture.guarded(&[sentinel::<f32>(); 64], sentinel::<f32>()));
    let args = buffers.each_ref().map(arg);
    let new_out = |use_h0| BuildTreeOutVulkanKernel::new(context, DataType::F32, DataType::F32, false, false, use_h0);
    let mut encoding = fixture.encoding();
    for (use_h0, n, wrong) in itertools::iproduct!([false, true], [2, 0], 0..4) {
        let solve = TreeUpdateSolveVulkanKernel::new(context, DataType::F32, 16, use_h0).expect("TreeUpdateSolve");
        let out = new_out(use_h0).expect("BuildTreeOut");
        let [kh0, v, prefix, beta, a, inverse, index, u, q, qkd, h0, o] = args.clone();
        let present = |position: usize, buffer| (use_h0 != (position == wrong)).then_some(buffer);
        let (kh0, solve_index) = (present(0, kh0), present(1, index.clone()));
        let (h0, out_index) = (present(2, h0), present(3, index));
        // SAFETY: the presence check panics before anything is recorded.
        let message = panics(|| unsafe {
            match wrong < 2 {
                true => solve.encode(kh0, v, prefix, beta, a, inverse, solve_index, u, 1, n, 1, 1, &mut encoding),
                false => out.encode(q, prefix, qkd, u, h0, out_index, o, 1.0, 1, n, 1, 1, 1, 1, &mut encoding),
            }
        });
        let (kernel, names) = (["TreeUpdateSolve", "BuildTreeOut"][wrong / 2], ["kh0", "h0_idx", "h0", "h0_indices"]);
        let presence = format!("{kernel}: argument '{}' must be present exactly when use_h0", names[wrong]);
        let expected = format!("assertion `left == right` failed: {presence}\n  left: {}\n right: {use_h0}", !use_h0);
        assert_eq!(message, expected, "use_h0 {use_h0}, N {n}");
    }
    let heads = "qk_heads != 0 && (batch_size == 0 || value_heads == 0 || value_heads >= qk_heads)";
    let out = new_out(false).expect("BuildTreeOut");
    for shape in [[1, 2, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], [1, 2, 2, 1, 1, 1], [1, 0, 2, 1, 1, 1]] {
        let ([b, n, hqk, hv, dk, dv], [_, _, prefix, _, _, _, _, u, q, qkd, _, o]) = (shape, args.clone());
        // SAFETY: the precondition panics before anything is recorded.
        let message = panics(|| unsafe {
            out.encode(q, prefix, qkd, u, None, None, o, 1.0, b, n, hqk, hv, dk, dv, &mut encoding)
        });
        assert_eq!(message, format!("BuildTreeOut: precondition {heads} violated"), "Vulkan {shape:?}");
        let cpu = panics(|| drop(cpu_out::<f32, f32>(shape, [false, false], (&[], &Default::default(), &[]), 1.0, 1)));
        assert_eq!(cpu, CPU_FAILURE, "CPU {shape:?}");
    }
    // The encoding stays usable: valid work recorded after the rejections, reading the sentinels -7, writes
    // u = a_inv (beta v) = -7 (-7 (-7)) and o = qkd u = (-7) (-7) into fresh guarded outputs.
    let solve = TreeUpdateSolveVulkanKernel::new(context, DataType::F32, 16, false).expect("TreeUpdateSolve");
    let outputs = [0; 2].map(|_| fixture.guarded(&[sentinel::<f32>()], sentinel::<f32>()));
    let [_, v, prefix, beta, a, inverse, _, u, q, qkd, ..] = args.clone();
    // SAFETY: at N = H = Dv = Dk = 1 without h0, Solve reads element 0 of v, prefix, beta and a_inv and no A, and Out
    // element 0 of qkd and u; the outputs alias nothing.
    unsafe {
        solve.encode(None, v, prefix.clone(), beta, a, inverse, None, arg(&outputs[0]), 1, 1, 1, 1, &mut encoding);
        out.encode(q, prefix, qkd, u, None, None, arg(&outputs[1]), 1.0, 1, 1, 1, 1, 1, 1, &mut encoding);
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for buffer in &buffers {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &[sentinel::<f32>(); 64], "buffer");
        }
        let written = outputs.each_ref().map(|output| KernelFixture::read_guarded(output, sentinel::<f32>())[0]);
        KernelFixture::assert_bits(&[-343.0, 49.0], &written, "valid work recorded after the rejections");
    }
    match TreeUpdateSolveVulkanKernel::new(context, DataType::F32, 64, false).err() {
        Some(Error::KernelPrecondition {
            kernel,
            condition,
        }) => assert_eq!((kernel, condition), ("TreeUpdateSolve", "BV == 16 || BV == 32")),
        _ => panic!("TreeUpdateSolve: BV 64 accepted"),
    }
    let variant = |error: Option<Error>| match error {
        Some(Error::KernelVariant {
            kernel,
            ..
        }) => kernel,
        _ => panic!("F16 accepted"),
    };
    assert_eq!(variant(TreeUpdateSolveVulkanKernel::new(context, DataType::F16, 32, false).err()), "TreeUpdateSolve");
    for [qkt, output] in [[DataType::F16, DataType::F32], [DataType::F32, DataType::F16]] {
        let out = BuildTreeOutVulkanKernel::new(context, qkt, output, false, false, false);
        assert_eq!(variant(out.err()), "BuildTreeOut", "Vulkan {qkt:?} {output:?}");
    }
    let missing = |types: &dyn Debug| format!("not implemented: variant doesn't exist: {types:?}");
    let solve = |bv: u32| panics(|| drop(cpu_solve::<f32>([1, 1, 1, 1], bv, (&[1.0], &Default::default(), &[]), 1)));
    assert_eq!(solve(64), missing(&(DataType::F32, 64)), "CPU BV 64");
    let f16_encode = panics(|| drop(cpu_solve::<f16>([1, 1, 1, 1], 16, (&[f16::ONE], &Default::default(), &[]), 1)));
    assert_eq!(f16_encode, missing(&(DataType::F16, 16)), "CPU F16");
    let qkt_f16 = panics(|| drop(cpu_out::<f16, f32>([1; 6], [false; 2], (&[], &Default::default(), &[]), 1.0, 1)));
    assert_eq!(qkt_f16, missing(&(DataType::F16, DataType::F32)), "CPU F16 QKT");
    let output_f16 = panics(|| drop(cpu_out::<f32, f16>([1; 6], [false; 2], (&[], &Default::default(), &[]), 1.0, 1)));
    assert_eq!(output_f16, missing(&(DataType::F32, DataType::F16)), "CPU F16 OutputT");
    fixture.assert_clean();
}

/// No batches, tokens or rows, v heads or values, u32::MAX elsewhere and empty ranges, with and without h0: nothing
/// is recorded or changed, and the CPU Solve returns early with its u untouched.
#[uzu_test]
fn tree_solve_out_zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (context, empty, m) = (&fixture.context, fixture.guarded::<f32>(&[], sentinel()), u32::MAX);
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    for use_h0 in [false, true] {
        let solve = TreeUpdateSolveVulkanKernel::new(context, DataType::F32, 32, use_h0).expect("TreeUpdateSolve");
        let out = BuildTreeOutVulkanKernel::new(context, DataType::F32, DataType::F32, false, false, use_h0);
        let (out, optional) = (out.expect("BuildTreeOut"), || use_h0.then(e));
        for [b, n, h, dv] in [[0, m, m, m], [m, 0, m, m], [m, m, 0, m], [m, m, m, 0]] {
            // SAFETY: without work nothing is indexed or recorded.
            unsafe {
                solve.encode(optional(), e(), e(), e(), e(), e(), optional(), e(), b, n, h, dv, &mut encoding);
                out.encode(e(), e(), e(), e(), optional(), optional(), e(), 1.0, b, n, m, h, m, dv, &mut encoding);
            }
            let slots = &[0][..usize::from(use_h0)];
            let u = cpu_solve::<f32>([b, n, h, dv], 16, (&[1.0], &Default::default(), slots), 1).remove(0).0;
            assert_eq!(u, [sentinel::<f32>()], "CPU TreeUpdateSolve early return at {:?}", [b, n, h, dv]);
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty") };
    fixture.assert_clean();
}

fn measure<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    n: u32,
) {
    let (size, ty, shape) = (size_of::<T>() as u64, format!("{:?}", T::data_type()), [1, n, 16, 48, 128, 128]);
    let (q, [prefix, beta, h0, a, qkd, inverse, kh0]) = tree_inputs::<T>(shape, "binary", &[0], true);
    // Before timing: finite prefixes whose exp sets hold no class on either side, nonzero kh0 and positive q and h0, so
    // every e takes the builtin exp and reaches the outputs; checking the first outputs then asserts every step finite.
    let finite = prefix.iter().all(|&p| p.is_finite() && decay_set(p, false).1 == 0 && decay_set(p, true).1 == 0);
    let positive = q.iter().all(|x| x.to_f32().unwrap() > 0.0) && h0.iter().all(|&x| x > 0.0);
    assert!(finite && positive && kh0.iter().all(|&x| x != 0.0), "{ty} N {n}: timed inputs");
    let median = |mut times: Vec<Duration>| {
        times.drain(..3);
        times.sort();
        times[times.len() / 2]
    };
    let print = |label: &str, bytes: u64, (cpu, gpu): (Vec<Duration>, Vec<(Duration, Duration)>)| {
        let wall = median(gpu.iter().map(|time| time.1).collect());
        let gpu = median(gpu.into_iter().map(|time| time.0).collect());
        let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
        eprintln!(
            "{label}: {bytes} B logical, provisional ({rate:.1} GB/s effective); GPU {gpu:?}, wall {wall:?}; CPU wall \
             {:?}",
            median(cpu)
        );
    };
    let (n64, d, v) = (u64::from(n), 128u64, conv1d_values::<T>(prefix.len() * 128, 6, 0));
    let (label, f32s) = (format!("{ty} TreeUpdateSolve N {n}"), [prefix.clone(), beta, a, inverse, kh0]);
    let ([u, _], times) = solve_check(fixture, [1, n, 48, 128], (&v, &f32s, &[0]), &[32], 13, &label);
    // Per v column: per token its prefix, beta, kh0 and v, per earlier token its u and the A entry of each of the
    // block's tokens, the block's a_inv and its u stores; and the slot.
    let block = |first: u64| 16.min(n64 - first) * (16 + size + 4 * first + 4 * 16.min(n64 - first)) + 4 * first;
    print(&label, 48 * d * (4 + (0..n64).step_by(16).map(block).sum::<u64>()), times);
    // Per output: the slot, its prefix, Dk q and h0 values, N qkd and u values and the store.
    let bytes = |o: u64| n64 * 48 * d * (8 + d * (size + 4) + 8 * n64 + o);
    let inputs = (&q[..], &[prefix, qkd, u, h0], &[0][..]);
    let label = format!("{ty} F32 BuildTreeOut N {n}");
    let (_, times) = out_check::<T, f32>(fixture, shape, inputs, -0.625, &FLAGS[..1], 13, &label);
    print(&label, bytes(4), times);
    let label = format!("{ty} BF16 BuildTreeOut N {n}");
    let (_, times) = out_check::<T, bf16>(fixture, shape, inputs, -0.625, &FLAGS[..1], 13, &label);
    print(&label, bytes(2), times);
}

/// Run alone, without sync validation: `... tree_solve_out_test::throughput -- --ignored --nocapture`. Synthetic inputs
/// at Qwen3.5-labelled shapes, not model files: Hv 48, Hqk 16, Dk = Dv = 128, slot 0, BV 32, binary tries of 49 and
/// 128 nodes in two rounds of opposite order, both dtypes and all four Out pairs; prefix, A, qkd, a_inv and kh0 from
/// the CPU Prefix and Gram, Out's u from the CPU Solve, the prefix finite so every e takes the builtin exp. Each side's
/// first output is checked by its monotone endpoint replay (Solve's conditional on its own u) before timing; then the
/// medians of 10 submissions after 3 warm-up ones, each into a fresh guarded output that, outside its measured time,
/// must equal the checked one bit for bit, its inputs and guards unchanged. Bytes are provisional logical loads and
/// stores, not bandwidth.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for lengths in [[49, 128], [128, 49]] {
        for n in lengths {
            measure::<f32>(&fixture, n);
            measure::<bf16>(&fixture, n);
        }
    }
    fixture.assert_clean();
}
