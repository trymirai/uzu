use std::{fmt::Debug, time::Duration};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    AttentionSinglePassCase, CPU_FAILURE, add, arg, assert_inputs, conv1d_values, cpu_buffer, cpu_submissions,
    decay_set, delta_net_check, delta_net_tree, kernel_fixture::KernelFixture, mul, negate, panics, point, sentinel,
    submit,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, gpu_types::trie::TrieNode, kernel::BuildTreeGramKernel},
        cpu::Cpu,
        vulkan::{BuildTreeGramVulkanKernel, Error, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// [B, N, Hk, Hv, Dk, Dv] with each batch's h0 slot (none: use_h0 false): one block; 17 rows; four blocks; a negative
/// second slot; the 3 over 2 ratio mapping v head 2 to k head 2; 65600 heads past 65535 groups; no h0; Dk 0; Dv 0.
const SHAPES: [([u32; 6], &[i32]); 9] = [
    ([1, 16, 1, 1, 4, 3], &[0]),
    ([1, 17, 1, 2, 5, 4], &[0]),
    ([1, 49, 2, 4, 8, 6], &[0]),
    ([2, 33, 1, 3, 4, 2], &[0, -1]),
    ([1, 40, 2, 3, 3, 1], &[0]),
    ([1, 1, 1, 65600, 1, 1], &[0]),
    ([1, 17, 1, 2, 5, 4], &[]),
    ([1, 17, 1, 1, 0, 2], &[0]),
    ([1, 17, 1, 1, 2, 0], &[0]),
];

fn extents(
    [b, n, _, hv, _, dv]: [u32; 6],
    use_h0: bool,
) -> [usize; 4] {
    let [b, n, hv, dv] = [b, n, hv, dv].map(|extent| extent as usize);
    let blocks = n.div_ceil(16);
    let kh0 = usize::from(use_h0) * b * n * hv * dv;
    [b * hv * blocks * blocks.div_ceil(2) * 512, b * hv * n * n, b * hv * blocks * 256, kh0]
}

/// Synthetic inputs, not model data: q and k through the CPU's last mapped k head, prefix specials every 9th element.
pub fn gram_inputs<T: ArrayElement + Float>(
    shape: [u32; 6],
    kind: &str,
    slots: &[i32],
) -> ([Vec<T>; 2], Vec<TrieNode>, [Vec<f32>; 3]) {
    let [b, n, hk, hv, dk, dv] = shape.map(|extent| extent as usize);
    let keys = match b * n * hv {
        0 => 0,
        _ => ((b * n - 1) * hk + (hv - 1) / (hv / hk) + 1) * dk,
    };
    let kinds = (0..b).map(|batch| [kind, "star"][batch % 2]);
    let nodes = kinds.flat_map(|kind| AttentionSinglePassCase::nodes(&delta_net_tree(kind, n as u32))).collect();
    let states = slots.iter().map(|&slot| slot + 1).max().unwrap_or(0).max(0) as usize;
    let lanes = b * n * hv;
    let f32s = [conv1d_values(lanes, 2, 9), conv1d_values(lanes, 3, 0), conv1d_values(states * hv * dv * dk, 4, 0)];
    ([0, 1].map(|seed| conv1d_values(keys, seed, 0)), nodes, f32s)
}

/// Per qkd and owned A element the FP32 p_r - p_c, the factor and the ascending dot, or index 0 for +0; A slack has no
/// owner. kh0 exact, sentinels under a negative slot.
fn gram_replay<T: ArrayElement + Float>(
    shape: [u32; 6],
    scale: f32,
    ([q, k], nodes, [prefix, beta, h0], slots): (&[Vec<T>; 2], &[TrieNode], &[Vec<f32>; 3], &[i32]),
) -> (Vec<Option<[f32; 3]>>, [Vec<Option<usize>>; 2], Vec<f32>) {
    let [b, n, hk, hv, dk, dv] = shape.map(|extent| extent as usize);
    let [a_len, qkd_len, _, kh0_len] = extents(shape, !slots.is_empty());
    let (blocks, pairs) = (n.div_ceil(16), n.div_ceil(16).div_ceil(2));
    let (mut terms, mut qkd_owner, mut a_owner) = (vec![None], vec![Some(0); qkd_len], vec![None; a_len]);
    let mut kh0 = vec![sentinel::<f32>(); kh0_len];
    let dot =
        |x: &[T], y: &[T]| x.iter().zip(y).fold(0.0f32, |s, (u, w)| s + u.to_f32().unwrap() * w.to_f32().unwrap());
    for (batch, head) in itertools::iproduct!(0..b, 0..hv) {
        let key = |row: usize| ((batch * n + row) * hk + head / (hv / hk)) * dk;
        let lane = |row: usize| (batch * n + row) * hv + head;
        let holds = |row: usize, col: usize| {
            (nodes[batch * n + col].trie_start..=nodes[batch * n + col].trie_end).contains(&(row as u32))
        };
        let mut term = |factor: f32, x: &[T], y: &[T], row: usize, col: usize| {
            terms.push(Some([prefix[lane(row)] - prefix[lane(col)], factor, dot(x, y)]));
            Some(terms.len() - 1)
        };
        for (row, col) in itertools::iproduct!(0..n, 0..n).filter(|&(row, col)| holds(row, col)) {
            let index = ((batch * hv + head) * n + row) * n + col;
            qkd_owner[index] = term(scale, &q[key(row)..][..dk], &k[key(col)..][..dk], row, col);
        }
        let entries = itertools::iproduct!(0..blocks, 0..pairs, 0..512).filter(|&(block, pair, _)| pair <= block / 2);
        for (block, pair, entry) in entries {
            let (row, col) = (block * 16 + entry / 32, pair * 32 + entry % 32);
            a_owner[(((batch * hv + head) * blocks + block) * pairs + pair) * 512 + entry] =
                match row != col && row < n && col < n && holds(row, col) {
                    true => term(beta[lane(row)], &k[key(row)..][..dk], &k[key(col)..][..dk], row, col),
                    false => Some(0),
                };
        }
        let slot = slots.get(batch).copied().unwrap_or(-1);
        for (row, dv_index) in itertools::iproduct!(0..n, 0..dv).filter(|_| slot >= 0) {
            let state = &h0[((slot as usize * hv + head) * dv + dv_index) * dk..][..dk];
            let values = k[key(row)..][..dk].iter().zip(state);
            kh0[lane(row) * dv + dv_index] = values.fold(0.0f32, |s, (x, &h)| s + x.to_f32().unwrap() * h);
        }
    }
    (terms, [qkd_owner, a_owner], kh0)
}

/// a_inv replayed in the CPU's row order from the diagonal blocks of one side's own returned A, identity-padded.
fn inverse_replay(
    shape: [u32; 6],
    a: &[f32],
) -> Vec<f32> {
    let [b, n, _, hv, ..] = shape.map(|extent| extent as usize);
    let (blocks, pairs) = (n.div_ceil(16), n.div_ceil(16).div_ceil(2));
    let mut inverse = vec![0.0f32; b * hv * blocks * 256];
    for (head, block) in itertools::iproduct!(0..b * hv, 0..blocks) {
        let tile = &a[((head * blocks + block) * pairs + block / 2) * 512 + block % 2 * 16..];
        let out = &mut inverse[(head * blocks + block) * 256..][..256];
        (0..16).for_each(|i| out[i * 17] = 1.0);
        for row in 0..16.min(n - block * 16) {
            for col in 0..row {
                out[row * 16 + col] = -(col..row).fold(0.0f32, |s, p| s + tile[row * 32 + p] * out[p * 16 + col]);
            }
        }
    }
    inverse
}

pub fn cpu_gram<T: ArrayElement + Float + Default>(
    shape: [u32; 6],
    [mxu, use_h0]: [bool; 2],
    scale: f32,
    ([q, k], nodes, f32s, slots): (&[Vec<T>; 2], &[TrieNode], &[Vec<f32>; 3], &[i32]),
    submissions: usize,
) -> ([Vec<f32>; 4], Vec<Duration>) {
    let [b, n, hk, hv, dk, dv] = shape;
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::BuildTreeGramKernel::new(&context, T::data_type(), mxu, use_h0)
            .expect("CPU BuildTreeGram");
    let [q, k] = [q, k].map(|values| cpu_buffer(&context, values));
    let trie = cpu_buffer(&context, bytemuck::cast_slice::<TrieNode, u32>(nodes));
    let index = cpu_buffer(&context, bytemuck::cast_slice::<i32, u32>(slots));
    let [prefix, beta, h0] = f32s.each_ref().map(|values| cpu_buffer(&context, values));
    let lengths = extents(shape, use_h0);
    let mut outputs = lengths.map(|len| cpu_buffer(&context, &vec![sentinel::<f32>(); len]));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let [a, qkd, inverse, kh0] = outputs.each_mut();
        let (h0, index, kh0) = (use_h0.then_some(&h0), use_h0.then_some(&index), use_h0.then_some(kh0));
        let (q, k, trie, prefix, beta) = (&q, &k, &trie, &prefix, &beta);
        kernel.encode(
            q,
            k,
            trie,
            prefix,
            beta,
            h0,
            index,
            a,
            qkd,
            inverse,
            kh0,
            scale,
            b,
            n,
            hk,
            hv,
            dk,
            dv,
            command_buffer,
        );
    });
    (std::array::from_fn(|i| buffer_prefix_to_vec::<Cpu, f32>(&outputs[i], lengths[i])), times)
}

fn gpu_gram<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &BuildTreeGramVulkanKernel,
    shape: [u32; 6],
    scale: f32,
    ([q, k], nodes, f32s, slots): (&[Vec<T>; 2], &[TrieNode], &[Vec<f32>; 3], &[i32]),
    timed: bool,
) -> ([Vec<f32>; 4], Option<(Duration, Duration)>) {
    let ([b, n, hk, hv, dk, dv], use_h0, word) = (shape, !slots.is_empty(), sentinel::<f32>().to_bits());
    let words = [bytemuck::cast_slice::<TrieNode, u32>(nodes), bytemuck::cast_slice::<i32, u32>(slots)];
    let stored = [q, k].map(|values| fixture.guarded(values, sentinel::<T>()));
    let inputs = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    let [trie, index] = words.map(|values| fixture.guarded(values, word));
    let outputs = extents(shape, use_h0).map(|len| fixture.guarded(&vec![sentinel::<f32>(); len], sentinel::<f32>()));
    // SAFETY: each range holds every element the shape addresses, aligned; the outputs alias nothing.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let ([q, k], [prefix, beta, h0]) = (stored.each_ref().map(arg), inputs.each_ref().map(arg));
        let [a, qkd, inverse, kh0] = outputs.each_ref().map(arg);
        let (h0, slot, kh0) = (use_h0.then_some(h0), use_h0.then(|| arg(&index)), use_h0.then_some(kh0));
        let trie = arg(&trie);
        kernel.encode(q, k, trie, prefix, beta, h0, slot, a, qkd, inverse, kh0, scale, b, n, hk, hv, dk, dv, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&stored, &[&q[..], &k[..]], "BuildTreeGram q and k");
        assert_inputs(&inputs, &f32s.each_ref().map(Vec::as_slice), "BuildTreeGram prefix, beta and h0");
        KernelFixture::assert_unchanged(&trie, word, words[0], "BuildTreeGram trie");
        KernelFixture::assert_unchanged(&index, word, words[1], "BuildTreeGram h0_idx");
        (outputs.each_ref().map(|output| KernelFixture::read_guarded(output, sentinel())), times)
    }
}

/// Each side independently: qkd and owned A within its own sets (exp set, factor, dot in the CPU's association), +0
/// exact, A slack untouched; then, conditional on that verified A, a_inv replayed from its own A; kh0 exact.
fn gram_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 6],
    scale: f32,
    inputs: (&[Vec<T>; 2], &[TrieNode], &[Vec<f32>; 3], &[i32]),
    submissions: usize,
    label: &str,
) -> ([[Vec<f32>; 4]; 2], Option<(Duration, Duration)>, Vec<Duration>) {
    let use_h0 = !inputs.3.is_empty();
    let kernel =
        BuildTreeGramVulkanKernel::new(&fixture.context, T::data_type(), false, use_h0).expect("BuildTreeGram");
    let (terms, [qkd_owner, a_owner], kh0) = gram_replay(shape, scale, inputs);
    let (cpu, cpu_times) = cpu_gram(shape, [false, use_h0], scale, inputs, submissions);
    let (gpu, times) = gpu_gram(fixture, &kernel, shape, scale, inputs, submissions > 1);
    for (shader, side, [a, qkd, inverse, out]) in [(false, "CPU", &cpu), (true, "Vulkan", &gpu)] {
        let set = |&term: &Option<[f32; 3]>| match term {
            None => point(0.0),
            Some([x, factor, dot]) => {
                mul::<f32>(mul::<f32>(decay_set(x, shader), point(factor.into())), point(dot.into()))
            },
        };
        let sets = terms.iter().map(set).collect::<Vec<_>>();
        delta_net_check(&sets, &qkd_owner, &vec![sentinel::<f32>(); qkd.len()], qkd, &format!("{label} {side} qkd"));
        delta_net_check(&sets, &a_owner, &vec![sentinel::<f32>(); a.len()], a, &format!("{label} {side} A"));
        let condition = format!("{label} {side} a_inv, conditional on its own verified A");
        KernelFixture::assert_bits(&inverse_replay(shape, a), inverse, &condition);
        KernelFixture::assert_bits(&kh0, out, &format!("{label} {side} kh0"));
    }
    ([cpu, gpu], times, cpu_times)
}

fn gram_matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for (index, (shape, slots)) in SHAPES.into_iter().enumerate() {
        let kind = ["chain", "star", "binary", "random"][index % 4];
        let (stored, nodes, f32s) = gram_inputs::<T>(shape, kind, slots);
        let label = format!("BuildTreeGram {:?} {shape:?} {kind} slots {slots:?}", T::data_type());
        gram_check(fixture, shape, 0.375, (&stored, &nodes, &f32s, slots), 1, &label);
    }
}

/// Every shape, tree and slot set with prefix specials, through `gram_check` on the CPU and Vulkan.
#[uzu_test]
fn build_tree_gram_matches_allowance() {
    let fixture = KernelFixture::new();
    gram_matches::<f32>(&fixture);
    gram_matches::<bf16>(&fixture);
    fixture.assert_clean();
}

fn gram_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let ty = format!("{:?}", T::data_type());
    let (big, huge, m) = (2f32.powi(24), 2f32.powi(100), 1.5 * 2f32.powi(26));
    let run = |name: &str, [n, dk, dv]: [u32; 3], kind: &str, [q, k, prefix, beta, h0]: [Vec<f32>; 5], scale: f32| {
        let stored = [q, k].map(|values| values.iter().map(|&x| T::from(x).unwrap()).collect::<Vec<T>>());
        let (nodes, f32s) = (AttentionSinglePassCase::nodes(&delta_net_tree(kind, n)), [prefix, beta, h0]);
        let label = format!("{ty} BuildTreeGram {name}");
        gram_check(fixture, [1, n, 1, 1, dk, dv], scale, (&stored, &nodes, &f32s, &[0]), 1, &label).0
    };
    // First, on each side: finite positive exp sets, e^0 in (0.99, 1.01) and e^-80 2^100 in [2^-16, 2^-14].
    let e0 = [true, false].map(|shader| decay_set(0.0, shader));
    for (shader, ((lo, hi), classes)) in [true, false].into_iter().zip(e0) {
        let ((low, high), scaled) = mul::<f32>(decay_set(-80.0, shader), point(huge.into()));
        assert!(classes == 0 && 0.99 < lo && hi < 1.01, "{ty} shader {shader}: e {lo} {hi}");
        assert!(scaled == 0 && 2f64.powi(-16) <= low && high <= 2f64.powi(-14), "{ty} shader {shader}: e 2^100");
    }
    // Dot order: q1 k0 and k1 k0 products 2^24, 1, -2^24 cancel to +0 ascending; descending d gives 1.
    let order = vec![1.0, 1.0, 1.0, big, 1.0, -big];
    let ordered = [order.clone(), order, vec![0.0; 2], vec![1.0; 2], vec![0.0; 3]];
    for [a, qkd, ..] in run("dot order", [2, 3, 1], "chain", ordered, 1.0) {
        assert!(qkd[2].to_bits() == 0 && a[32].to_bits() == 0, "{ty} dot order: qkd {} A {}", qkd[2], a[32]);
    }
    // Association, qk = kk = 2^100 at x = -80: (e scale) qk and (beta e) kk are finite; scale qk or beta kk is +inf.
    let wide = [vec![0.0, 2f32.powi(50)], vec![2f32.powi(50); 2], vec![0.0, -80.0], vec![1.0, huge], vec![0.0]];
    for [a, qkd, ..] in run("association", [2, 1, 1], "chain", wide, huge) {
        assert!(qkd[2].is_finite() && a[32].is_finite(), "{ty} association: qkd {} A {}", qkd[2], a[32]);
    }
    // A = +0 from orthogonal keys gives a_inv -(+0 + A 1) = -0 below the diagonal; 0 - s would give +0.
    let orthogonal = [vec![1.0; 4], vec![1.0, 0.0, 0.0, 1.0], vec![0.0; 2], vec![1.0; 2], vec![0.0; 2]];
    for [a, _, inverse, _] in run("a_inv sign", [2, 2, 1], "chain", orthogonal, 1.0) {
        let negative_zero = inverse[16].to_bits() == (-0.0f32).to_bits();
        assert!(a[32].to_bits() == 0 && negative_zero, "{ty} a_inv sign: {}", inverse[16]);
    }
    // a_inv p order on the returned A: kk30 = M = 1.5 2^26, kk31 = kk32 = 3, kk10 = kk20 = -1, kk21 = +0. Every A set
    // member has A30 in [2^26, 2^27) at spacing 8, products A31 (-A10), A32 (-A20) in (2.5, 3.9) and their sum in
    // (5, 7.8): ascending p absorbs both into -A30, descending gives -(A30 + 8).
    let within = |((lo, hi), classes): ((f64, f64), u8), low: f64, high: f64| classes == 0 && low < lo && hi < high;
    for (shader, e) in [true, false].into_iter().zip(e0) {
        let a = |kk: f64| mul::<f32>(mul::<f32>(point(1.0), e), point(kk));
        let [a30, a31, a32, a10, a20] = [f64::from(m), 3.0, 3.0, -1.0, -1.0].map(a);
        let [p1, p2] = [(a31, a10), (a32, a20)].map(|(x, y)| mul::<f32>(x, negate(y)));
        let top = a30.1 == 0 && 2f64.powi(26) <= a30.0.0 && a30.0.1 < 2f64.powi(27);
        let products = within(p1, 2.5, 3.9) && within(p2, 2.5, 3.9) && within(add::<f32>(p1, p2), 5.0, 7.8);
        assert!(top && products, "{ty} shader {shader}: a_inv order sets");
    }
    let keys = vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -2.0, 1.0, 0.0, m, m, 3.0, 9.0, 0.0];
    let chained = [keys.clone(), keys, vec![0.0; 4], vec![1.0; 4], vec![0.0; 5]];
    for [a, _, inverse, _] in run("a_inv order", [4, 5, 1], "chain", chained, 1.0) {
        assert_eq!(inverse[48].to_bits(), (-a[96]).to_bits(), "{ty} a_inv order: {} for A30 {}", inverse[48], a[96]);
    }
    // kh0 exactly: the exact product -(1 + 2^-7 + 2^-17), then (1 + 2^-7)(1 + 2^-17) rounding to its magnitude, so
    // the unfused dot is +0 while fusing the second product keeps 2^-24; 2^-65 2^-65 is the subnormal 2^-130.
    let (step, tiny, small) = (1.0 + 2f32.powi(-7), 2f32.powi(-17), 2f32.powi(-65));
    let h0 = vec![-(step + tiny), 1.0 + tiny, small, 0.0];
    let fused = [vec![0.0; 4], vec![1.0, step, small, 0.0], vec![0.0; 2], vec![1.0; 2], h0];
    for [.., kh0] in run("kh0", [2, 2, 2], "chain", fused, 1.0) {
        let subnormal = kh0[3] == small * small && !kh0[3].is_normal();
        assert!(kh0[0].to_bits() == 0 && subnormal, "{ty} kh0: {kh0:?}");
    }
    // A NaN root prefix: NaN in the root's whole column, which every row descends from; non-ancestors (0, 1), (2, 1) +0.
    let root = [vec![1.0; 3], vec![1.0; 3], vec![f32::NAN, 0.0, 0.0], vec![1.0; 3], vec![0.0]];
    for [a, qkd, ..] in run("NaN prefix", [3, 1, 1], "star", root, 1.0) {
        let column = [qkd[0], qkd[3], qkd[6], a[32], a[64]].iter().all(|x| x.is_nan());
        assert!(column && qkd[1].to_bits() == 0 && a[65].to_bits() == 0 && qkd[4] > 0.0, "{ty} NaN prefix: {qkd:?}");
    }
    // p_1 = -inf: e = +0, so qkd(1, 0) = (+0 1)(-2^100) = -0 and A(1, 0) = (+0)(2^200 = +inf) = NaN.
    let negative = [vec![0.0, -1.0], vec![huge; 2], vec![0.0, f32::NEG_INFINITY], vec![1.0; 2], vec![0.0]];
    for [a, qkd, ..] in run("-inf prefix", [2, 1, 1], "chain", negative, 1.0) {
        assert!(qkd[2].to_bits() == (-0.0f32).to_bits() && a[32].is_nan(), "{ty} -inf prefix: {} {}", qkd[2], a[32]);
    }
    // Dk 0 still writes every output: p_1 = +inf gives e = +inf and (+inf 1)(+0) = NaN in qkd(1, 0) and A(1, 0).
    let empty = [vec![], vec![], vec![0.0, f32::INFINITY], vec![1.0; 2], vec![]];
    for [a, qkd, inverse, kh0] in run("Dk 0", [2, 0, 1], "chain", empty, 1.0) {
        let zeros = qkd[0].to_bits() == 0 && kh0.iter().all(|x| x.to_bits() == 0) && inverse[0] == 1.0;
        assert!(qkd[2].is_nan() && a[32].is_nan() && zeros, "{ty} Dk 0: {qkd:?} {kh0:?}");
    }
}

/// Exact witnesses derived here, on both sides.
#[uzu_test]
fn build_tree_gram_witnesses() {
    let fixture = KernelFixture::new();
    gram_witnesses::<f32>(&fixture);
    gram_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Optional presence and the k-head precondition fail before recording, also without work, while the CPU fails at
/// execution; USE_MXU is KernelPrecondition while the CPU panics at encode; F16 is KernelVariant.
#[uzu_test]
fn build_tree_gram_presence_and_rejects() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let buffers = [0; 11].map(|_| fixture.guarded(&[sentinel::<f32>(); 64], sentinel::<f32>()));
    let args = buffers.each_ref().map(arg);
    let mut encoding = fixture.encoding();
    for (use_h0, n, wrong) in itertools::iproduct!([false, true], [2, 0], 0..3) {
        let kernel = BuildTreeGramVulkanKernel::new(context, DataType::F32, false, use_h0).expect("BuildTreeGram");
        let [q, k, trie, prefix, beta, h0, index, a, qkd, inverse, kh0] = args.clone();
        let present = |position: usize, buffer| (use_h0 != (position == wrong)).then_some(buffer);
        let (h0, index, kh0) = (present(0, h0), present(1, index), present(2, kh0));
        // SAFETY: the presence check panics before anything is recorded.
        let message = panics(|| unsafe {
            kernel.encode(
                q,
                k,
                trie,
                prefix,
                beta,
                h0,
                index,
                a,
                qkd,
                inverse,
                kh0,
                1.0,
                1,
                n,
                1,
                1,
                1,
                1,
                &mut encoding,
            )
        });
        let name = ["h0", "h0_idx", "kh0"][wrong];
        let presence = format!("BuildTreeGram: argument '{name}' must be present exactly when use_h0");
        let expected = format!("assertion `left == right` failed: {presence}\n  left: {}\n right: {use_h0}", !use_h0);
        assert_eq!(message, expected, "use_h0 {use_h0}, N {n}");
    }
    let cpu = |shape: [u32; 6], mxu: bool| {
        panics(|| {
            drop(cpu_gram::<f32>(shape, [mxu, false], 1.0, (&Default::default(), &[], &Default::default(), &[]), 1))
        })
    };
    let kernel = BuildTreeGramVulkanKernel::new(context, DataType::F32, false, false).expect("BuildTreeGram");
    let heads = "k_heads != 0 && (batch_size == 0 || value_heads == 0 || value_heads >= k_heads)";
    for shape in [[1, 2, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], [1, 2, 2, 1, 1, 1], [1, 0, 2, 1, 1, 1]] {
        let [b, n, hk, hv, dk, dv] = shape;
        let [q, k, trie, prefix, beta, _, _, a, qkd, inverse, _] = args.clone();
        // SAFETY: the precondition panics before anything is recorded.
        let message = panics(|| unsafe {
            kernel.encode(
                q,
                k,
                trie,
                prefix,
                beta,
                None,
                None,
                a,
                qkd,
                inverse,
                None,
                1.0,
                b,
                n,
                hk,
                hv,
                dk,
                dv,
                &mut encoding,
            )
        });
        assert_eq!(message, format!("BuildTreeGram: precondition {heads} violated"), "Vulkan {shape:?}");
        assert_eq!(cpu(shape, false), CPU_FAILURE, "CPU {shape:?}");
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for buffer in &buffers {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &[sentinel::<f32>(); 64], "buffer");
        }
    }
    match BuildTreeGramVulkanKernel::new(context, DataType::F32, true, false).err() {
        Some(Error::KernelPrecondition {
            kernel,
            condition,
        }) => assert_eq!((kernel, condition), ("BuildTreeGram", "!USE_MXU")),
        _ => panic!("BuildTreeGram: USE_MXU accepted"),
    }
    let variant = format!("not implemented: variant doesn't exist: {:?}", (DataType::F32, true));
    assert_eq!(cpu([1, 1, 1, 1, 1, 1], true), variant, "CPU USE_MXU");
    match BuildTreeGramVulkanKernel::new(context, DataType::F16, false, false).err() {
        Some(Error::KernelVariant {
            kernel,
            ..
        }) => assert_eq!(kernel, "BuildTreeGram"),
        _ => panic!("BuildTreeGram: F16 accepted"),
    }
    fixture.assert_clean();
}

/// No batches, rows or v heads, u32::MAX elsewhere and empty ranges, with and without h0: nothing recorded or changed.
#[uzu_test]
fn build_tree_gram_zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (context, empty, m) = (&fixture.context, fixture.guarded::<f32>(&[], sentinel()), u32::MAX);
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    for use_h0 in [false, true] {
        let kernel = BuildTreeGramVulkanKernel::new(context, DataType::F32, false, use_h0).expect("BuildTreeGram");
        let optional = || use_h0.then(e);
        for [b, n, hv] in [[0, m, m], [m, 0, m], [m, m, 0]] {
            let (h0, index, kh0) = (optional(), optional(), optional());
            // SAFETY: without work nothing is indexed or recorded.
            unsafe {
                kernel.encode(
                    e(),
                    e(),
                    e(),
                    e(),
                    e(),
                    h0,
                    index,
                    e(),
                    e(),
                    e(),
                    kh0,
                    1.0,
                    b,
                    n,
                    m,
                    hv,
                    m,
                    m,
                    &mut encoding,
                )
            };
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty") };
    fixture.assert_clean();
}

fn gram_measure<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    reversed: bool,
) {
    let (size, ty) = (size_of::<T>() as u64, format!("{:?}", T::data_type()));
    let mut lengths = [49u32, 128];
    if reversed {
        lengths.reverse();
    }
    for n in lengths {
        let shape = [1, n, 16, 48, 128, 128];
        let (stored, nodes, mut f32s) = gram_inputs::<T>(shape, "binary", &[0]);
        f32s[0] = conv1d_values::<f32>((n * 48) as usize, 2, 0).iter().map(|x| x / 8.0).collect();
        assert!(f32s[0].iter().all(|x| x.is_finite()), "{ty} N {n}: timed prefix");
        let label = format!("{ty} BuildTreeGram N {n}");
        let scale = 128f32.sqrt().recip();
        let (_, times, mut cpu) = gram_check(fixture, shape, scale, (&stored, &nodes, &f32s, &[0]), 13, &label);
        // Per v head as executed: each invocation's prefix row per qkd row and slot per block; per qkd element trie node
        // and store, per ancestor q, k rows and column prefix; per owned A entry store, per in-range off-diagonal trie
        // node, per proper ancestor two k rows, both prefixes, beta; 256 a_inv stores per block; per kh0 element Dk k
        // and h0 values and the store.
        let (n64, d, blocks) = (u64::from(n), 128u64, u64::from(n).div_ceil(16));
        let ancestors = nodes.iter().map(|node| u64::from(node.trie_end - node.trie_start) + 1).sum::<u64>();
        let tiles = (0..blocks).map(|block| block / 2 + 1).sum::<u64>();
        let rows = |block: u64| 16.min(n64 - 16 * block);
        let off_diagonal = (0..blocks).map(|block| rows(block) * ((block / 2 + 1) * 32).min(n64) - rows(block));
        let qkd = 64 * n64 * 4 + n64 * n64 * 16 + ancestors * (2 * d * size + 4);
        let a = tiles * 512 * 4 + off_diagonal.sum::<u64>() * 12 + (ancestors - n64) * (2 * d * size + 12);
        let bytes = 48 * (qkd + a + blocks * 256 * 4 + 64 * blocks * 4 + n64 * d * (d * (size + 4) + 4));
        let (gpu, wall) = times.expect("timed");
        cpu.drain(..3);
        cpu.sort();
        let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
        eprintln!(
            "{label}: {bytes} B logical, provisional ({rate:.1} GB/s effective); GPU {gpu:?}, wall {wall:?}; CPU wall \
             {:?}",
            cpu[cpu.len() / 2]
        );
    }
}

/// Run alone, without sync validation: `... tree_gram_test::throughput -- --ignored --nocapture`. Synthetic inputs,
/// finite prefixes, Qwen3.5-labelled head shapes (Hv 48, Hk 16, Dk = Dv = 128), not model files: binary trees of 49
/// and 128 nodes, slot 0, two rounds of opposite order; medians of 10 submissions after 3 warm-up ones, each side's
/// final outputs fully checked. Bytes are logical source loads and stores, not measured bandwidth.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for reversed in [false, true] {
        eprintln!("BuildTreeGram throughput round reversed {reversed}");
        gram_measure::<f32>(&fixture, reversed);
        gram_measure::<bf16>(&fixture, reversed);
    }
    fixture.assert_clean();
}
