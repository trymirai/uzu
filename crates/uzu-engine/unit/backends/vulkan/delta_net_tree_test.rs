use std::{fmt::Debug, time::Duration};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    AttentionSinglePassCase, CPU_FAILURE, add, arg, assert_inputs, assert_same_bits, bounds, conv1d_values, cpu_buffer,
    cpu_submissions, decay, delta_net_check, exp, interval, kernel_fixture::KernelFixture, member, mul, negate, panics,
    point, sentinel, silu_set, submit,
};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::trie::TrieNode,
            kernel::{BuildTreePrefixKernel, ConvTreeScanKernel, StateAdvanceKernel},
        },
        cpu::Cpu,
        vulkan::{
            BuildTreePrefixVulkanKernel, ConvTreeScanVulkanKernel, Error, StateAdvanceVulkanKernel,
            VkCommandBufferEncoding,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// The guard of the u32 parent, trie and accepted-index words, none of which the tests store.
const WORD_SENTINEL: u32 = 0x5A5A_5A5A;

/// [C, P, K, Q] (conv_dim, total_proj_dim, kernel_size, suffix_len) of ConvTreeScan with the tree of its rows: a
/// model-like row with passthrough channels under each tree; one tap; two taps with passthrough; state channels past
/// total_proj_dim, which stay untouched; passthrough only; a channel tail past one workgroup; 70000 chained rows past
/// 65535 groups; row 0 under the later row 1.
const CONV_SHAPES: [([u32; 4], &str); 11] = [
    ([40, 53, 4, 7], "chain"),
    ([40, 53, 4, 7], "star"),
    ([40, 53, 4, 7], "binary"),
    ([40, 53, 4, 7], "random"),
    ([9, 9, 1, 5], "binary"),
    ([9, 12, 2, 5], "random"),
    ([12, 9, 4, 5], "binary"),
    ([0, 6, 4, 3], "chain"),
    ([67, 67, 4, 3], "star"),
    ([1, 2, 3, 70000], "chain"),
    ([40, 53, 4, 2], "reversed"),
];

/// [B, N, H] (batch_size, tree_size, value_heads) of BuildTreePrefix with the tree of its even batches (odd ones are
/// stars): each tree; two batches; one element; heads past one group; 65600 heads past 65535 groups.
const PREFIX_SHAPES: [([u32; 3], &str); 8] = [
    ([1, 49, 5], "chain"),
    ([1, 49, 5], "star"),
    ([1, 49, 5], "binary"),
    ([1, 49, 5], "random"),
    ([2, 17, 3], "random"),
    ([1, 1, 1], "chain"),
    ([1, 40, 67], "binary"),
    ([1, 2, 65600], "chain"),
];

/// Parent links of `count` rows: a chain, a star, a binary tree in preorder, AttentionSinglePassCase's random preorder
/// trie, or row 0 under row 1, which is no preorder.
fn tree(
    kind: &str,
    count: u32,
) -> Vec<Option<u32>> {
    match kind {
        "chain" => (0..count).map(|row| row.checked_sub(1)).collect(),
        "star" => (0..count).map(|row| (row > 0).then_some(0)).collect(),
        "binary" => {
            let (mut parents, mut stack) = (Vec::new(), vec![(0u32, None)]);
            while let Some((heap, parent)) = stack.pop() {
                if heap < count {
                    let node = Some(parents.len() as u32);
                    stack.extend([(2 * heap + 2, node), (2 * heap + 1, node)]);
                    parents.push(parent);
                }
            }
            parents
        },
        "random" => AttentionSinglePassCase::new(1, 1, 1, 0, count, 3).with_random_trie(5).parents.unwrap(),
        _ => vec![Some(1), None],
    }
}

/// ConvTreeScan's in_proj, [conv_weight, bias, base_state] over the owned channels min(C, P), the specials at every
/// `every`-th element of in_proj and base_state when nonzero, and the parents as -1 or row links.
fn conv_inputs<T: ArrayElement + Float>(
    shape: [u32; 4],
    kind: &str,
    every: usize,
) -> (Vec<T>, [Vec<f32>; 3], Vec<i32>) {
    let [c, p, k, q] = shape.map(|n| n as usize);
    let owned = c.min(p);
    let parents = tree(kind, q as u32).iter().map(|parent| parent.map_or(-1, |row| row as i32)).collect();
    let f32s = [conv1d_values(owned * k, 1, 0), conv1d_values(owned, 2, 0), conv1d_values(owned * (k - 1), 3, every)];
    (conv1d_values(q * p, 0, every), f32s, parents)
}

/// ConvTreeScan replayed in the CPU's order and FP32 roundings: every owned channel's accumulator in (row, channel)
/// order, and suffix_state holding the raw samples, every other element a sentinel.
fn conv_replay<T: ArrayElement + Float>(
    shape: [u32; 4],
    has_bias: bool,
    in_proj: &[T],
    [weight, bias, base]: &[Vec<f32>; 3],
    parents: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let [c, p, k, q] = shape.map(|n| n as usize);
    let (mut accs, mut state) = (Vec::new(), vec![sentinel::<f32>(); q * c * (k - 1)]);
    for (row, channel) in itertools::iproduct!(0..q, 0..c.min(p)) {
        let (mut acc, mut source) = (
            if has_bias {
                bias[channel]
            } else {
                0.0
            },
            row as i32,
        );
        for h in 0..k {
            let sample = match source >= 0 {
                true => in_proj[source as usize * p + channel].to_f32().unwrap(),
                false => base[channel * (k - 1) + k - 1 - source.unsigned_abs() as usize],
            };
            acc += sample * weight[channel * k + k - 1 - h];
            if h + 1 < k {
                state[(row * c + channel) * (k - 1) + k - 2 - h] = sample;
                // A row follows its parent link; a negative source, no index, steps further into the base state.
                source = parents.get(source as usize).copied().unwrap_or(source - 1);
            }
        }
        accs.push(acc);
    }
    (accs, state)
}

/// The CPU ConvTreeScan on a fresh context in `submissions` submissions: out_proj, suffix_state and the wall times.
fn cpu_conv<T: ArrayElement + Float + Default>(
    shape: [u32; 4],
    has_bias: bool,
    (in_proj, f32s, parents): (&[T], &[Vec<f32>; 3], &[u32]),
    submissions: usize,
) -> (Vec<T>, Vec<f32>, Vec<Duration>) {
    let [c, p, k, q] = shape;
    let state_len = q as usize * c as usize * (k as usize).saturating_sub(1);
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ConvTreeScanKernel::new(&context, T::data_type(), k, has_bias)
        .expect("CPU ConvTreeScan");
    let [weight, bias, base] = f32s.each_ref().map(|values| cpu_buffer(&context, values));
    let (projection, links) = (cpu_buffer(&context, in_proj), cpu_buffer(&context, parents));
    let mut out = cpu_buffer(&context, &vec![sentinel::<T>(); in_proj.len()]);
    let mut state = cpu_buffer(&context, &vec![sentinel::<f32>(); state_len]);
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let bias = has_bias.then_some(&bias);
        kernel.encode(&projection, &weight, bias, &base, &links, &mut out, &mut state, q, p, c, command_buffer);
    });
    (buffer_prefix_to_vec::<Cpu, T>(&out, in_proj.len()), buffer_prefix_to_vec::<Cpu, f32>(&state, state_len), times)
}

/// The Vulkan ConvTreeScan over guarded ranges, its outputs starting as sentinels, once or in timed submissions:
/// out_proj and suffix_state after asserting every input, the parent words and every guard.
fn gpu_conv<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &ConvTreeScanVulkanKernel,
    shape: [u32; 4],
    has_bias: bool,
    (in_proj, f32s, parents): (&[T], &[Vec<f32>; 3], &[u32]),
    timed: bool,
) -> (Vec<T>, Vec<f32>, Option<(Duration, Duration)>) {
    let [c, p, k, q] = shape;
    let state_len = q as usize * c as usize * (k as usize - 1);
    let projection = fixture.guarded(in_proj, sentinel::<T>());
    let inputs = f32s.each_ref().map(|values| fixture.guarded(values, sentinel::<f32>()));
    let links = fixture.guarded(parents, WORD_SENTINEL);
    let out = fixture.guarded(&vec![sentinel::<T>(); in_proj.len()], sentinel::<T>());
    let state = fixture.guarded(&vec![sentinel::<f32>(); state_len], sentinel::<f32>());
    // SAFETY: each range holds every element the shape addresses, aligned; the written ranges alias nothing.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let [weight, bias, base] = inputs.each_ref().map(arg);
        let (projection, links, out, state) = (arg(&projection), arg(&links), arg(&out), arg(&state));
        kernel.encode(projection, weight, has_bias.then_some(bias), base, links, out, state, q, p, c, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&projection, sentinel::<T>(), in_proj, "ConvTreeScan in_proj");
        assert_inputs(&inputs, &f32s.each_ref().map(Vec::as_slice), "ConvTreeScan input");
        KernelFixture::assert_unchanged(&links, WORD_SENTINEL, parents, "ConvTreeScan parents");
        (KernelFixture::read_guarded(&out, sentinel()), KernelFixture::read_guarded(&state, sentinel()), times)
    }
}

/// ConvTreeScan on the CPU and Vulkan against the replay, each independently: every owned output in the SiLU set of
/// its accumulator, every passthrough element the in_proj bits, suffix_state the raw samples and its untouched channels
/// sentinels bit for bit. Returns the replayed accumulators and the times.
fn conv_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    shape: [u32; 4],
    has_bias: bool,
    (in_proj, f32s, parents): (&[T], &[Vec<f32>; 3], &[i32]),
    submissions: usize,
    label: &str,
) -> (Vec<f32>, Option<(Duration, Duration)>, Vec<Duration>) {
    let kernel =
        ConvTreeScanVulkanKernel::new(&fixture.context, T::data_type(), shape[2], has_bias).expect("ConvTreeScan");
    let (accs, state) = conv_replay(shape, has_bias, in_proj, f32s, parents);
    let sets = accs.iter().map(|&acc| silu_set::<T>(f64::from(acc))).collect::<Vec<_>>();
    let [c, p, ..] = shape.map(|n| n as usize);
    let owned = c.min(p);
    let owner = (0..in_proj.len()).map(|i| (i % p < owned).then(|| i / p * owned + i % p)).collect::<Vec<_>>();
    let words = parents.iter().map(|&parent| parent as u32).collect::<Vec<_>>();
    let inputs = (in_proj, f32s, &words[..]);
    let (cpu_out, cpu_state, cpu_times) = cpu_conv(shape, has_bias, inputs, submissions);
    let (gpu_out, gpu_state, times) = gpu_conv(fixture, &kernel, shape, has_bias, inputs, submissions > 1);
    for (side, out, out_state) in [("CPU", &cpu_out, &cpu_state), ("Vulkan", &gpu_out, &gpu_state)] {
        delta_net_check(&sets, &owner, in_proj, out, &format!("{label} {side} out_proj"));
        assert_same_bits(&state, out_state, &format!("{label} {side} suffix_state"));
    }
    (accs, times, cpu_times)
}

fn conv_matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    for (shape, kind) in CONV_SHAPES {
        for has_bias in [false, true] {
            let (in_proj, f32s, parents) = conv_inputs::<T>(shape, kind, 7);
            let label = format!("ConvTreeScan {:?} {shape:?} {kind} bias {has_bias}", T::data_type());
            conv_check(fixture, shape, has_bias, (&in_proj, &f32s, &parents), 1, &label);
        }
    }
}

/// Every ConvTreeScan shape and tree, with and without a bias and the specials in in_proj and base_state: outputs
/// within the SiLU set of the replayed accumulator, passthrough channels and suffix_state bit for bit, untouched state
/// channels, every input, the parents and every guard unchanged, on the CPU and Vulkan.
#[uzu_test]
fn conv_tree_scan_matches_replay() {
    let fixture = KernelFixture::new();
    conv_matches::<f32>(&fixture);
    conv_matches::<bf16>(&fixture);
    fixture.assert_clean();
}

fn conv_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let (big, step, tiny, small) = (2f32.powi(24), 1.0 + 2f32.powi(-7), 2f32.powi(-17), 2f32.powi(-65));
    let (nan, ty) = (f32::from_bits(0x7FC1_2345), format!("{:?}", T::data_type()));
    // [K, bias, x, conv_weight, base_state, accumulator] of one row under the base state: products 2^24, 1, -2^24 in
    // history order cancel to +0 only ascending; -(1 + 2^-7 + 2^-17) then (1 + 2^-17)(1 + 2^-7), which rounds to its
    // magnitude, give +0 only unfused; +0 plus a -0 sample is +0; a -0 bias keeps -0; a subnormal product; the NaN
    // payload 0x7FC12345, 2^-149 and -0 recorded raw into suffix_state.
    let witnesses: [(u32, Option<f32>, f32, Vec<f32>, Vec<f32>, f32); 6] = [
        (3, None, 1.0, vec![-big, 1.0, big], vec![1.0, 1.0], 0.0),
        (2, None, 1.0, vec![step, -(1.0 + 2f32.powi(-7) + tiny)], vec![1.0 + tiny], 0.0),
        (1, None, -0.0, vec![1.0], vec![], 0.0),
        (1, Some(-0.0), -0.0, vec![1.0], vec![], -0.0),
        (1, None, small, vec![small], vec![], small * small),
        (4, None, -0.0, vec![1.0; 4], vec![1.0, f32::from_bits(1), nan], f32::NAN),
    ];
    for (index, (k, bias, x, weight, base, expected)) in witnesses.into_iter().enumerate() {
        let (shape, label) = ([1, 1, k, 1], format!("{ty} ConvTreeScan witness {index}"));
        let (f32s, projection) = ([weight, bias.into_iter().collect(), base], [T::from(x).unwrap()]);
        let inputs = (&projection[..], &f32s, &[-1][..]);
        let (accs, ..) = conv_check(fixture, shape, bias.is_some(), inputs, 1, &label);
        assert!(accs[0].to_bits() == expected.to_bits() || accs[0].is_nan() && expected.is_nan(), "{label}: {accs:?}");
    }
    let f32s = [vec![1.0; 4], vec![], vec![1.0, f32::from_bits(1), nan]];
    let (_, state) = conv_replay([1, 1, 4, 1], false, &[T::from(-0.0).unwrap()], &f32s, &[-1]);
    assert_same_bits(&[f32::from_bits(1), nan, -0.0], &state, &format!("{ty} raw state taps"));
}

/// Exact ConvTreeScan witnesses derived here, independently of the CPU code: the history order, the unfused product,
/// the +0 start, a -0 bias, a subnormal product and raw state taps, on the CPU and Vulkan.
#[uzu_test]
fn conv_tree_scan_witnesses() {
    let fixture = KernelFixture::new();
    conv_witnesses::<f32>(&fixture);
    conv_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// BuildTreePrefix replayed in the CPU's order: per (batch, row, head) the FP32 sum from +0 over columns ascending
/// whose interval holds the row.
fn prefix_replay(
    [b, n, h]: [u32; 3],
    nodes: &[TrieNode],
    log_decay: &[f32],
) -> Vec<f32> {
    let [b, n, h] = [b, n, h].map(|extent| extent as usize);
    let prefix = (0..b * n * h).map(|i| {
        let (batch, row, head) = (i / (n * h), (i / h % n) as u32, i % h);
        let holds = |&col: &usize| (nodes[batch * n + col].trie_start..=nodes[batch * n + col].trie_end).contains(&row);
        (0..n).filter(holds).fold(0.0f32, |sum, col| sum + log_decay[(batch * n + col) * h + head])
    });
    prefix.collect()
}

/// The CPU BuildTreePrefix on a fresh context in `submissions` submissions: the prefix and the wall times.
fn cpu_prefix(
    shape: [u32; 3],
    words: &[u32],
    log_decay: &[f32],
    submissions: usize,
) -> (Vec<f32>, Vec<Duration>) {
    let [b, n, h] = shape;
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::BuildTreePrefixKernel::new(&context).expect("CPU prefix");
    let (trie, decays) = (cpu_buffer(&context, words), cpu_buffer(&context, log_decay));
    let mut prefix = cpu_buffer(&context, &vec![sentinel::<f32>(); log_decay.len()]);
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        kernel.encode(&trie, &decays, &mut prefix, b, n, h, command_buffer);
    });
    (buffer_prefix_to_vec::<Cpu, f32>(&prefix, log_decay.len()), times)
}

/// BuildTreePrefix on the CPU in `submissions` submissions and on Vulkan, timed from 2, over the trie as its u32 words:
/// both prefixes equal to the replay bit for bit up to NaN payloads, log_decay, the trie words and every guard
/// unchanged. Returns the times.
fn prefix_check(
    fixture: &KernelFixture,
    shape: [u32; 3],
    nodes: &[TrieNode],
    log_decay: &[f32],
    submissions: usize,
    label: &str,
) -> (Option<(Duration, Duration)>, Vec<Duration>) {
    let [b, n, h] = shape;
    let (expected, words) = (prefix_replay(shape, nodes, log_decay), bytemuck::cast_slice::<TrieNode, u32>(nodes));
    let (cpu, cpu_times) = cpu_prefix(shape, words, log_decay, submissions);
    let kernel = &BuildTreePrefixVulkanKernel::new(&fixture.context).expect("BuildTreePrefix");
    let (trie, decays) = (fixture.guarded(words, WORD_SENTINEL), fixture.guarded(log_decay, sentinel::<f32>()));
    let prefix = fixture.guarded(&vec![sentinel::<f32>(); log_decay.len()], sentinel::<f32>());
    // SAFETY: each range holds every element the shape addresses, aligned; the prefix aliases nothing.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        kernel.encode(arg(&trie), arg(&decays), arg(&prefix), b, n, h, encoding)
    };
    let times = submit(fixture, submissions > 1, record);
    // SAFETY: every command buffer using these buffers has completed.
    let gpu = unsafe {
        KernelFixture::assert_unchanged(&trie, WORD_SENTINEL, words, &format!("{label} trie"));
        KernelFixture::assert_unchanged(&decays, sentinel::<f32>(), log_decay, &format!("{label} log_decay"));
        KernelFixture::read_guarded(&prefix, sentinel())
    };
    KernelFixture::assert_bits(&expected, &cpu, &format!("{label} CPU"));
    KernelFixture::assert_bits(&expected, &gpu, &format!("{label} Vulkan"));
    (times, cpu_times)
}

/// Every BuildTreePrefix shape and tree, the specials in log_decay: the CPU and Vulkan prefix equal to the replay.
#[uzu_test]
fn build_tree_prefix_matches_replay() {
    let fixture = KernelFixture::new();
    for (shape, kind) in PREFIX_SHAPES {
        let [b, n, h] = shape;
        let kinds = (0..b as usize).map(|batch| [kind, "star"][batch % 2]);
        let nodes = kinds.flat_map(|kind| AttentionSinglePassCase::nodes(&tree(kind, n))).collect::<Vec<_>>();
        let log_decay = conv1d_values::<f32>((b * n * h) as usize, 4, 9);
        prefix_check(&fixture, shape, &nodes, &log_decay, 1, &format!("BuildTreePrefix {shape:?} {kind}"));
    }
    fixture.assert_clean();
}

/// Exact BuildTreePrefix witnesses: a chain 2^24, 1, -2^24 cancelling to +0 only ascending; one -0 ancestor summing to
/// +0 from the +0 start; a row in no interval, +0; 2^-149 + 2^-149; NaN and the infinities.
#[uzu_test]
fn build_tree_prefix_witnesses() {
    let fixture = KernelFixture::new();
    let (big, tiny, inf) = (2f32.powi(24), f32::from_bits(1), f32::INFINITY);
    let chain = |n: u32| AttentionSinglePassCase::nodes(&tree("chain", n));
    let outside = vec![
        TrieNode {
            trie_start: 1,
            trie_end: 1,
            height: 0
        };
        2
    ];
    let witnesses = [
        (chain(3), vec![big, 1.0, -big], vec![big, big, 0.0]),
        (chain(1), vec![-0.0], vec![0.0]),
        (outside, vec![5.0, 7.0], vec![0.0, 12.0]),
        (chain(2), vec![tiny, tiny], vec![tiny, 2.0 * tiny]),
        (chain(3), vec![inf, -inf, f32::NAN], vec![inf, f32::NAN, f32::NAN]),
        (AttentionSinglePassCase::nodes(&tree("star", 3)), vec![-inf, 1.0, 2.0], vec![-inf; 3]),
    ];
    for (index, (nodes, log_decay, expected)) in witnesses.into_iter().enumerate() {
        let (shape, label) = ([1, nodes.len() as u32, 1], format!("BuildTreePrefix witness {index}"));
        KernelFixture::assert_bits(&expected, &prefix_replay(shape, &nodes, &log_decay), &format!("{label} replay"));
        prefix_check(&fixture, shape, &nodes, &log_decay, 1, &label);
    }
    fixture.assert_clean();
}

/// e^a as the shader's delta_net_exp (`shader`) or the CPU's expf decides it: NaN and +inf themselves; a <= 0, -inf
/// and both zeros included, the canonical decay oracle of e^-(-a), with Vulkan's exp bound and its flush of subnormal
/// results to +0 for the shader and 1 ULP for the CPU; finite a > 0 Vulkan's exp bound or 1 ULP.
fn decay_set(
    a: f32,
    shader: bool,
) -> ((f64, f64), u8) {
    let value = f64::from(a);
    match () {
        _ if a.is_nan() || a == f32::INFINITY => point(value),
        _ if a <= 0.0 => decay::<f32, f32>(point(-value), shader),
        _ => bounds::<f32>([interval(value.exp(), value.exp(), 1.0), exp(value).0][usize::from(shader)]),
    }
}

/// StateAdvance's [k_norm, v] for `tokens` nodes, k_norm through the last k head the CPU maps (beyond the nominal ones
/// for nondivisible ratios), and [log_decay, beta, state]: eighths, k scaled by 1/16, beta by 1/4, log_decay in
/// [-2, 0) or, when `positive`, (0, 2).
fn advance_inputs<T: ArrayElement + Float>(
    [hv, hk]: [u32; 2],
    tokens: usize,
    positive: bool,
) -> ([Vec<T>; 2], [Vec<f32>; 3]) {
    let [hv, hk] = [hv, hk].map(|n| n as usize);
    let keys = match hv {
        0 => 0,
        _ => (tokens - 1) * hk * 128 + ((hv - 1) / (hv / hk) + 1) * 128,
    };
    let scaled = |len: usize, seed: usize, scale: f32, shift: f32| -> Vec<f32> {
        conv1d_values::<f32>(len, seed, 0).iter().map(|x| x / scale + shift).collect()
    };
    let k = scaled(keys, 0, 16.0, 0.0).iter().map(|&x| T::from(x).unwrap()).collect();
    let log_decay = scaled(tokens * hv, 2, 4.0, [-1.0, 1.0][usize::from(positive)]);
    let f32s = [log_decay, scaled(tokens * hv, 3, 4.0, 0.0), conv1d_values(hv * 128 * 128, 4, 0)];
    ([k, conv1d_values(tokens * hv * 128, 1, 0)], f32s)
}

/// StateAdvance's allowance set of every state element on the shader or the CPU side: per row and accepted token in
/// order, d = `decay_set` with that side's exp bound, then over dk ascending s <- s d and kv from +0 adding s k,
/// delta = beta (v - kv), and over dk ascending s <- s + k delta, each an FP32 set operation on one member set per value,
/// with hk = hv / floor(Hv / Hk) as the CPU maps it; nothing is enumerated.
fn advance_sets<T: ArrayElement + Float>(
    [hv, hk]: [u32; 2],
    [k, v]: &[Vec<T>; 2],
    [log_decay, beta, state]: &[Vec<f32>; 3],
    accepted: &[u32],
    shader: bool,
) -> Vec<((f64, f64), u8)> {
    let [hv, hk] = [hv, hk].map(|n| n as usize);
    let x = |values: &[T], i: usize| point(values[i].to_f64().unwrap());
    let mut sets = state.iter().map(|&s| point(f64::from(s))).collect::<Vec<_>>();
    for (head, dv) in itertools::iproduct!(0..hv, 0..128) {
        let row = &mut sets[(head * 128 + dv) * 128..][..128];
        for &t in accepted {
            let (lane, key) = (t as usize * hv + head, (t as usize * hk + head / (hv / hk)) * 128);
            let (d, mut kv) = (decay_set(log_decay[lane], shader), point(0.0));
            for (dk, s) in row.iter_mut().enumerate() {
                *s = mul::<f32>(*s, d);
                kv = add::<f32>(kv, mul::<f32>(*s, x(k, key + dk)));
            }
            let delta = mul::<f32>(point(f64::from(beta[lane])), add::<f32>(x(v, lane * 128 + dv), negate(kv)));
            for (dk, s) in row.iter_mut().enumerate() {
                *s = add::<f32>(*s, mul::<f32>(x(k, key + dk), delta));
            }
        }
    }
    sets
}

/// The CPU StateAdvance on a fresh context, one fresh state per submission: every state and the wall times.
fn cpu_advance<T: ArrayElement + Float + Default>(
    [hv, hk]: [u32; 2],
    head_k_dim: u32,
    [k, v]: &[Vec<T>; 2],
    [log_decay, beta, state]: &[Vec<f32>; 3],
    accepted: &[u32],
    submissions: usize,
) -> (Vec<Vec<f32>>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(&context, T::data_type(), head_k_dim, hv, hk)
            .expect("CPU StateAdvance");
    let [k, v] = [k, v].map(|values| cpu_buffer(&context, values));
    let [log_decay, beta] = [log_decay, beta].map(|values| cpu_buffer(&context, values));
    let indices = cpu_buffer(&context, accepted);
    let mut bundles = (0..submissions).map(|_| cpu_buffer(&context, state)).collect::<Vec<_>>();
    let mut next = bundles.iter_mut();
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let state = next.next().expect("one state per submission");
        kernel.encode(&k, &v, &log_decay, &beta, &indices, state, accepted.len() as u32, command_buffer);
    });
    (bundles.iter().map(|bundle| buffer_prefix_to_vec::<Cpu, f32>(bundle, state.len())).collect(), times)
}

/// The Vulkan StateAdvance over guarded ranges, one fresh guarded state per submission (13 when `timed`): every
/// state after asserting the inputs, the accepted indices and every guard.
fn gpu_advance<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &StateAdvanceVulkanKernel,
    [k, v]: &[Vec<T>; 2],
    f32s: &[Vec<f32>; 3],
    accepted: &[u32],
    timed: bool,
) -> (Vec<Vec<f32>>, Option<(Duration, Duration)>) {
    let stored = [k, v].map(|values| fixture.guarded(values, sentinel::<T>()));
    let inputs = [&f32s[0], &f32s[1]].map(|values| fixture.guarded(values, sentinel::<f32>()));
    let indices = fixture.guarded(accepted, WORD_SENTINEL);
    let bundles =
        (0..1 + 12 * usize::from(timed)).map(|_| fixture.guarded(&f32s[2], sentinel::<f32>())).collect::<Vec<_>>();
    let mut next = bundles.iter();
    // SAFETY: each range holds every element the CPU mapping addresses, aligned; each submission writes its own state.
    let record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let ([k, v], [log_decay, beta]) = (stored.each_ref().map(arg), inputs.each_ref().map(arg));
        let state = arg(next.next().expect("one state per submission"));
        kernel.encode(k, v, log_decay, beta, arg(&indices), state, accepted.len() as u32, encoding)
    };
    let times = submit(fixture, timed, record);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&stored, &[&k[..], &v[..]], "StateAdvance k_norm and v");
        assert_inputs(&inputs, &[&f32s[0][..], &f32s[1][..]], "StateAdvance log_decay and beta");
        KernelFixture::assert_unchanged(&indices, WORD_SENTINEL, accepted, "StateAdvance accepted_indices");
        (bundles.iter().map(|state| KernelFixture::read_guarded(state, sentinel())).collect(), times)
    }
}

/// StateAdvance on the CPU in `submissions` submissions and on Vulkan, timed from 2, with a fresh state per submission:
/// every CPU state within the CPU's allowance sets and every Vulkan state within the shader's, each side independently.
/// Returns the shader's sets, the first CPU and Vulkan states and the times.
fn advance_check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    heads: [u32; 2],
    (stored, f32s, accepted): (&[Vec<T>; 2], &[Vec<f32>; 3], &[u32]),
    submissions: usize,
    label: &str,
) -> (Vec<((f64, f64), u8)>, [Vec<f32>; 2], Option<(Duration, Duration)>, Vec<Duration>) {
    let kernel =
        StateAdvanceVulkanKernel::new(&fixture.context, T::data_type(), 128, heads[0], heads[1]).expect("StateAdvance");
    let [sets, cpu_sets] = [true, false].map(|shader| advance_sets(heads, stored, f32s, accepted, shader));
    let owner = (0..sets.len()).map(Some).collect::<Vec<_>>();
    let (mut cpu, cpu_times) = cpu_advance(heads, 128, stored, f32s, accepted, submissions);
    let (mut states, times) = gpu_advance(fixture, &kernel, stored, f32s, accepted, submissions > 1);
    for (side, side_sets, side_states) in [("CPU", &cpu_sets, &cpu), ("Vulkan", &sets, &states)] {
        for (index, state) in side_states.iter().enumerate() {
            delta_net_check(side_sets, &owner, &f32s[2], state, &format!("{label} {side} bundle {index}"));
        }
    }
    (sets, [cpu.swap_remove(0), states.swap_remove(0)], times, cpu_times)
}

fn advance_matches<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    // [[Hv, Hk], tokens, accepted, positive log_decay]: one head; two v heads per k head over three tokens; the
    // nondivisible 3 over 2, mapping v head 2 to k head 2; a duplicate; an unordered path; growth; no token.
    let cases: [([u32; 2], usize, Vec<u32>, bool); 7] = [
        ([1, 1], 1, vec![0], false),
        ([2, 1], 3, vec![0, 1, 2], false),
        ([3, 2], 2, vec![0, 1], false),
        ([2, 1], 6, vec![5, 5], false),
        ([3, 2], 4, vec![3, 0, 2], false),
        ([1, 1], 2, vec![0, 1], true),
        ([2, 1], 2, vec![], false),
    ];
    for (heads, tokens, accepted, positive) in cases {
        let (stored, f32s) = advance_inputs::<T>(heads, tokens, positive);
        let label = format!("StateAdvance {:?} {heads:?} accepted {accepted:?} positive {positive}", T::data_type());
        advance_check(fixture, heads, (&stored, &f32s, &accepted), 1, &label);
    }
}

/// Every StateAdvance case over k_norm through its last mapped head: the CPU's and Vulkan's state within the allowance
/// sets, a path without tokens leaving it bit for bit, every input, the indices and every guard unchanged.
#[uzu_test]
fn state_advance_matches_allowance() {
    let fixture = KernelFixture::new();
    advance_matches::<f32>(&fixture);
    advance_matches::<bf16>(&fixture);
    fixture.assert_clean();
}

fn advance_witnesses<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
    let ty = format!("{:?}", T::data_type());
    let row = |head: &[f32], tail: f32| [head.to_vec(), vec![tail; 128 - head.len()]].concat();
    let to_t = |values: Vec<f32>| values.iter().map(|&x| T::from(x).unwrap()).collect::<Vec<T>>();
    // One head, one token: [state row 0, k, log_decay, beta, v[0]], the other rows +0.
    let inputs = |state: Vec<f32>, k: Vec<f32>, log_decay: f32, beta: f32, v: f32| {
        let f32s = [vec![log_decay], vec![beta], [state, vec![0.0; 127 * 128]].concat()];
        ([to_t(k), to_t(row(&[v], 0.0))], f32s)
    };
    let run = |(stored, f32s): ([Vec<T>; 2], [Vec<f32>; 3]), name: &str| {
        advance_check(fixture, [1, 1], (&stored, &f32s, &[0]), 1, &format!("{ty} StateAdvance {name}"))
    };
    // dk order: for every d of the allowance, s d = 1.5 2^26 d lies in [2^26, 2^27) at spacing 8 and each 3d below 4 is
    // absorbed, so ascending kv = s_0 d and row 0 becomes s_0 d - kv = +0, while descending adds 6d < 12 first and
    // rounds kv to s_0 d + 8, leaving -8. Asserted on the shader set's endpoints, each condition monotone in d; the CPU's
    // 1-ULP set lies inside it.
    let ((lo, hi), classes) = decay_set(0.0, true);
    for d in [lo, hi] {
        let first = 1.5 * 2f64.powi(26) * d;
        assert!(classes == 0 && (2f64.powi(26)..2f64.powi(27)).contains(&first), "{ty}: d {d}");
        assert!(2.5 < 3.0 * d && 3.0 * d < 3.9 && 5.0 < 6.0 * d && 6.0 * d < 7.8, "{ty}: d {d}");
    }
    let order = inputs(row(&[1.5 * 2f32.powi(26), 3.0, 3.0], 0.0), row(&[1.0; 3], 0.0), 0.0, 1.0, 0.0);
    let (_, states, ..) = run(order, "dk order");
    for state in &states {
        assert_eq!(state[0].to_bits(), 0, "{ty} dk order: row 0 {} where descending dk gives -8", state[0]);
    }
    // Decay before the dot: d = +0 makes kv = +0 and row 0 (+0) + 1 (-(+0)) = +0; the dot first gives kv = 1 and -1.
    let (sets, ..) = run(inputs(row(&[1.0], 0.0), vec![1.0; 128], f32::NEG_INFINITY, 1.0, 0.0), "decay first");
    assert!(sets[0] == point(0.0) && !member(sets[0], -1.0), "{ty} decay first: {:?}", sets[0]);
    // d = +0 leaves delta = beta v exactly, 1/2 3, so row 0 takes 1 delta and row 1 -2 delta; kv - v would negate them.
    let (sets, ..) =
        run(inputs(row(&[5.0, -3.0, 2.0], 0.0), row(&[1.0, -2.0], 0.0), f32::NEG_INFINITY, 0.5, 3.0), "beta v");
    assert_eq!(sets[..3], [point(1.5), point(-3.0), point(0.0)], "{ty} beta v");
    // -0 rows: -0 d = -0, kv = +0, delta = 1 (-0 - (+0)) = -0, and -0 + 0 (-0) = -0.
    let (sets, ..) = run(inputs(vec![-0.0; 128], vec![0.0; 128], 0.0, 1.0, -0.0), "-0");
    assert!(sets[..128].iter().all(|&set| set == point(-0.0)), "{ty} -0: {:?}", &sets[..4]);
    // +inf scales every element to NaN (s = 0) or a signed infinity; NaN decays give NaN rows; an infinite element times
    // the +0 decay is NaN. The sets decide each class exactly.
    let nan_row = |sets: &[((f64, f64), u8)]| sets[..128].iter().all(|&set| set == point(f64::NAN));
    let (sets, ..) = run(inputs(row(&[2.0, -2.0], 0.0), vec![1.0; 128], f32::INFINITY, 1.0, 0.0), "+inf");
    assert!(nan_row(&sets), "{ty} +inf");
    let (sets, ..) = run(inputs(row(&[1.0], 0.0), vec![1.0; 128], f32::NAN, 1.0, 0.0), "NaN");
    assert!(nan_row(&sets), "{ty} NaN");
    let (sets, ..) =
        run(inputs(row(&[f32::INFINITY], 0.0), row(&[1.0], 0.0), f32::NEG_INFINITY, 1.0, 0.0), "inf state");
    assert!(sets[0] == point(f64::NAN), "{ty} inf state");
}

/// Exact StateAdvance witnesses, F32 first: the robust dk-order and decay-before-dot separations, beta v under a +0
/// decay, -0 signs, and the +inf, NaN and infinite-state classes, on the CPU and Vulkan.
#[uzu_test]
fn state_advance_witnesses() {
    let fixture = KernelFixture::new();
    advance_witnesses::<f32>(&fixture);
    advance_witnesses::<bf16>(&fixture);
    fixture.assert_clean();
}

/// ConvTreeScan's bias presence, the generated binding's invariant, failing before anything is recorded, also without
/// work; each constructor precondition as KernelPrecondition, with the CPU's own outcome on a fresh context: its
/// kernel_size - 1 overflow panic, which only debug builds check (release wraps it, so it is asserted only with debug
/// assertions), its variant panic at encode for HEAD_K_DIM 64, its divisions by num_k_heads = 0 and by
/// floor(Hv / Hk) = 0, also without tokens; F16 as KernelVariant.
#[uzu_test]
fn tree_presence_and_rejects() {
    let fixture = KernelFixture::new();
    let context = &fixture.context;
    let buffers = [0; 7].map(|_| fixture.guarded(&[sentinel::<f32>(); 64], sentinel::<f32>()));
    let [a, b, c, d, e, f, g] = buffers.each_ref().map(arg);
    let mut encoding = fixture.encoding();
    for (has_bias, q) in [(true, 2), (true, 0), (false, 2), (false, 0)] {
        let conv = ConvTreeScanVulkanKernel::new(context, DataType::F32, 4, has_bias).expect("ConvTreeScan");
        let wrong = (!has_bias).then(|| c.clone());
        // SAFETY: the presence check panics before anything is recorded.
        let message = panics(|| unsafe {
            let (a, b, d, e, f, g) = (a.clone(), b.clone(), d.clone(), e.clone(), f.clone(), g.clone());
            conv.encode(a, b, wrong, d, e, f, g, q, 8, 4, &mut encoding)
        });
        let presence = "ConvTreeScan: argument 'bias' must be present exactly when has_bias";
        let expected =
            format!("assertion `left == right` failed: {presence}\n  left: {}\n right: {has_bias}", !has_bias);
        assert_eq!(message, expected, "bias {has_bias}, Q {q}");
    }
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        for buffer in &buffers {
            KernelFixture::assert_unchanged(buffer, sentinel::<f32>(), &[sentinel::<f32>(); 64], "buffer");
        }
    }
    let rejected = |kernel: &str, condition: &str, error: Option<Error>| match error {
        Some(Error::KernelPrecondition {
            kernel: name,
            condition: text,
        }) => assert_eq!((name, text), (kernel, condition)),
        _ => panic!("{kernel}: {condition} accepted"),
    };
    rejected("ConvTreeScan", "kernel_size >= 1", ConvTreeScanVulkanKernel::new(context, DataType::F32, 0, false).err());
    let heads = "num_k_heads != 0 && (num_v_heads == 0 || num_v_heads >= num_k_heads)";
    for (dim, [hv, hk], condition) in [
        (64, [1, 1], "HEAD_K_DIM == 128"),
        (64, [1, 0], "HEAD_K_DIM == 128"),
        (128, [1, 0], heads),
        (128, [1, 2], heads),
    ] {
        rejected("StateAdvance", condition, StateAdvanceVulkanKernel::new(context, DataType::F32, dim, hv, hk).err());
    }
    if cfg!(debug_assertions) {
        let inputs = (&[1.0f32][..], &[vec![], vec![], vec![]], &[u32::MAX][..]);
        assert_eq!(panics(|| drop(cpu_conv([1, 1, 0, 1], false, inputs, 1))), CPU_FAILURE, "CPU kernel_size 0");
    }
    let advance = |heads: [u32; 2], dim: u32| {
        panics(|| drop(cpu_advance::<f32>(heads, dim, &Default::default(), &Default::default(), &[], 1)))
    };
    let variant = format!("not implemented: variant doesn't exist: {:?}", (DataType::F32, 64));
    assert_eq!(advance([1, 1], 64), variant, "CPU HEAD_K_DIM 64");
    assert_eq!(advance([1, 0], 128), CPU_FAILURE, "CPU num_k_heads 0");
    assert_eq!(advance([1, 2], 128), CPU_FAILURE, "CPU Hv < Hk");
    let variant = |kernel: &str, error: Option<Error>| match error {
        Some(Error::KernelVariant {
            kernel: name,
            ..
        }) => assert_eq!(name, kernel),
        _ => panic!("{kernel}: F16 accepted"),
    };
    variant("ConvTreeScan", ConvTreeScanVulkanKernel::new(context, DataType::F16, 4, false).err());
    variant("StateAdvance", StateAdvanceVulkanKernel::new(context, DataType::F16, 128, 1, 1).err());
    fixture.assert_clean();
}

/// No work at u32::MAX elsewhere and empty ranges: ConvTreeScan without rows or channels, BuildTreePrefix without
/// batches, rows or heads, StateAdvance without tokens or v heads. Nothing is recorded or changed.
#[uzu_test]
fn tree_zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (context, empty, m) = (&fixture.context, fixture.guarded::<f32>(&[], sentinel()), u32::MAX);
    let e = || arg(&empty);
    let conv = ConvTreeScanVulkanKernel::new(context, DataType::F32, 4, false).expect("ConvTreeScan");
    let prefix = BuildTreePrefixVulkanKernel::new(context).expect("BuildTreePrefix");
    let mut encoding = fixture.encoding();
    // SAFETY: without work nothing is indexed or recorded.
    unsafe {
        for [q, p] in [[0, m], [m, 0]] {
            conv.encode(e(), e(), None, e(), e(), e(), e(), q, p, m, &mut encoding);
        }
        for [b, n, h] in [[0, m, m], [m, 0, m], [m, m, 0]] {
            prefix.encode(e(), e(), e(), b, n, h, &mut encoding);
        }
        for [hv, len] in [[m, 0], [0, m]] {
            let advance = StateAdvanceVulkanKernel::new(context, DataType::F32, 128, hv, 1).expect("StateAdvance");
            advance.encode(e(), e(), e(), e(), e(), e(), len, &mut encoding);
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, sentinel::<f32>(), &[], "empty") };
    fixture.assert_clean();
}

fn tree_measure<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    reversed: bool,
) {
    let (size, ty) = (size_of::<T>() as u64, format!("{:?}", T::data_type()));
    let print =
        |kernel: &str, extent: u32, bytes: u64, (times, mut cpu): (Option<(Duration, Duration)>, Vec<Duration>)| {
            let (gpu, wall) = times.expect("timed");
            cpu.drain(..3);
            cpu.sort();
            let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
            eprintln!(
                "{kernel} {ty} {extent}: {bytes} B logical, provisional ({rate:.1} GB/s effective); GPU {gpu:?}, wall \
             {wall:?}; CPU wall {:?}",
                cpu[cpu.len() / 2]
            );
        };
    let order = |mut values: [u32; 2]| {
        if reversed {
            values.reverse();
        }
        values
    };
    let (c, p, k) = (10240u32, 16480u32, 4u32);
    for q in order([49, 128]) {
        let shape = [c, p, k, q];
        let (in_proj, f32s, parents) = conv_inputs::<T>(shape, "binary", 0);
        let label = format!("{ty} ConvTreeScan Q {q}");
        let (_, times, cpu) = conv_check(fixture, shape, true, (&in_proj, &f32s, &parents), 13, &label);
        // Per conv element: K samples (T rows or FP32 base taps), K weights, the parent words of the walk, K - 1 state
        // taps, the bias and the output; passthrough elements one load and one store.
        let walk = |row: usize| {
            let (mut source, mut rows, mut links) = (row as i32, 0u64, 0u64);
            for h in 0..k {
                rows += u64::from(source >= 0);
                if h + 1 < k {
                    links += u64::from(source >= 0);
                    source = parents.get(source as usize).copied().unwrap_or(source - 1);
                }
            }
            rows * size + (u64::from(k) - rows) * 4 + 4 * links
        };
        let conv = (0..q as usize).map(walk).sum::<u64>() + u64::from(q) * (8 * u64::from(k) + size);
        print("ConvTreeScan", q, u64::from(c) * conv + u64::from(q) * u64::from(p - c) * 2 * size, (times, cpu));
        let (n, h) = (q, 48u32);
        let nodes = AttentionSinglePassCase::nodes(&tree("binary", n));
        let log_decay = conv1d_values::<f32>((n * h) as usize, 4, 0).iter().map(|x| x / 8.0).collect::<Vec<_>>();
        let result = prefix_check(fixture, [1, n, h], &nodes, &log_decay, 13, &format!("BuildTreePrefix N {n}"));
        // Per (row, head): every trie node, the log_decay of each ancestor and the store.
        let ancestors = nodes.iter().map(|node| u64::from(node.height) + 1).sum::<u64>();
        print("BuildTreePrefix", n, u64::from(h) * (u64::from(n * n) * 12 + 4 * ancestors + 4 * u64::from(n)), result);
    }
    for len in order([1, 4]) {
        let (heads, accepted) = ([48, 16], [0u32, 3, 5, 7][..len as usize].to_vec());
        let (stored, f32s) = advance_inputs::<T>(heads, 8, false);
        let label = format!("{ty} StateAdvance L {len}");
        let (_, _, times, cpu) = advance_check(fixture, heads, (&stored, &f32s, &accepted), 13, &label);
        // Per state row: the row loaded and stored once; per token the index, log_decay, beta, k twice and v.
        print("StateAdvance", len, 48 * 128 * (1024 + u64::from(len) * (12 + 257 * size)), (times, cpu));
    }
}

/// Run alone, without sync validation: `... delta_net_tree_test::throughput -- --ignored --nocapture`. Synthetic inputs
/// at the Qwen3.5-labelled shapes, not verified model files: ConvTreeScan (conv_dim 10240, total_proj_dim 16480,
/// kernel size 4, a binary tree) and BuildTreePrefix (48 heads) for 49 and 128 nodes, StateAdvance (48 v heads, 16 k
/// heads, finite varying log_decay) for 1 and 4 accepted tokens, in two rounds of opposite order. Prints the GPU and
/// wall medians of 10 Vulkan submissions after 3 warm-up ones and the CPU kernels' wall medians. StateAdvance writes its
/// state, so every CPU and Vulkan submission gets its own fresh state, each checked against its side's allowance sets
/// (about 16M set operations per side at L 4); the other outputs are checked after timing. The bytes are the logical
/// source loads and stores, provisional until reviewed, not measured bandwidth.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for reversed in [false, true] {
        eprintln!("DeltaNet tree throughput round reversed {reversed}");
        tree_measure::<f32>(&fixture, reversed);
        tree_measure::<bf16>(&fixture, reversed);
    }
    fixture.assert_clean();
}
