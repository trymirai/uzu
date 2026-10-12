use std::time::Duration;

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::WeaverTopChildrenKernel,
        },
        cpu::Cpu,
        vulkan::vk_kernels::{
            TestExactFmaVulkanKernel, TestWeaverTopValuesVulkanKernel, WeaverTopChildrenVulkanKernel,
        },
    },
    encodable_block::sampling::{gumbel_float, gumbel_float_full_word, revidx},
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

const SENTINEL: u32 = 0xa5a5_a5a5;
const PATTERNS: [u32; 20] = [
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x33800000,
    0xb3800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321, 0x3f800001, 0x00800001,
];

fn inputs(
    shape: [u32; 4],
    mode: u32,
) -> (Vec<bf16>, Vec<f32>, Vec<u32>, Vec<u64>, Vec<u32>) {
    let [rows, candidates, _, vocab] = shape;
    let count = rows as usize * candidates as usize;
    let mut residual =
        (0..count).map(|i| bf16::from_f32(((i as f32 * 0.017).cos() * 4.0).round() * 0.125)).collect::<Vec<_>>();
    let mut logits = (0..count).map(|i| ((i as f32 * 0.011).sin() * 3.0).round() * 0.125).collect::<Vec<_>>();
    let mut ids =
        (0..rows).flat_map(|row| (0..candidates).rev().map(move |i| 70_000 + row * candidates + i)).collect::<Vec<_>>();
    let mut seeds = vec![0x9e37_79b9_7f4a_7c15, 0xd1b5_4a32_d192_ed03, 0x2545_f491_4f6c_dd1d, u64::MAX];
    let mut metadata = vec![0; rows as usize * MetadataIdx::COUNT];
    for row in 0..rows as usize {
        metadata[MetadataIdx::Depth as usize * rows as usize + row] = (row % seeds.len()) as u32;
        metadata[MetadataIdx::TreeSlot as usize * rows as usize + row] = row as u32;
    }
    match mode {
        1 => {
            residual.fill(bf16::ZERO);
            logits.fill(0.0);
            ids.fill(u32::MAX);
        },
        2 => {
            residual.fill(bf16::ZERO);
            for (i, value) in logits.iter_mut().enumerate() {
                *value = 67_108_864.0 + (i % 4) as f32 * 8.0;
            }
        },
        3 => {
            for (i, value) in residual.iter_mut().enumerate() {
                *value = bf16::from_bits((i % 7 + 1) as u16);
            }
            for (i, value) in logits.iter_mut().enumerate() {
                *value = f32::from_bits((i as u32 % 7) | ((i as u32 % 2) << 31));
            }
        },
        4 => {
            residual.fill(bf16::ZERO);
            seeds.fill(2_871_038);
            for row in 0..rows as usize {
                for i in 0..candidates as usize {
                    logits[row * candidates as usize + i] = -1000.0 - i as f32 * 32.0;
                }
                if candidates >= 2 {
                    logits[row * candidates as usize] = 0.0;
                    logits[row * candidates as usize + 1] = 100.0;
                    ids[row * candidates as usize] = 2048;
                    ids[row * candidates as usize + 1] = 1;
                }
            }
        },
        5 => {
            assert_eq!(candidates, 2);
            residual.fill(bf16::ZERO);
            seeds = (0..rows).map(|i| 0xd1b5_4a32_d192_ed03u64.wrapping_mul(u64::from(i) + 1)).collect();
            for row in 0..rows as usize {
                metadata[MetadataIdx::Depth as usize * rows as usize + row] = row as u32;
                ids[2 * row] = 70_000 + 2 * row as u32;
                ids[2 * row + 1] = ids[2 * row] + 1;
                // The host's two perturbed scores tie exactly by commutativity; GPU log rounding can separate them.
                logits[2 * row] = gumbel_float_full_word(seeds[row], revidx(ids[2 * row + 1], vocab));
                logits[2 * row + 1] = gumbel_float_full_word(seeds[row], revidx(ids[2 * row], vocab));
            }
        },
        6 => {
            residual.fill(bf16::ZERO);
            let words = [0xffc0_0000, 0x7fc0_0000, 0x7f80_0000, 0xff80_0000, 0, 0x8000_0000, 1, 0x8000_0001];
            for (i, value) in logits.iter_mut().enumerate() {
                *value = f32::from_bits(words[i % words.len()]);
            }
        },
        _ => {},
    }
    (residual, logits, ids, seeds, metadata)
}

fn cpu(
    data: (&[bf16], &[f32], &[u32], &[u64], &[u32]),
    shape: [u32; 4],
    scale: Option<f32>,
    tree: &[u32],
) -> (Vec<u32>, Duration) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::WeaverTopChildrenKernel::new(&context, scale.is_some())
        .expect("CPU WeaverTopChildren");
    let residual = cpu_buffer(&context, data.0);
    let logits = cpu_buffer(&context, data.1);
    let ids = cpu_buffer(&context, data.2);
    let seeds = cpu_buffer(&context, data.3);
    let metadata = cpu_buffer(&context, data.4);
    let (rows, capacity) = (shape[0] as usize, shape[0] as usize * shape[2] as usize);
    let tree = cpu_buffer(&context, tree);
    let valid = cpu_buffer(&context, &vec![1u32; rows]);
    let mut frontier = cpu_buffer(&context, &vec![SENTINEL; FrontierIdx::COUNT * capacity]);
    let times = cpu_submissions(&context, 1, |encoding| {
        kernel.encode(
            &residual,
            &logits,
            &ids,
            &seeds,
            &metadata,
            &tree,
            &valid,
            &mut frontier,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            capacity as u32,
            shape[0],
            scale,
            encoding,
        );
    });
    (buffer_prefix_to_vec::<Cpu, u32>(&frontier, FrontierIdx::COUNT * capacity), times[0])
}

// Tokens and model logprobs (the path over a +0 parent with pruning when the tree holds +0 paths, the edge without), then prune logprobs, per slot.
fn fields(
    frontier: &[u32],
    capacity: usize,
    scale: Option<f32>,
) -> [Vec<u32>; 3] {
    let field = |field: FrontierIdx| frontier[field as usize * capacity..][..capacity].to_vec();
    match scale {
        None => [field(FrontierIdx::TokenId), field(FrontierIdx::EdgeLogprobBits), vec![SENTINEL; capacity]],
        Some(_) => {
            [field(FrontierIdx::TokenId), field(FrontierIdx::PathLogprobBits), field(FrontierIdx::EdgeLogprobBits)]
        },
    }
}

// Observe the score stage's words, including its NaNs, then independently sort on the host. ExactFma(x,1,y) verifies
// finite score sums independently; NaN words need not match between different arithmetic operations.
fn gpu_scores(
    fixture: &KernelFixture,
    residual: &[bf16],
    logits: &[f32],
    ids: &[u32],
    seeds: &[u64],
    metadata: &[u32],
    shape: [u32; 4],
) -> Vec<u32> {
    let [rows, candidates, _, vocab] = shape;
    let words = ids
        .iter()
        .enumerate()
        .flat_map(|(i, &token)| {
            let row = i / candidates as usize;
            let seed = seeds[metadata[MetadataIdx::Depth as usize * rows as usize + row] as usize];
            [logits[i].to_bits(), u32::from(residual[i].to_bits()), seed as u32, (seed >> 32) as u32, token, vocab]
        })
        .collect::<Vec<_>>();
    let input = fixture.guarded(&words, SENTINEL);
    let output = fixture.guarded(&vec![SENTINEL; ids.len() * 4], SENTINEL);
    let stage = TestWeaverTopValuesVulkanKernel::new(&fixture.context).expect("weaver score stage");
    let mut encoding = fixture.encoding();
    // SAFETY: complete, distinct guarded six-word input and four-word output records, one per candidate.
    unsafe {
        stage.encode(arg(&input), arg(&output), ids.len() as u32, &mut encoding);
    }
    KernelFixture::complete(encoding);
    // SAFETY: the stage submission has completed; input and all guards are checked.
    let records = unsafe {
        KernelFixture::assert_unchanged(&input, SENTINEL, &words, "score stage records");
        KernelFixture::read_guarded::<u32>(&output, SENTINEL)
    };
    let combined = logits.iter().zip(residual).map(|(&a, &b)| a + b.to_f32()).collect::<Vec<_>>();
    let values = records.as_chunks::<4>().0;
    for (i, (&logit, value)) in combined.iter().zip(values).enumerate() {
        if logit.is_nan() {
            assert!(f32::from_bits(value[0]).is_nan());
        } else {
            assert_eq!(value[0], logit.to_bits(), "candidate/residual staged sum");
        }
        let record = &words[i * 6..(i + 1) * 6];
        let seed = u64::from(record[2]) | u64::from(record[3]) << 32;
        let expected = gumbel_float_full_word(seed, revidx(record[4], record[5]));
        let noise = f32::from_bits(value[2]);
        if expected.is_finite() {
            assert!(noise.is_finite() && (noise - expected).abs() <= 1e-5, "Metal's full-word selection noise");
        } else {
            assert_eq!(noise.to_bits(), expected.to_bits(), "selection noise endpoint");
        }
        let target = gumbel_float(seed, revidx(record[4], record[5]));
        let prune = f32::from_bits(value[3]);
        assert!(prune.is_finite() && (prune - target).abs() <= 1e-5, "the target sampler's pruning noise");
    }
    let triples = combined
        .iter()
        .zip(values)
        .flat_map(|(&logit, value)| [logit.to_bits(), 1.0f32.to_bits(), value[2]])
        .collect::<Vec<_>>();
    let input = fixture.guarded(&triples, SENTINEL);
    let output = fixture.guarded(&vec![SENTINEL; ids.len()], SENTINEL);
    let addition = TestExactFmaVulkanKernel::new(&fixture.context).expect("exact score addition");
    let mut encoding = fixture.encoding();
    // SAFETY: one aligned FP32 triple and one output per candidate; no aliases.
    unsafe {
        addition.encode(arg(&input), arg(&output), ids.len() as u32, &mut encoding);
    }
    KernelFixture::complete(encoding);
    // SAFETY: the exact-add submission has completed.
    let exact = unsafe {
        KernelFixture::assert_unchanged(&input, SENTINEL, &triples, "score triples");
        KernelFixture::read_guarded::<u32>(&output, SENTINEL)
    };
    for ((&logit, value), exact) in combined.iter().zip(values).zip(exact) {
        if logit.is_finite() {
            assert_eq!(value[1], exact, "finite score addition");
        }
    }
    values.iter().map(|v| v[1]).collect()
}

fn compare_logprob(
    actual: u32,
    expected: u32,
    tolerance: f32,
) {
    let actual = f32::from_bits(actual);
    let expected = f32::from_bits(expected);
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected.is_infinite() {
        assert_eq!(actual.to_bits(), expected.to_bits());
    } else {
        let step = expected.abs().next_up() - expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance + 4.0 * step,
            "expected {expected:?}, actual {actual:?}"
        );
    }
}

fn check(
    fixture: &KernelFixture,
    shape: [u32; 4],
    mode: u32,
    scale: Option<f32>,
    timed: bool,
) -> Option<(Duration, Duration, Duration)> {
    let data = inputs(shape, mode);
    let refs = (data.0.as_slice(), data.1.as_slice(), data.2.as_slice(), data.3.as_slice(), data.4.as_slice());
    let (rows, capacity) = (shape[0] as usize, shape[0] as usize * shape[2] as usize);
    let mut paths = vec![0u32; TreeIdx::COUNT * rows];
    if mode == 7 {
        for row in 0..rows {
            paths[TreeIdx::PathLogprobBits as usize * rows + row] = PATTERNS[row % PATTERNS.len()];
        }
    }
    let (expected_frontier, cpu_time) = cpu(refs, shape, scale, &paths);
    let expected = fields(&expected_frontier, capacity, scale);
    let residual = fixture.guarded(&data.0, bf16::from_bits(SENTINEL as u16));
    let logits = fixture.guarded(&data.1, f32::from_bits(SENTINEL));
    let ids = fixture.guarded(&data.2, SENTINEL);
    let seeds = fixture.guarded(&data.3, u64::from(SENTINEL));
    let metadata = fixture.guarded(&data.4, SENTINEL);
    let tree = fixture.guarded(&paths, SENTINEL);
    let valid = fixture.guarded(&vec![1u32; rows], SENTINEL);
    let frontier = fixture.guarded(&vec![SENTINEL; FrontierIdx::COUNT * capacity], SENTINEL);
    let kernel = WeaverTopChildrenVulkanKernel::new(&fixture.context, scale.is_some()).expect("WeaverTopChildren");
    let mut encode = |encoding: &mut _| {
        // SAFETY: complete CPU-shaped distinct buffers, in-range depth seeds, guarded nonzero byte offsets.
        unsafe {
            kernel.encode(
                arg(&residual),
                arg(&logits),
                arg(&ids),
                arg(&seeds),
                arg(&metadata),
                arg(&tree),
                arg(&valid),
                arg(&frontier),
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                capacity as u32,
                shape[0],
                scale,
                encoding,
            );
        }
    };
    let times = if timed {
        let (gpu, wall) = fixture.median_times(&mut encode);
        let mut cpu_times = vec![cpu_time];
        for _ in 1..13 {
            let (again, time) = cpu(refs, shape, scale, &paths);
            assert_eq!(again, expected_frontier);
            cpu_times.push(time);
        }
        cpu_times.drain(..3);
        cpu_times.sort();
        Some((gpu, wall, cpu_times[cpu_times.len() / 2]))
    } else {
        let mut encoding = fixture.encoding();
        encode(&mut encoding);
        KernelFixture::complete(encoding);
        None
    };
    // SAFETY: all submissions touching these buffers completed. Check every input and the frontier guard.
    let actual_frontier = unsafe {
        KernelFixture::assert_unchanged(&residual, bf16::from_bits(SENTINEL as u16), &data.0, "residual logits");
        KernelFixture::assert_unchanged(&logits, f32::from_bits(SENTINEL), &data.1, "candidate logits");
        KernelFixture::assert_unchanged(&ids, SENTINEL, &data.2, "candidate ids");
        KernelFixture::assert_unchanged(&seeds, u64::from(SENTINEL), &data.3, "depth seeds");
        KernelFixture::assert_unchanged(&metadata, SENTINEL, &data.4, "node metadata");
        KernelFixture::assert_unchanged(&tree, SENTINEL, &paths, "packed tree");
        KernelFixture::assert_unchanged(&valid, SENTINEL, &vec![1u32; rows], "node valid");
        KernelFixture::read_guarded::<u32>(&frontier, SENTINEL)
    };
    let actual = fields(&actual_frontier, capacity, scale);
    let [rows, candidates, width, _] = shape;
    if rows == 0 || candidates == 0 || candidates > 512 || width == 0 || width > candidates {
        assert_eq!(actual, expected, "guarded no work: {shape:?}");
        return times;
    }
    let combined = data.1.iter().zip(&data.0).map(|(&a, &b)| a + b.to_f32()).collect::<Vec<_>>();
    let scores = gpu_scores(fixture, &data.0, &data.1, &data.2, &data.3, &data.4, shape);
    let mut all_shape = shape;
    all_shape[2] = candidates;
    // Model and prune logprobs do not depend on selection noise: the CPU kernel's over every candidate, by token.
    let all_cpu = fields(&cpu(refs, all_shape, scale, &paths).0, shape[0] as usize * candidates as usize, scale);
    // A row's candidates best first in Metal's priority: the score's sortable key (total_cmp's order), then the lower
    // token, then the lower index.
    let order = |base: usize, score: &dyn Fn(usize) -> f32| {
        let mut order = (0..candidates as usize).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            score(b).total_cmp(&score(a)).then_with(|| data.2[base + a].cmp(&data.2[base + b])).then_with(|| a.cmp(&b))
        });
        order
    };
    let mut metal_tokens = Vec::new();
    let mut differing_rows = 0;
    for row in 0..rows as usize {
        let base = row * candidates as usize;
        let destination = row * width as usize..(row + 1) * width as usize;
        let seed = data.3[data.4[MetadataIdx::Depth as usize * rows as usize + row] as usize];
        // Metal's scores on the host: the combined logit plus the full-word selection noise.
        let metal_scores = (0..candidates as usize)
            .map(|i| combined[base + i] + gumbel_float_full_word(seed, revidx(data.2[base + i], shape[3])))
            .collect::<Vec<_>>();
        let metal_order = order(base, &|i| metal_scores[i]);
        let chosen = metal_order.iter().take(width as usize).map(|&i| data.2[base + i]).collect::<Vec<_>>();
        // Noise differs by at most 1e-5. Each independently rounded addition contributes at most one adjacent ULP.
        // Disjoint score intervals require Metal's choices, including the boundary just beyond the requested width.
        let separated = metal_scores.iter().all(|v| v.is_finite())
            && (0..width.min(candidates - 1) as usize).all(|rank| {
                let (a, b) = (metal_scores[metal_order[rank]], metal_scores[metal_order[rank + 1]]);
                let step = |value: f32| value.abs().next_up() - value.abs();
                a - b > 2e-5 + step(a) + step(b)
            });
        if separated {
            assert_eq!(actual[0][destination.clone()], chosen, "separated Metal scores");
        }
        differing_rows += usize::from(actual[0][destination] != chosen);
        metal_tokens.extend(chosen);
        for (rank, &index) in order(base, &|i| f32::from_bits(scores[base + i])).iter().take(width as usize).enumerate()
        {
            let output = row * width as usize + rank;
            let token = data.2[base + index];
            assert_eq!(
                actual[0][output], token,
                "computed GPU score order: {shape:?}, mode {mode}, row {row}, rank {rank}"
            );
            let cpu_index = (0..candidates as usize).find(|&i| all_cpu[0][base + i] == token).unwrap();
            compare_logprob(actual[1][output], all_cpu[1][base + cpu_index], 1e-5);
            if scale.is_some() {
                compare_logprob(actual[2][output], all_cpu[2][base + cpu_index], 1e-3);
            }
            if candidates == 1 && combined[base].is_finite() {
                assert_eq!(actual[1][output], expected[1][output], "singleton model probability");
            }
        }
    }
    if mode == 1 || mode == 4 || (mode == 0 && shape == [3, 512, 8, 131_072]) {
        assert_eq!(
            actual[0], metal_tokens,
            "Metal's choices for exact duplicate noise, the full-word endpoint, and the shared reference corpus"
        );
    }
    if mode == 7 {
        // Path sums over exceptional parents: a finite or infinite sum has exactly the CPU kernel's bits, a NaN only
        // needs to be a NaN (payloads are unspecified across architectures), and each priority is the sortable key of
        // the stored bits; every other field matches exactly.
        let field = |frontier: &[u32], field: usize| frontier[field * capacity..][..capacity].to_vec();
        let (path, key) = (FrontierIdx::PathLogprobBits as usize, FrontierIdx::PathScoreKey as usize);
        for index in (0..FrontierIdx::COUNT).filter(|&index| index != path && index != key) {
            assert_eq!(field(&actual_frontier, index), field(&expected_frontier, index), "exceptional parents {index}");
        }
        let (paths, keys) = (field(&actual_frontier, path), field(&actual_frontier, key));
        for (slot, (&actual, &expected)) in paths.iter().zip(&field(&expected_frontier, path)).enumerate() {
            let nan = |bits: u32| f32::from_bits(bits).is_nan();
            assert!(
                nan(expected) && nan(actual) || actual == expected,
                "path sum {slot}: {actual:#x} for {expected:#x}"
            );
            let sortable = if actual >> 31 == 0 {
                actual ^ 1 << 31
            } else {
                !actual
            };
            assert_eq!(keys[slot], sortable, "priority of path sum {slot}");
        }
    }
    if mode == 5 {
        eprintln!(
            "WeaverTopChildren {rows} constructed tie rows, prune={}: {differing_rows} GPU choices differ from the host",
            scale.is_some()
        );
    }
    times
}

#[uzu_test]
fn matches_cpu_and_computed_score_order() {
    let fixture = KernelFixture::new();
    for [rows, candidates, width] in [
        [1, 1, 1],
        [5, 3, 3],
        [9, 4, 1],
        [3, 31, 7],
        [2, 32, 32],
        [3, 33, 8],
        [3, 33, 33],
        [1, 127, 127],
        [1, 255, 1],
        [1, 256, 16],
        [4, 257, 32],
        [2, 257, 257],
        [2, 511, 8],
        [2, 511, 511],
        [3, 512, 8],
        [3, 512, 31],
        [3, 512, 33],
        [3, 512, 64],
        [3, 512, 128],
        [1, 512, 512],
    ] {
        for mode in 0..3 {
            for scale in [None, Some(1.0 / 1.5)] {
                check(&fixture, [rows, candidates, width, 131_072], mode, scale, false);
            }
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn ties_endpoints_and_tiny_logits() {
    let fixture = KernelFixture::new();
    for candidates in [1, 2, 33, 257, 512] {
        for scale in [None, Some(1.0 / 1.5)] {
            check(&fixture, [3, candidates, 1, 131_072], 3, scale, false);
        }
    }
    for scale in [None, Some(1.0 / 1.5)] {
        check(&fixture, [3, 512, 8, 131_072], 4, scale, false);
        check(&fixture, [256, 2, 2, 131_072], 5, scale, false);
        check(&fixture, [3, 8, 8, 131_072], 6, scale, false);
        check(&fixture, [20, 1, 1, 131_072], 7, scale, false);
    }
    for scale in [0.0, -0.0, -0.5, f32::MAX, f32::INFINITY, f32::NAN] {
        check(&fixture, [2, 8, 8, u32::MAX], 0, Some(scale), false);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn zero_work_and_invalid_shapes() {
    let fixture = KernelFixture::new();
    for shape in [[0, 1, 1, 0], [2, 0, 0, 1], [2, 0, 1, 1], [2, 1, 0, 1], [2, 1, 2, 1], [2, 513, 1, 1], [2, 2, 3, 1]] {
        for scale in [None, Some(1.0 / 1.5)] {
            check(&fixture, shape, 0, scale, false);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for shape in [[1, 512, 8, 131_072], [32, 512, 8, 131_072], [32, 512, 32, 131_072], [32, 512, 512, 131_072]] {
        for scale in [None, Some(1.0 / 1.5)] {
            let (gpu, wall, cpu) = check(&fixture, shape, 0, scale, true).expect("timed submissions");
            eprintln!(
                "WeaverTopChildren shape={shape:?} prune={}: GPU {gpu:?}, wall {wall:?}, CPU wall {cpu:?}",
                scale.is_some()
            );
        }
    }
    fixture.assert_clean();
}
