use std::{
    collections::BTreeMap,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{AncestorAttentionCase as Case, AttentionSinglePassCase, HEAD_DIM, kernel_fixture::KernelFixture};
use crate::backends::vulkan::{AncestorAttentionVulkanKernel, Error, VkBuffer, VkCommandBufferEncoding};

/// Successive frontiers on the CPU and Vulkan: every row's output against the AttentionSinglePass oracle of its
/// gathered keys, the CPU's within the bound and Vulkan's within it with the CPU's nonfinite classes, and the whole node
/// cache against the rotation oracle and the CPU, bit for bit except that the words of newly rotated keys, results of
/// FP32 arithmetic whose NaN payloads are not portable, match any NaN by any NaN. Copied values and untouched words keep
/// every bit. Returns the CPU outputs.
fn check_frontiers(
    fixture: &KernelFixture,
    label: &str,
    frontiers: &[Case],
    totals: &mut BTreeMap<String, [f64; 4]>,
) -> Vec<Vec<bf16>> {
    let kernel = frontiers[0].vulkan_kernel(fixture);
    let (cpu, cpu_cache) = Case::cpu(frontiers);
    let (vulkan, vulkan_cache) = Case::gpu(fixture, &kernel, frontiers);
    let model_dim = (frontiers[0].num_heads * HEAD_DIM) as usize;
    let mut expected_cache = frontiers[0].node_kv.clone();
    let mut rotated_keys = vec![false; expected_cache.len()];
    let assert_cache = |expected: &[bf16], actual: &[bf16], rotated_keys: &[bool], name: &str| {
        assert_eq!(expected.len(), actual.len(), "{label}: {name} length");
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let same =
                expected.to_bits() == actual.to_bits() || (rotated_keys[index] && expected.is_nan() && actual.is_nan());
            assert!(
                same,
                "{label}: {name} word {index}: expected {:#06x}, got {:#06x}",
                expected.to_bits(),
                actual.to_bits()
            );
        }
    };
    for (number, frontier) in frontiers.iter().enumerate() {
        assert_cache(&expected_cache, &frontier.node_kv, &rotated_keys, &format!("frontier {number} input"));
        let rotated = frontier.rotated();
        for row in 0..frontier.rows() as usize {
            let expected = frontier.single_pass(row, &rotated).oracle::<bf16>();
            let cpu_row = &cpu[number][row * model_dim..][..model_dim];
            let vulkan_row = &vulkan[number][row * model_dim..][..model_dim];
            let name = format!("{label} {} frontier {number} row {row}", frontier.label());
            for (backend, actual, reference) in [("CPU", cpu_row, None), ("Vulkan", vulkan_row, Some(cpu_row))] {
                let errors =
                    AttentionSinglePassCase::compare(&expected, actual, reference, &format!("{name} {backend}"));
                let total = totals.entry(format!("{label} {backend}")).or_default();
                *total = [total[0].max(errors[0]), total[1].max(errors[1]), total[2] + errors[2], total[3] + errors[3]];
            }
        }
        expected_cache = frontier.expected_cache(&rotated);
        if frontier.node_capacity > 0 {
            for &slot in &frontier.destinations {
                rotated_keys[slot as usize * model_dim..][..model_dim].fill(true);
            }
        }
    }
    assert_cache(&expected_cache, &cpu_cache, &rotated_keys, "CPU node cache");
    assert_cache(&cpu_cache, &vulkan_cache, &rotated_keys, "Vulkan node cache");
    cpu
}

/// Checks each case as one frontier; prints per-label maxima and fails on any violation.
fn check(
    group: &str,
    cases: &[Case],
) {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for case in cases {
        check_frontiers(&fixture, group, std::slice::from_ref(case), &mut totals);
    }
    report(&totals);
    fixture.assert_clean();
}

fn report(totals: &BTreeMap<String, [f64; 4]>) {
    for (label, [absolute, ratio, nans, violations]) in totals {
        eprintln!(
            "AncestorAttention {label}: max error {absolute:.3e}, max error/bound {ratio:.3e}, {nans} NaN, \
             {violations} violations"
        );
    }
    assert!(totals.values().all(|total| total[3] == 0.0), "AncestorAttention: elements exceed the bound");
}

/// 1, 3, 16 and 32 heads over 1 to 8 rows after prefixes of 0, 1, 5 and 130, with 0 to 4 listed ancestors of stride
/// 4 and depths across the tables.
#[uzu_test]
fn shapes() {
    let cases = itertools::iproduct!([1, 3, 16, 32], [(1, 0), (5, 1), (8, 5), (5, 130)])
        .map(|(heads, (rows, prefix))| Case::new(heads, rows, prefix, 4, heads + rows + prefix))
        .collect::<Vec<_>>();
    check("shapes", &cases);
}

/// Two successive frontiers in one command buffer over one node cache: the second lists slots the first writes,
/// repeated and out of order, beside earlier slots.
#[uzu_test]
fn successive_frontiers() {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for heads in [3, 16] {
        let whole = Case::new(heads, 7, 5, 3, heads);
        let model_dim = (heads * HEAD_DIM) as usize;
        let split = |rows: std::ops::Range<usize>| {
            let mut frontier = whole.clone();
            frontier.depths = whole.depths[rows.clone()].to_vec();
            frontier.ancestors = whole.ancestors[rows.clone()].to_vec();
            frontier.destinations = whole.destinations[rows.clone()].to_vec();
            frontier.current_qkv = whole.current_qkv[rows.start * 3 * model_dim..rows.end * 3 * model_dim].to_vec();
            frontier
        };
        let first = split(0..4);
        let mut second = split(4..7);
        // The first frontier writes slots 5 to 8.
        second.ancestors = vec![vec![8, 5, 1], vec![6, 6], vec![7, 2, 5]];
        second.node_kv = first.expected_cache(&first.rotated());
        check_frontiers(&fixture, "frontiers", &[first, second], &mut totals);
    }
    report(&totals);
    fixture.assert_clean();
}

/// Rows listing few of 16 entries: the unused ones name slot 0, whose keys and values are NaN, and are never read.
#[uzu_test]
fn unused_stride_entries() {
    let mut case = Case::new(4, 5, 3, 16, 2);
    for (row, listed) in case.ancestors.iter_mut().enumerate() {
        *listed = (1..=row as u32 % 3).map(|slot| slot * 4).collect();
    }
    check("unused entries", &[case]);
}

/// No prefix and no ancestors: the output is the row's value bit for bit, subnormal, signed zero or not (-0 summing to
/// +0 as on the CPU), and the row still enters its slot.
#[uzu_test]
fn current_only_is_value() {
    let mut case = Case::new(3, 4, 0, 2, 5);
    case.ancestors.iter_mut().for_each(Vec::clear);
    let model_dim = (3 * HEAD_DIM) as usize;
    let specials = [bf16::from_bits(1), bf16::from_bits(0x8003), bf16::from_bits(0x0060), bf16::NEG_ZERO, bf16::ZERO];
    for (row, element) in itertools::iproduct!(0..4, 0..model_dim) {
        if element % 3 == 0 {
            case.current_qkv[(row * 3 + 2) * model_dim + element] = specials[element % 5];
        }
    }
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    let cpu = check_frontiers(&fixture, "current only", std::slice::from_ref(&case), &mut totals);
    report(&totals);
    let (vulkan, _) = Case::gpu(&fixture, &case.vulkan_kernel(&fixture), std::slice::from_ref(&case));
    let values = (0..4)
        .flat_map(|row| &case.current_qkv[(row * 3 + 2) * model_dim..][..model_dim])
        .map(|value| bf16::from_f32(value.to_f32() + 0.0))
        .collect::<Vec<_>>();
    let bits = |values: &[bf16]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&cpu[0]), bits(&values), "CPU output is not the value");
    assert_eq!(bits(&vulkan[0]), bits(&values), "Vulkan output is not the value");
    fixture.assert_clean();
}

/// Without node slots the destinations are `u32::MAX` and the empty node cache sits between guards: nothing reads a
/// destination or writes the cache.
#[uzu_test]
fn zero_capacity() {
    let mut case = Case::new(4, 3, 5, 2, 3);
    case.ancestors.iter_mut().for_each(Vec::clear);
    (case.node_capacity, case.node_kv, case.padding, case.destinations) = (0, Vec::new(), u32::MAX, Vec::new());
    check("zero capacity", &[case]);
}

/// One head, one row at depth 0 over two prefix keys, scale 1024. Element `j` of the rotary row 1 has cosine and sine
/// `rotary(j)`, others 1 and 0; `edit` sets the row and the prefix.
fn witness(
    rotary: impl Fn(usize) -> (f32, f32),
    edit: impl Fn(&mut Case),
) -> Case {
    let mut case = Case::new(1, 1, 2, 1, 0);
    case.ancestors[0].clear();
    case.depths[0] = 0;
    case.scale = 1024.0;
    for j in 0..HEAD_DIM as usize {
        (case.cosines[HEAD_DIM as usize + j], case.sines[HEAD_DIM as usize + j]) = rotary(j);
    }
    case.current_qkv.fill(bf16::ZERO);
    case.prefix_kv.fill(bf16::ZERO);
    edit(&mut case);
    case
}

/// The CPU rounds each rotated element to BF16 before attending. The query witness rotates query element 0 to
/// 1 + 2^-10, which rounds to 1, so the two prefix keys score 1024 each and the output is 0.5; attending the unrounded
/// query scores the first key 1025, giving e / (1 + e). The key witness does the same through the row's own key against
/// one prefix key.
#[uzu_test]
fn bf16_rotation_staging() {
    let off = 1.0 + 2f32.powi(-10);
    let dim = HEAD_DIM as usize;
    let query = witness(
        |j| {
            (
                if j == 0 {
                    off
                } else {
                    1.0
                },
                0.0,
            )
        },
        |case| {
            (case.current_qkv[0], case.current_qkv[1]) = (bf16::ONE, bf16::ONE);
            // Prefix keys `[2, 128]` then values: key 0 has element 0, key 1 element 1; values 1 and 0.
            (case.prefix_kv[0], case.prefix_kv[dim + 1]) = (bf16::ONE, bf16::ONE);
            case.prefix_kv[2 * dim..3 * dim].fill(bf16::ONE);
        },
    );
    let key = witness(
        |j| {
            if j == 0 {
                (1.0, off)
            } else {
                (1.0, 0.0)
            }
        },
        |case| {
            case.prefix_length = 1;
            case.prefix_kv = vec![bf16::ZERO; 2 * dim];
            case.prefix_kv[0] = bf16::ONE;
            // Query element 0 is 1; the row's key element 64 is -1, rotating key element 0 to 1 + 2^-10; value 1.
            case.current_qkv[0] = bf16::ONE;
            case.current_qkv[dim + 64] = bf16::NEG_ONE;
            case.current_qkv[2 * dim..3 * dim].fill(bf16::ONE);
        },
    );
    for case in [&query, &key] {
        let (cpu, _) = Case::cpu(std::slice::from_ref(case));
        assert!(cpu[0].iter().all(|&value| value == bf16::from_f32(0.5)), "staging witness: CPU {:?}", &cpu[0][..4]);
    }
    check("staging", &[query, key]);
}

/// Rows and tables whose products and rotated elements are subnormal: the node cache keeps them bit for bit.
#[uzu_test]
fn subnormal_rotation() {
    let mut case = Case::new(4, 3, 5, 3, 9);
    case.current_qkv.iter_mut().for_each(|value| *value = bf16::from_f32(value.to_f32() * 2f32.powi(-120)));
    case.cosines.iter_mut().chain(&mut case.sines).for_each(|value| *value *= 2f32.powi(-8));
    check("subnormal", &[case]);
}

/// NaN and infinities in listed ancestors, the prefix and the row compared with the CPU's classes; quiet and signaling
/// NaNs of either sign in the row's values enter its slot with every bit.
#[uzu_test]
fn nonfinite_inputs() {
    let base = || {
        let mut case = Case::new(2, 2, 3, 2, 1);
        case.ancestors = vec![vec![2, 1], vec![3]];
        case
    };
    let model_dim = (2 * HEAD_DIM) as usize;
    let mut cases = Vec::new();
    // Element 5 of the value of slot 2, element 0 of the key of slot 1, element 5 of prefix key 1, element 0 of prefix
    // key 0, element 7 of row 1's query of head 1 and element 9 of row 0's value.
    let mut ancestor_value = base();
    ancestor_value.node_kv[(ancestor_value.node_capacity as usize + 2) * model_dim + 5] = bf16::NAN;
    let mut ancestor_key = base();
    ancestor_key.node_kv[model_dim] = bf16::NEG_INFINITY;
    let mut prefix_nan = base();
    prefix_nan.prefix_kv[model_dim + 5] = bf16::NAN;
    for infinity in [bf16::INFINITY, bf16::NEG_INFINITY] {
        let mut prefix_infinite = base();
        prefix_infinite.prefix_kv[0] = infinity;
        cases.push(prefix_infinite);
    }
    let mut query_nan = base();
    query_nan.current_qkv[3 * model_dim + HEAD_DIM as usize + 7] = bf16::NAN;
    let mut value_infinite = base();
    value_infinite.current_qkv[2 * model_dim + 9] = bf16::INFINITY;
    let mut value_payloads = base();
    for (offset, bits) in [0x7f81, 0xff81, 0x7fc5, 0xffe0].into_iter().enumerate() {
        value_payloads.current_qkv[2 * model_dim + 3 + offset] = bf16::from_bits(bits);
        value_payloads.current_qkv[5 * model_dim + 130 + offset] = bf16::from_bits(bits);
    }
    cases.extend([ancestor_value, ancestor_key, prefix_nan, query_nan, value_infinite, value_payloads]);
    check("nonfinite", &cases);
}

/// A prefix of 4095 keys and 8 ancestors per row crossing 65 tiles, scale 4.
#[uzu_test]
fn long_prefix() {
    let mut case = Case::new(16, 2, 4095, 8, 4);
    case.ancestors = vec![vec![9, 1, 4, 4, 7, 2, 8, 3], vec![5, 6, 1, 9, 2, 2, 3, 7]];
    case.scale = 4.0;
    check("long", &[case]);
}

/// Construction rejects every head dimension but 128.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for head_dim in [64, 256] {
        let kernel = AncestorAttentionVulkanKernel::new(&fixture.context, head_dim, 4);
        assert!(matches!(kernel, Err(Error::KernelPrecondition { .. })), "head dimension {head_dim} accepted");
    }
    fixture.assert_clean();
}

/// No row or no head records nothing: the guarded output is empty and the node cache unchanged.
#[uzu_test]
fn zero_groups_record_nothing() {
    let fixture = KernelFixture::new();
    for case in [Case::new(4, 0, 5, 2, 1), Case::new(0, 3, 5, 2, 1)] {
        let (outputs, cache) = Case::gpu(&fixture, &case.vulkan_kernel(&fixture), std::slice::from_ref(&case));
        assert!(outputs[0].is_empty(), "{}: output", case.label());
        let bits = |values: &[bf16]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&case.node_kv), bits(&cache), "{}: node cache", case.label());
    }
    fixture.assert_clean();
}

/// Run alone: `cargo test ... ancestor_attention_test::throughput -- --ignored --nocapture`. Weaver frontiers of 8 rows
/// and 16 heads after prefixes of 16, 1024 and 4096 with 8 ancestors each; the frontier of the Metal
/// benchmark_ancestor_attention (`Runner::new(8, 16, 8, 65)`: 8 rows, 16 heads, prefix 16, stride 8, 65 slots, row r
/// listing r % 9 slots (8 r + offset) % 57, destinations 57 to 64; hashed data and tables here); single rows of 32 heads
/// after 1024 and 16384 with 8 ancestors. Each frontier is checked against the CPU and the oracle, then timed; then
/// AttentionSinglePass over each row's own gathered keys, its case materialized on the host outside timing in the
/// model's layout, one dispatch of `heads` workgroups per row in one command buffer, checked against its oracle, then
/// timed. AncestorAttention reads one copy of the prefix and of each slot for every row; the materialized rows hold a
/// copy each, so memory reuse differs and the ratio is not an indirection overhead alone. "Unique" counts the bytes of
/// distinct K, V and row storage read, "requested" every key and value each row and head reads plus its query. GPU and
/// wall are medians of 10 after 3 warm-up submissions, encode the median of all 13 recordings.
#[uzu_test]
#[ignore]
fn throughput() {
    fn timed(
        fixture: &KernelFixture,
        mut encode: impl FnMut(&mut VkCommandBufferEncoding),
    ) -> [Duration; 3] {
        let mut encodes = Vec::new();
        let (gpu, wall) = fixture.median_times(|encoding| {
            let start = Instant::now();
            encode(encoding);
            encodes.push(start.elapsed());
        });
        encodes.sort();
        [gpu, encodes[encodes.len() / 2], wall]
    }
    fn whole(buffer: &Arc<VkBuffer>) -> (&Arc<VkBuffer>, Range<u64>) {
        (buffer, 0..buffer.size())
    }
    let fixture = KernelFixture::new();
    let mut shapes = Vec::new();
    for prefix in [16, 1024, 4096] {
        let mut case = Case::new(16, 8, prefix, 8, 1);
        case.ancestors = (0..8).map(|row| (0..8).map(|offset| 1 + (row * 8 + offset) % 9).collect()).collect();
        shapes.push(("Weaver", case));
    }
    // Stride 55 makes 65 slots with destinations 57 to 64; seed 0 gives depths 3 r % 8.
    let mut metal = Case::new(16, 8, 16, 55, 0);
    metal.ancestor_stride = 8;
    metal.ancestors = (0..8).map(|row| (0..row % 9).map(|offset| (row * 8 + offset) % 57).collect()).collect();
    // Row 7 lists slot 0, which holds finite values here.
    let model_dim = (16 * HEAD_DIM) as usize;
    for plane in [0, 65 * model_dim] {
        metal.node_kv.copy_within(plane + model_dim..plane + 2 * model_dim, plane);
    }
    shapes.push(("Metal benchmark", metal));
    for prefix in [1024, 16384] {
        let mut case = Case::new(32, 1, prefix, 8, 1);
        case.ancestors = vec![(1..9).collect()];
        shapes.push(("decode", case));
    }
    let mut totals = BTreeMap::new();
    for (name, case) in shapes {
        let label = format!("{name} {}", case.label());
        check_frontiers(&fixture, &label, std::slice::from_ref(&case), &mut totals);
        let (model_dim, rows) = ((case.num_heads * HEAD_DIM) as u64, case.rows() as usize);
        let kernel = case.vulkan_kernel(&fixture);
        let [indices, counts, destinations] = case.indices();
        let words = [case.metadata(), indices, counts, destinations].map(|words| fixture.buffer(&words));
        let caches = [&case.prefix_kv, &case.node_kv, &case.current_qkv].map(|values| fixture.buffer(values));
        let tables = [&case.cosines, &case.sines].map(|values| fixture.buffer(values));
        let output = fixture.buffer(&vec![bf16::ZERO; rows * model_dim as usize]);
        let ancestor = timed(&fixture, |encoding| {
            let ([prefix_kv, node_kv, current_qkv], [cosines, sines]) = (&caches, &tables);
            let [metadata, indices, counts, destinations] = &words;
            let buffers =
                [prefix_kv, node_kv, current_qkv, cosines, sines, metadata, indices, counts, destinations, &output]
                    .map(whole);
            // SAFETY: whole buffers of the checked frontier, which meets the caller preconditions.
            unsafe { case.encode(&kernel, buffers, encoding) };
        });
        let rotated = case.rotated();
        let singles = (0..rows).map(|row| case.single_pass(row, &rotated)).collect::<Vec<_>>();
        let single_kernel = singles[0].vulkan_kernel::<bf16>(&fixture);
        let materialized = singles
            .iter()
            .map(|single| {
                let errors = AttentionSinglePassCase::compare(
                    &single.oracle::<bf16>(),
                    &single.gpu::<bf16>(&fixture, &single_kernel),
                    None,
                    &format!("{label} AttentionSinglePass"),
                );
                let total = totals.entry(format!("{name} AttentionSinglePass Vulkan")).or_default();
                *total = [total[0].max(errors[0]), total[1].max(errors[1]), total[2] + errors[2], total[3] + errors[3]];
                let [queries, keys, values] = [&single.queries, &single.keys, &single.values]
                    .map(|values| fixture.buffer(&AttentionSinglePassCase::stored::<bf16>(values)));
                [queries, keys, values, fixture.buffer(&vec![bf16::ZERO; model_dim as usize])]
            })
            .collect::<Vec<_>>();
        let single = timed(&fixture, |encoding| {
            for (single, buffers) in singles.iter().zip(&materialized) {
                // SAFETY: whole buffers of the checked case's heads and query; the output aliases nothing.
                unsafe { single.encode(&single_kernel, buffers.each_ref().map(whole), None, None, encoding) };
            }
        });
        let row_bytes = 2 * model_dim * 2;
        let keys = case.ancestors.iter().map(|listed| u64::from(case.prefix_length) + listed.len() as u64 + 1);
        let requested = keys.sum::<u64>() * row_bytes + rows as u64 * model_dim * 2;
        let slots = case.ancestors.iter().flatten().collect::<std::collections::BTreeSet<_>>().len() as u64;
        let unique = (u64::from(case.prefix_length) + slots) * row_bytes + rows as u64 * 3 * model_dim * 2;
        for (kernel, [gpu, encode, wall], unique) in
            [("AncestorAttention", ancestor, unique), ("AttentionSinglePass materialized", single, requested)]
        {
            eprintln!(
                "MEASURE {name} {kernel} {}: unique {unique} B ({:.1} GB/s), requested {requested} B ({:.1} GB/s); GPU \
                 {gpu:?}, encode {encode:?}, wall {wall:?}",
                case.label(),
                unique as f64 / gpu.as_secs_f64() / 1e9,
                requested as f64 / gpu.as_secs_f64() / 1e9
            );
        }
        eprintln!(
            "MEASURE {name}: AncestorAttention / materialized AttentionSinglePass GPU {:.3}",
            ancestor[0].as_secs_f64() / single[0].as_secs_f64()
        );
    }
    report(&totals);
    fixture.assert_clean();
}
