use std::{sync::Arc, time::Instant};

use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, kernel_fixture::KernelFixture, panics};
use crate::{
    backends::{
        common::{Backend, Context, Kernels, kernel::ContextRingUpdateKernel},
        cpu::Cpu,
        vulkan::{
            ContextRingUpdateVulkanKernel, VkCommandBufferEncoding, vk_kernels::TestContextRingIndexVulkanKernel,
        },
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

const SENTINEL: u32 = 0xa5a5_a5a5;
const MAX: u32 = u32::MAX;

/// W1-W4, C1: (offset, length, capacity, count, step) to (first, slot, next offset, next length).
const WITNESSES: [([u32; 5], [u32; 4]); 5] = [
    ([MAX - 1, MAX - 1, MAX, MAX, MAX - 1], [MAX - 2, MAX - 3, MAX - 2, MAX]),
    ([3, 0, 7, 7, 4], [3, 0, 3, 7]),
    ([2, 5, 7, 20, 6], [6, 5, 6, 7]),
    ([MAX - 5, 2, MAX, 4, 3], [MAX - 3, 0, MAX - 5, 6]),
    ([MAX - 1, 2, MAX, 3, 2], [1, 3, MAX - 1, 5]),
];

fn hash(index: u32) -> u32 {
    index.wrapping_mul(0x9e37_79b9).rotate_left(13).wrapping_mul(0x85eb_ca6b)
}

/// `length` hashed tokens, every 97th 0 and the next u32::MAX.
fn tokens(
    length: u32,
    seed: u32,
) -> Vec<u32> {
    (0..length)
        .map(|i| match i % 97 {
            0 => 0,
            1 => u32::MAX,
            _ => hash(i ^ seed),
        })
        .collect()
}

/// The header (offset, length) followed by `slots` hashed tokens.
fn ring(
    offset: u32,
    length: u32,
    slots: u32,
    seed: u32,
) -> Vec<u32> {
    [offset, length].into_iter().chain((0..slots).map(|i| hash(i ^ !seed))).collect()
}

/// Exact u64 closed form, independent of the CPU's sequential appends and the shader's overflow-avoiding branches: of
/// count appends, token k of the last min(count, capacity) lands in slot (offset + length + k) mod capacity. Returns
/// that first slot, the slot of `step` past it and the next header; appends past the capacity advance the offset.
fn index_oracle([offset, length, capacity, count, step]: [u32; 5]) -> [u32; 4] {
    let [offset, length, capacity, count, step] = [offset, length, capacity, count, step].map(u64::from);
    let first = (offset + length + count - count.min(capacity)) % capacity;
    let next = (offset + (length + count).saturating_sub(capacity)) % capacity;
    [first, (first + step) % capacity, next, (length + count).min(capacity)].map(|value| value as u32)
}

/// The ring (capacity > 0) after appending `input`: its last min(n, capacity) tokens from index_oracle's first slot on.
fn oracle(
    ring: &[u32],
    capacity: u32,
    input: &[u32],
) -> Vec<u32> {
    let mut result = ring.to_vec();
    let n = input.len() as u32;
    let [first, _, offset, length] = index_oracle([ring[0], ring[1], capacity, n, 0]);
    for (step, &token) in input[(n - n.min(capacity)) as usize..].iter().enumerate() {
        result[2 + ((u64::from(first) + step as u64) % u64::from(capacity)) as usize] = token;
    }
    [result[0], result[1]] = [offset, length];
    result
}

/// A valid productive case: capacities near 2^32, small or anywhere; lengths up to the capacity, every fifth ring full;
/// counts near 2^32, small or anywhere, in every pairing with the capacity class; a step below min(count, capacity).
fn hashed_case(index: u32) -> [u32; 5] {
    let h = |k: u32| hash(5 * index + k);
    let capacity = match index % 3 {
        0 => MAX - h(0) % 16,
        1 => h(0) % 300 + 1,
        _ => h(0).max(1),
    };
    let length = match index % 5 {
        0 => capacity,
        _ => (u64::from(h(1)) % (u64::from(capacity) + 1)) as u32,
    };
    let count = match index / 3 % 3 {
        0 => MAX - h(3) % 16,
        1 => h(3) % 600 + 1,
        _ => h(3).max(1),
    };
    [h(2) % capacity, length, capacity, count, h(4) % count.min(capacity)]
}

/// The canonical CPU kernel through its trait on a copy of `ring`; an empty input gets cpu_buffer's unread placeholder.
fn cpu(
    context: &<Cpu as Backend>::Context,
    ring: &[u32],
    capacity: u32,
    input: &[u32],
) -> Vec<u32> {
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::ContextRingUpdateKernel::new(context).expect("CPU ContextRingUpdate");
    let input_buffer = cpu_buffer(context, input);
    let mut ring_buffer = create_buffer_with_data::<Cpu, u32>(context, ring);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(&input_buffer, &mut ring_buffer, capacity, input.len() as u32, &mut command_buffer);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, u32>(&ring_buffer)
}

/// Each update on its own guarded ring and input in `encoding`; CPU and Vulkan match the oracle on every ring word,
/// every guard holds and every input is unchanged.
fn check(
    fixture: &KernelFixture,
    kernel: &ContextRingUpdateVulkanKernel,
    cases: &[(Vec<u32>, u32, Vec<u32>)],
    mut encoding: VkCommandBufferEncoding,
) {
    let buffers = cases
        .iter()
        .map(|(ring, _, input)| (fixture.guarded(input, SENTINEL), fixture.guarded(ring, SENTINEL)))
        .collect::<Vec<_>>();
    for ((_, capacity, input), (input_buffer, ring_buffer)) in cases.iter().zip(&buffers) {
        // SAFETY: each ring holds a valid header and `capacity` tokens, and no input aliases a ring.
        unsafe { kernel.encode(arg(input_buffer), arg(ring_buffer), *capacity, input.len() as u32, &mut encoding) };
    }
    KernelFixture::complete(encoding);
    let context = create_context::<Cpu>();
    for ((ring, capacity, input), (input_buffer, ring_buffer)) in cases.iter().zip(&buffers) {
        let case = format!("capacity {capacity} header {:?} tokens {}", &ring[..2], input.len());
        let expected = oracle(ring, *capacity, input);
        assert_eq!(cpu(&context, ring, *capacity, input), expected, "{case}: CPU");
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            KernelFixture::assert_unchanged(input_buffer, SENTINEL, input, "input");
            assert_eq!(KernelFixture::read_guarded(ring_buffer, SENTINEL), expected, "{case}: Vulkan");
        }
    }
}

/// Capacities around a 256-thread workgroup and beyond; empty, wrapped, full and one-short headers; counts around them.
#[uzu_test]
fn matches_cpu_and_oracle() {
    let fixture = KernelFixture::new();
    let kernel = ContextRingUpdateVulkanKernel::new(&fixture.context).expect("ContextRingUpdate");
    let mut cases = Vec::new();
    for capacity in [1u32, 2, 7, 256, 257, 1000, 70000] {
        let headers = [
            (0, 0),
            (capacity - 1, capacity / 2),
            (0, capacity),
            (capacity / 2, capacity),
            (capacity - 1, capacity - 1),
        ];
        let mut counts =
            vec![1, 2, capacity - 1, capacity, capacity + 1, 255, 256, 257, 511, 512, 513, 2 * capacity + 3];
        counts.sort();
        counts.dedup();
        for (header, (offset, length)) in headers.into_iter().enumerate() {
            for &count in &counts {
                let seed = capacity ^ (count << 8) ^ ((header as u32) << 24);
                cases.push((ring(offset, length, capacity, seed), capacity, tokens(count, seed)));
            }
        }
    }
    check(&fixture, &kernel, &cases, fixture.encoding());
    fixture.assert_clean();
}

/// One command buffer: A += 10; B += A's tokens from A's buffer (RAW); A += 5 (WAR, WAW); A += 0; B += 1000 (WAW).
#[uzu_test]
fn dependent_updates_share_one_command_buffer() {
    let fixture = KernelFixture::new();
    let kernel = ContextRingUpdateVulkanKernel::new(&fixture.context).expect("ContextRingUpdate");
    let (a, b) = (ring(2, 3, 7, 1), ring(150, 299, 300, 2));
    let (x, y, z) = (tokens(10, 3), tokens(5, 4), tokens(1000, 5));
    let buffers = [&a, &b, &x, &y, &z].map(|values| fixture.guarded(values, SENTINEL));
    let [a_buffer, b_buffer, x_buffer, y_buffer, z_buffer] = &buffers;
    let empty = fixture.guarded::<u32>(&[], SENTINEL);
    let mut encoding = fixture.encoding();
    // SAFETY: both rings hold valid headers and their tokens; no update's input aliases the ring it writes.
    unsafe {
        kernel.encode(arg(x_buffer), arg(a_buffer), 7, 10, &mut encoding);
        kernel.encode((&a_buffer.0, a_buffer.1.start + 8..a_buffer.1.end), arg(b_buffer), 300, 7, &mut encoding);
        kernel.encode(arg(y_buffer), arg(a_buffer), 7, 5, &mut encoding);
        kernel.encode(arg(&empty), arg(a_buffer), 7, 0, &mut encoding);
        kernel.encode(arg(z_buffer), arg(b_buffer), 300, 1000, &mut encoding);
    }
    KernelFixture::complete(encoding);
    let a1 = oracle(&a, 7, &x);
    let b1 = oracle(&b, 300, &a1[2..]);
    let (a2, b2) = (oracle(&a1, 7, &y), oracle(&b1, 300, &z));
    let context = create_context::<Cpu>();
    let cpu_a1 = cpu(&context, &a, 7, &x);
    let cpu_b1 = cpu(&context, &b, 300, &cpu_a1[2..]);
    assert_eq!(cpu(&context, &cpu_a1, 7, &y), a2, "CPU A");
    assert_eq!(cpu(&context, &cpu_b1, 300, &z), b2, "CPU B");
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        assert_eq!(KernelFixture::read_guarded(a_buffer, SENTINEL), a2, "Vulkan A");
        assert_eq!(KernelFixture::read_guarded(b_buffer, SENTINEL), b2, "Vulkan B");
        for (buffer, values) in [(x_buffer, &x), (y_buffer, &y), (z_buffer, &z), (&empty, &Vec::new())] {
            KernelFixture::assert_unchanged(buffer, SENTINEL, values, "input");
        }
    }
    fixture.assert_clean();
}

/// Without input nothing is recorded at any capacity (no buffer retained) and the CPU leaves its ring as it was; input
/// into capacity 0 fails before recording, then the same command buffer completes a valid update.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    let kernel = ContextRingUpdateVulkanKernel::new(&fixture.context).expect("ContextRingUpdate");
    let (values, empty) = (ring(3, 5, 7, 9), fixture.guarded::<u32>(&[], SENTINEL));
    let untouched = fixture.guarded(&values, SENTINEL);
    let retained = || (Arc::strong_count(&untouched.0), Arc::strong_count(&empty.0));
    let before = retained();
    let mut encoding = fixture.encoding();
    for capacity in [0, 1, 7, u32::MAX] {
        // SAFETY: without input nothing is indexed or recorded.
        unsafe { kernel.encode(arg(&empty), arg(&untouched), capacity, 0, &mut encoding) };
    }
    for count in [1, 256, u32::MAX] {
        // SAFETY: never dispatched: the precondition fails before recording.
        let message = panics(|| unsafe { kernel.encode(arg(&empty), arg(&untouched), 0, count, &mut encoding) });
        let expected = "ContextRingUpdate: precondition input_length == 0 || suffix_repetition_length != 0 violated";
        assert_eq!(message, expected, "{count}");
    }
    assert_eq!(retained(), before, "an empty or rejected update was recorded");
    let context = create_context::<Cpu>();
    assert_eq!(cpu(&context, &values[..2], 0, &[]), &values[..2], "CPU capacity 0 without input");
    assert_eq!(cpu(&context, &values, u32::MAX, &[]), values, "CPU capacity 2^32 - 1 without input");
    check(&fixture, &kernel, &[(ring(1, 6, 7, 11), 7, tokens(9, 12))], encoding);
    // SAFETY: the completed command buffer recorded only the valid update.
    unsafe { KernelFixture::assert_unchanged(&untouched, SENTINEL, &values, "rejected ring") };
    fixture.assert_clean();
}

/// The shared ring arithmetic on the GPU, over the named witnesses and 4096 hashed valid full-width cases, against the
/// exact oracle, which the named expectations check in turn. Pure values only: no computed slot forms an address.
#[uzu_test]
fn full_width_ring_arithmetic() {
    for (case, expected) in WITNESSES {
        assert_eq!(index_oracle(case), expected, "oracle {case:?}");
    }
    let fixture = KernelFixture::new();
    let kernel = TestContextRingIndexVulkanKernel::new(&fixture.context).expect("TestContextRingIndex");
    let cases = WITNESSES.iter().map(|(case, _)| *case).chain((0..4096).map(hashed_case)).collect::<Vec<_>>();
    let flat = cases.as_flattened();
    let input = fixture.guarded(flat, SENTINEL);
    let results = fixture.guarded(&vec![SENTINEL; 4 * cases.len()], SENTINEL);
    let mut encoding = fixture.encoding();
    // SAFETY: cases holds 5 words and results 4 words per case; they do not alias.
    unsafe { kernel.encode(arg(&input), arg(&results), cases.len() as u32, &mut encoding) };
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&input, SENTINEL, flat, "cases");
        let actual = KernelFixture::read_guarded::<u32>(&results, SENTINEL);
        for (case, actual) in cases.iter().zip(actual.as_chunks::<4>().0) {
            assert_eq!(*actual, index_oracle(*case), "case {case:?}");
        }
    }
    fixture.assert_clean();
}

/// C1 through the canonical CPU kernel: the growth sum offset + length that crossed 2^32 before its u64 fix, on a ring of
/// capacity 2^32 - 1 allocated only through slot 3. From header (2^32 - 2, 2) all three appends grow the ring (length 2,
/// 3 and 4 below the capacity), token i to slot (2^32 - 2 + 2 + i) mod (2^32 - 1) = 1 + i: slots 1 to 3, inside the
/// allocation, with slot 0 kept and the header (2^32 - 2, 5).
#[uzu_test]
fn cpu_full_width() {
    let (values, input) = (ring(MAX - 1, 2, 4, 13), tokens(3, 14));
    let expected = oracle(&values, MAX, &input);
    assert_eq!(expected, [&[MAX - 1, 5, values[2]][..], &input[..]].concat(), "oracle");
    assert_eq!(cpu(&create_context::<Cpu>(), &values, MAX, &input), expected, "CPU");
}

/// Run alone: `cargo test ... context_ring_update_test::throughput -- --ignored --nocapture`; test-only presence
/// flag UZU_RING_REVERSE runs the descending round of the 12 cases first: 24 rows. After a CPU, Vulkan and oracle
/// check, 3 warm-up and 10 timed submissions each update their own guarded input and ring, each pair checked after
/// completion before the next encode (wall includes it). Host encode: median of the 10 timed. Raw kernel timings.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let kernel = ContextRingUpdateVulkanKernel::new(&fixture.context).expect("ContextRingUpdate");
    let reverse = std::env::var_os("UZU_RING_REVERSE").is_some();
    let ascending =
        [64u32, 1024, 4096].into_iter().flat_map(|c| [1u32, 8, 64, 1024].map(|n| (c, n))).collect::<Vec<_>>();
    let mut rounds = [ascending.clone(), ascending.iter().rev().copied().collect()];
    if reverse {
        rounds.reverse();
    }
    for (round, cases) in rounds.iter().enumerate() {
        for &(capacity, count) in cases {
            let (values, input) = (ring(capacity - 3, capacity / 2, capacity, count), tokens(count, capacity));
            check(&fixture, &kernel, &[(values.clone(), capacity, input.clone())], fixture.encoding());
            let expected = oracle(&values, capacity, &input);
            let pairs = (0..13)
                .map(|_| (fixture.guarded(&input, SENTINEL), fixture.guarded(&values, SENTINEL)))
                .collect::<Vec<_>>();
            let verify = |index: usize| {
                // SAFETY: the submission updating this pair has completed and no recording uses it.
                unsafe {
                    KernelFixture::assert_unchanged(&pairs[index].0, SENTINEL, &input, "input");
                    assert_eq!(KernelFixture::read_guarded(&pairs[index].1, SENTINEL), expected, "submission {index}");
                }
            };
            let (mut submitted, mut encodes) = (0, Vec::new());
            let (gpu, wall) = fixture.median_times(|encoding| {
                if submitted > 0 {
                    verify(submitted - 1);
                }
                let start = Instant::now();
                // SAFETY: each ring holds a valid header and its tokens; its input does not alias it.
                unsafe { kernel.encode(arg(&pairs[submitted].0), arg(&pairs[submitted].1), capacity, count, encoding) };
                encodes.push(start.elapsed());
                submitted += 1;
            });
            assert_eq!(submitted, pairs.len(), "submissions");
            verify(submitted - 1);
            let mut timed = encodes.split_off(3);
            timed.sort();
            let encode = timed[timed.len() / 2];
            eprintln!(
                "MEASURE ContextRingUpdate reverse {reverse} round {round} capacity {capacity} tokens {count}: median of 10 after 3 warm-up: GPU {gpu:?}, host encode {encode:?}, wall {wall:?} (with the previous check)"
            );
        }
    }
    fixture.assert_clean();
}
