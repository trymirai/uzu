use std::{collections::BTreeSet, fmt::Debug, mem::size_of, ops::Range, sync::Arc, time::Duration};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, kernel::RepetitionPenaltyKernel},
        cpu::Cpu,
        vulkan::{RepetitionPenaltyVulkanKernel, VkBuffer},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

// Build the chronological history directly, independently of the shader's suffix decomposition and hash lists.
fn oracle<T: ArrayElement + Float>(
    original: &[T],
    prefill: &[T],
    ring: &[u32],
    tokens: &[u32],
    penalty: f32,
    (capacity, vocab, start, length): (u32, u32, u32, u32),
) -> Vec<T> {
    let mut out = prefill.to_vec();
    for sample in 0..u64::from(length) {
        let mut history = Vec::new();
        for index in 0..u64::from(ring[1]) {
            let slot = (u64::from(ring[0]) + index) % u64::from(capacity);
            history.push(ring[2 + slot as usize]);
        }
        history.extend_from_slice(&tokens[..(u64::from(start) + sample + 1) as usize]);
        let suffix = history.len().saturating_sub(capacity as usize);
        for token in history[suffix..].iter().copied().collect::<BTreeSet<_>>() {
            let index = (sample * u64::from(vocab) + u64::from(token)) as usize;
            let value = original[index].to_f32().unwrap();
            out[index] = T::from(if value > 0.0 {
                value / penalty
            } else {
                value * penalty
            })
            .unwrap();
        }
    }
    out
}

fn cpu<T: ArrayElement + Float + Default>(
    original: &[T],
    prefill: &[T],
    ring: &[u32],
    tokens: &[u32],
    penalty: f32,
    (capacity, vocab, start, length): (u32, u32, u32, u32),
    submissions: usize,
) -> (Vec<T>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel::new(&context, T::data_type())
        .expect("CPU RepetitionPenalty");
    let original = cpu_buffer(&context, original);
    let mut out = cpu_buffer(&context, prefill);
    let ring = cpu_buffer(&context, ring);
    let tokens = cpu_buffer(&context, tokens);
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(&original, &mut out, &ring, &tokens, penalty, capacity, vocab, start, length, encoding);
    });
    (buffer_prefix_to_vec::<Cpu, T>(&out, prefill.len()), times)
}

// The extra sentinel shifts BF16 payloads to address 2 mod 4, including odd vocabulary row strides.
fn shifted<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    values: &[T],
) -> (Arc<VkBuffer>, Range<u64>) {
    let sentinel = T::from(-31.0).unwrap();
    let mut guarded = fixture.guarded(&[vec![sentinel], values.to_vec()].concat(), sentinel);
    guarded.1.start += size_of::<T>() as u64;
    guarded
}

fn check<T: ArrayElement + Float + Default + Debug>(
    fixture: &KernelFixture,
    original: &[T],
    prefill: &[T],
    ring: &[u32],
    tokens: &[u32],
    penalty: f32,
    shape: (u32, u32, u32, u32),
) -> Vec<T> {
    let expected = oracle(original, prefill, ring, tokens, penalty, shape);
    let reference = cpu(original, prefill, ring, tokens, penalty, shape, 1).0;
    KernelFixture::assert_bits(&expected, &reference, "independent history / CPU");
    let input = shifted(fixture, original);
    let out = shifted(fixture, prefill);
    let ring_buffer = fixture.guarded(ring, u32::MAX);
    let token_buffer = fixture.guarded(tokens, u32::MAX);
    let kernel = RepetitionPenaltyVulkanKernel::new(&fixture.context, T::data_type()).expect("RepetitionPenalty");
    let (capacity, vocab, start, length) = shape;
    let mut encoding = fixture.encoding();
    // SAFETY: valid ring and token IDs, disjoint buffers, complete aligned input/output ranges for every row.
    unsafe {
        kernel.encode(
            arg(&input),
            arg(&out),
            arg(&ring_buffer),
            arg(&token_buffer),
            penalty,
            capacity,
            vocab,
            start,
            length,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: submission completed; all reads include guards.
    let actual = unsafe {
        KernelFixture::assert_unchanged(&input, T::from(-31.0).unwrap(), original, "original logits");
        KernelFixture::assert_unchanged(&ring_buffer, u32::MAX, ring, "context ring");
        KernelFixture::assert_unchanged(&token_buffer, u32::MAX, tokens, "token IDs");
        KernelFixture::read_guarded(&out, T::from(-31.0).unwrap())
    };
    KernelFixture::assert_bits(&reference, &actual, &format!("{:?} {shape:?} p={penalty:?}", T::data_type()));
    actual
}

fn histories<T: ArrayElement + Float + Default + Debug>(fixture: &KernelFixture) {
    let shapes = [
        (1, 7, 0, 1, 0),
        (2, 17, 3, 2, 1),
        (7, 33, 5, 17, 7),
        (255, 257, 0, 2, 254),
        (256, 257, 0, 2, 256),
        (257, 257, 3, 17, 257),
        (1024, 257, 0, 17, 1024),
        (1025, 257, 259, 2, 1025),
        (4099, 257, 0, 1, 4099),
        (7, 33, 5, 257, 7),
        (1024, 248320, 3, 1, 1024),
        (1024, 248320, 0, 17, 1024),
    ];
    for (capacity, vocab, start, length, old_length) in shapes {
        for mode in 0..3 {
            let offset = capacity - 1;
            let token = |index: u32| match mode {
                0 => index % vocab,
                1 => index % 3,
                _ => 1,
            };
            let ring = [vec![offset, old_length], (0..capacity).map(token).collect()].concat();
            let tokens = (0..start + length).map(|i| token(i + 1)).collect::<Vec<_>>();
            let original = (0..vocab as usize * length as usize)
                .map(|i| T::from((i % 37) as f32 * 0.25 - 4.5).unwrap())
                .collect::<Vec<_>>();
            let prefill = vec![T::from(19.0).unwrap(); original.len()];
            check(
                fixture,
                &original,
                &prefill,
                &ring,
                &tokens,
                if mode == 0 {
                    1.05
                } else {
                    1.1
                },
                (capacity, vocab, start, length),
            );
        }
    }
    // 256 distinct IDs in one hash bucket, then the same IDs again in another chunk.
    let collisions = (0u32..70001).filter(|id| id.wrapping_mul(0x9e3779b9) >> 24 == 0).take(256).collect::<Vec<_>>();
    assert_eq!(collisions.len(), 256);
    let ring = [vec![511, 512], collisions.repeat(2)].concat();
    let original = vec![T::from(2.0).unwrap(); 70001 * 2];
    check(
        fixture,
        &original,
        &vec![T::from(19.0).unwrap(); original.len()],
        &ring,
        &collisions[..2],
        1.5,
        (512, 70001, 0, 2),
    );
}

#[uzu_test]
fn matches_cpu_and_history() {
    let fixture = KernelFixture::new();
    histories::<f32>(&fixture);
    histories::<bf16>(&fixture);
    fixture.assert_clean();
}

#[uzu_test]
fn ieee_edges_and_literals() {
    let fixture = KernelFixture::new();
    let bits = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x40000000,
    ];
    let f32_values = bits.map(f32::from_bits);
    let bf16_values = [
        0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1,
        0xffc1, 0xbf81,
    ]
    .map(bf16::from_bits);
    let tokens = (0..17).collect::<Vec<_>>();
    for penalty in [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.05,
        1.1,
        1.5,
        -1.5,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        check(&fixture, &f32_values, &[19.0; 17], &[0, 0], &tokens, penalty, (17, 17, 16, 1));
        check(&fixture, &bf16_values, &[bf16::from_f32(19.0); 17], &[0, 0], &tokens, penalty, (17, 17, 16, 1));
    }
    let result =
        check::<f32>(&fixture, &[2.0, -2.0, 0.0, -0.0, 7.0], &[19.0; 5], &[0, 0], &[0, 1, 2, 3], 1.5, (4, 5, 3, 1));
    assert_eq!(
        result.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [0x3faaaaab, 0xc0400000, 0, 0x80000000, 0x41980000]
    );
    let result = check(
        &fixture,
        &[bf16::from_bits(0xbf81), bf16::from_f32(7.0)],
        &[bf16::from_f32(19.0); 2],
        &[0, 0],
        &[0],
        1.5,
        (1, 2, 0, 1),
    );
    assert_eq!(result[0].to_bits(), 0xbfc2);
    assert!(RepetitionPenaltyVulkanKernel::new(&fixture.context, DataType::F16).is_err());
    assert!(RepetitionPenaltyVulkanKernel::new(&fixture.context, DataType::U32).is_err());
    fixture.assert_clean();
}

fn dependent<T: ArrayElement + Float + Default + Debug>(fixture: &KernelFixture) {
    let initial = (0..21).map(|i| T::from(i as f32 - 10.0).unwrap()).collect::<Vec<_>>();
    let prefill = vec![T::from(19.0).unwrap(); 21];
    let ring = [0, 4, 0, 1, 2, 3];
    let tokens = [4, 5, 6];
    let shape = (4, 7, 0, 3);
    let mut expected = [initial.clone(), prefill.clone(), prefill.clone()];
    let buffers = [shifted(fixture, &initial), shifted(fixture, &prefill), shifted(fixture, &prefill)];
    let ring_buffer = fixture.guarded(&ring, u32::MAX);
    let token_buffer = fixture.guarded(&tokens, u32::MAX);
    let kernel = RepetitionPenaltyVulkanKernel::new(&fixture.context, T::data_type()).expect("RepetitionPenalty");
    let mut encoding = fixture.encoding();
    // Consecutive dispatches exercise RAW, WAR and WAW on ranges with ReadWrite output access.
    for (source, destination) in [(0, 1), (1, 2), (2, 1), (2, 0), (0, 2)] {
        let next = cpu(&expected[source], &expected[destination], &ring, &tokens, 1.5, shape, 1).0;
        KernelFixture::assert_bits(
            &oracle(&expected[source], &expected[destination], &ring, &tokens, 1.5, shape),
            &next,
            "dependent CPU / history",
        );
        expected[destination] = next;
        // SAFETY: each dispatch has distinct original/output buffers and valid guarded ranges.
        unsafe {
            kernel.encode(
                arg(&buffers[source]),
                arg(&buffers[destination]),
                arg(&ring_buffer),
                arg(&token_buffer),
                1.5,
                shape.0,
                shape.1,
                shape.2,
                shape.3,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the entire dependent command buffer completed.
    unsafe {
        for (buffer, expected) in buffers.iter().zip(&expected) {
            KernelFixture::assert_bits(
                expected,
                &KernelFixture::read_guarded(buffer, T::from(-31.0).unwrap()),
                "dependent Vulkan / CPU",
            );
        }
        KernelFixture::assert_unchanged(&ring_buffer, u32::MAX, &ring, "dependent ring");
        KernelFixture::assert_unchanged(&token_buffer, u32::MAX, &tokens, "dependent tokens");
    }
    check(fixture, &initial, &prefill, &[0, 0], &tokens, 1.5, (0, 7, 0, 3));
    check::<T>(fixture, &[], &[], &[0, 0], &[], 1.5, (7, 7, 0, 0));
}

#[uzu_test]
fn dependent_updates_and_zero_work() {
    let fixture = KernelFixture::new();
    dependent::<f32>(&fixture);
    dependent::<bf16>(&fixture);
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Default + Debug>(fixture: &KernelFixture) {
        for length in [1, 17, 257] {
            let (capacity, vocab) = (1024, 248320);
            let shape = (capacity, vocab, 0, length);
            let ring = [vec![1023, capacity], (0..capacity).map(|i| i % 257).collect()].concat();
            let tokens = (0..length).map(|i| (i + 17) % 257).collect::<Vec<_>>();
            let original = vec![T::from(2.0).unwrap(); vocab as usize * length as usize];
            let prefill = vec![T::from(19.0).unwrap(); original.len()];
            for penalty in [1.05, 1.1] {
                let expected = check(fixture, &original, &prefill, &ring, &tokens, penalty, shape);
                let (cpu_output, mut cpu_times) = cpu(&original, &prefill, &ring, &tokens, penalty, shape, 13);
                KernelFixture::assert_bits(&expected, &cpu_output, "timed CPU output");
                cpu_times.drain(..3);
                cpu_times.sort();
                let cpu_wall = cpu_times[cpu_times.len() / 2];
                let input = shifted(fixture, &original);
                let out = shifted(fixture, &prefill);
                let ring_buffer = fixture.guarded(&ring, u32::MAX);
                let token_buffer = fixture.guarded(&tokens, u32::MAX);
                let kernel =
                    RepetitionPenaltyVulkanKernel::new(&fixture.context, T::data_type()).expect("RepetitionPenalty");
                // SAFETY: validated above; every timed submission completes before the next.
                let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                    kernel.encode(
                        arg(&input),
                        arg(&out),
                        arg(&ring_buffer),
                        arg(&token_buffer),
                        penalty,
                        capacity,
                        vocab,
                        0,
                        length,
                        encoding,
                    );
                });
                // SAFETY: all timing submissions completed; inputs and output guards remain checked.
                unsafe {
                    KernelFixture::assert_bits(
                        &expected,
                        &KernelFixture::read_guarded(&out, T::from(-31.0).unwrap()),
                        "timed output",
                    );
                    KernelFixture::assert_unchanged(&input, T::from(-31.0).unwrap(), &original, "timed logits");
                    KernelFixture::assert_unchanged(&ring_buffer, u32::MAX, &ring, "timed ring");
                    KernelFixture::assert_unchanged(&token_buffer, u32::MAX, &tokens, "timed tokens");
                }
                eprintln!(
                    "RepetitionPenalty {:?} vocab={vocab} suffix={capacity} batch={length} penalty={penalty}: GPU {gpu:?}, wall {wall:?}, CPU wall {cpu_wall:?}",
                    T::data_type()
                );
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
