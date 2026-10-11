use std::{fmt::Debug, mem::size_of, time::Duration};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, kernel::UnifiedSamplingKernel},
        cpu::Cpu,
        vulkan::UnifiedSamplingVulkanKernel,
    },
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

fn cpu<T: ArrayElement + Float + Default>(
    logits: &[T],
    seeds: &[u64],
    mask: Option<&[u32]>,
    options: (bool, Option<f32>, Option<u32>, Option<f32>, Option<f32>),
    vocab: u32,
    submissions: usize,
) -> (Vec<u32>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let (stochastic, temperature, top_k, top_p, min_p) = options;
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::UnifiedSamplingKernel::new(
        &context,
        T::data_type(),
        stochastic,
        mask.is_some(),
        temperature.is_some(),
        top_k.is_some(),
        top_p.is_some(),
        min_p.is_some(),
    )
    .expect("CPU UnifiedSampling");
    let input = cpu_buffer(&context, logits);
    let seed_buffer = cpu_buffer(&context, seeds);
    let mask_buffer = mask.map(|values| cpu_buffer(&context, values));
    let mut output = cpu_buffer(&context, &vec![u32::MAX; seeds.len()]);
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(
            &input,
            &mut output,
            stochastic.then_some(&seed_buffer),
            mask_buffer.as_ref(),
            temperature,
            top_k,
            top_p,
            min_p,
            vocab,
            seeds.len() as u32,
            encoding,
        );
    });
    (buffer_prefix_to_vec::<Cpu, u32>(&output, seeds.len()), times)
}

fn check<T: ArrayElement + Float + Default + Debug>(
    fixture: &KernelFixture,
    logits: &[T],
    seeds: &[u64],
    mask: Option<&[u32]>,
    options: (bool, Option<f32>, Option<u32>, Option<f32>, Option<f32>),
    vocab: u32,
) -> Vec<u32> {
    let expected = cpu(logits, seeds, mask, options, vocab, 1).0;
    let sentinel = T::from(-31.0).unwrap();
    let mut input = fixture.guarded(&[vec![sentinel], logits.to_vec()].concat(), sentinel);
    input.1.start += size_of::<T>() as u64;
    let seed_buffer = fixture.guarded(seeds, u64::MAX);
    let mask_buffer = mask.map(|values| fixture.guarded(values, 0xa5a5_a5a5u32));
    let output = fixture.guarded(&vec![u32::MAX; seeds.len()], u32::MAX);
    let (stochastic, temperature, top_k, top_p, min_p) = options;
    let kernel = UnifiedSamplingVulkanKernel::new(
        &fixture.context,
        T::data_type(),
        stochastic,
        mask.is_some(),
        temperature.is_some(),
        top_k.is_some(),
        top_p.is_some(),
        min_p.is_some(),
    )
    .expect("Vulkan UnifiedSampling");
    let mut encoding = fixture.encoding();
    // SAFETY: disjoint guarded spans cover [rows,vocab], one seed per row and ceil(vocab/32) mask words per row.
    unsafe {
        kernel.encode(
            arg(&input),
            arg(&output),
            stochastic.then(|| arg(&seed_buffer)),
            mask_buffer.as_ref().map(arg),
            temperature,
            top_k,
            top_p,
            min_p,
            vocab,
            seeds.len() as u32,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the submission completed before reading inputs, output and all guards.
    let actual = unsafe {
        KernelFixture::assert_unchanged(&input, sentinel, logits, "sampling logits");
        KernelFixture::assert_unchanged(&seed_buffer, u64::MAX, seeds, "sampling seeds");
        if let (Some(buffer), Some(values)) = (&mask_buffer, mask) {
            KernelFixture::assert_unchanged(buffer, 0xa5a5_a5a5u32, values, "sampling mask");
        }
        KernelFixture::read_guarded(&output, u32::MAX)
    };
    assert_eq!(actual, expected, "{:?} V={vocab} options={options:?} mask={}", T::data_type(), mask.is_some());
    actual
}

fn modes<T: ArrayElement + Float + Default + Debug>(fixture: &KernelFixture) {
    let seeds = [0, 1, u64::MAX];
    for vocab in [1u32, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 123456, 248320] {
        let logits = (0..3usize * vocab as usize)
            .map(|index| {
                let row = index / vocab as usize;
                let token = index % vocab as usize;
                let bits = (token as u32).wrapping_mul(747_796_405).wrapping_add(row as u32 * 289_133_645);
                T::from(((bits ^ (bits >> 16)) % 4096) as f32 / 256.0 - 8.0).unwrap()
            })
            .collect::<Vec<_>>();
        let mask = (0..3 * vocab.div_ceil(32)).map(|i| 0xb6db_6db6u32.rotate_left(i % 32)).collect::<Vec<_>>();
        let flags = if vocab == 4097 {
            (0..64).collect::<Vec<_>>()
        } else {
            vec![0, 1, 3, 7, 63]
        };
        for flags in flags {
            check(
                fixture,
                &logits,
                &seeds,
                (flags & 2 != 0).then_some(mask.as_slice()),
                (
                    flags & 1 != 0,
                    (flags & 4 != 0).then_some(0.6),
                    (flags & 8 != 0).then_some(20.min(vocab)),
                    (flags & 16 != 0).then_some(0.95),
                    (flags & 32 != 0).then_some(0.05),
                ),
                vocab,
            );
        }
    }
}

#[uzu_test]
fn all_modes_and_vocabulary_boundaries() {
    let fixture = KernelFixture::new();
    modes::<f32>(&fixture);
    modes::<bf16>(&fixture);
    fixture.assert_clean();
}

#[uzu_test]
fn ties_classes_and_filter_boundaries() {
    let fixture = KernelFixture::new();
    let rows = [
        [0.0, -0.0, 0.0, -0.0],
        [f32::NAN, -1.0, 2.0, f32::NAN],
        [f32::NAN; 4],
        [f32::NEG_INFINITY; 4],
        [f32::INFINITY, f32::INFINITY, 0.0, f32::NEG_INFINITY],
        [f32::from_bits(1), -f32::from_bits(1), 0.0, -0.0],
        [0.0, -90.0, -100.0, -1000.0],
    ];
    for row in rows {
        for stochastic in [false, true] {
            for (temperature, top_k, top_p, min_p) in [
                (None, None, None, None),
                (None, Some(1), None, None),
                (None, Some(0), None, None),
                (None, None, Some(0.0), None),
                (None, None, Some(0.5), None),
                (None, None, Some(f32::NAN), None),
                (None, None, None, Some(0.0)),
                (None, None, None, Some(f32::from_bits(1))),
                (None, None, None, Some(-1.0)),
                (None, None, None, Some(f32::NAN)),
                (None, None, None, Some(2.0)),
                (Some(0.0), None, None, None),
                (Some(-1.0), None, None, None),
                (Some(f32::NAN), None, None, None),
            ] {
                let options = (stochastic, temperature, top_k, top_p, min_p);
                check(&fixture, &row, &[u64::MAX], None, options, 4);
                check(&fixture, &row.map(bf16::from_f32), &[u64::MAX], None, options, 4);
            }
        }
    }
    for mask in [[0u32], [2], [8], [15]] {
        assert_eq!(
            check::<f32>(&fixture, &[1.0, 4.0, 3.0, 2.0], &[0], Some(&mask), (false, None, None, None, None), 4)[0],
            match mask[0] {
                0 => 0,
                8 => 3,
                _ => 1,
            }
        );
    }
    assert_eq!(
        check::<f32>(&fixture, &[0.0, -0.0, 0.0, -0.0], &[0], None, (false, None, Some(2), Some(0.25), None), 4),
        [0]
    );
    fixture.assert_clean();
}

#[uzu_test]
fn batches_and_zero_work() {
    let fixture = KernelFixture::new();
    let rows = 257;
    check(&fixture, &vec![1.0f32; rows], &vec![0; rows], None, (false, None, None, None, None), 1);
    check(&fixture, &[] as &[f32], &[], None, (false, None, None, None, None), 1);
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for vocab in [65536u32, 128000, 202048, 248320] {
        for rows in [1, 8] {
            let seeds = vec![u64::MAX; rows];
            let logits = (0..vocab as usize * rows).map(|i| (i % 8192) as f32 / 512.0 - 8.0).collect::<Vec<_>>();
            for options in [
                (false, None, None, None, None),
                (true, None, None, None, None),
                (true, Some(0.6), Some(20), Some(0.95), None),
            ] {
                let expected = check(&fixture, &logits, &seeds, None, options, vocab);
                let (_, mut times) = cpu(&logits, &seeds, None, options, vocab, 13);
                times.drain(..3);
                times.sort();
                let input = fixture.guarded(&logits, -31.0f32);
                let seed_buffer = fixture.guarded(&seeds, u64::MAX);
                let output = fixture.guarded(&vec![u32::MAX; rows], u32::MAX);
                let (stochastic, temperature, top_k, top_p, min_p) = options;
                let kernel = UnifiedSamplingVulkanKernel::new(
                    &fixture.context,
                    crate::data_type::DataType::F32,
                    stochastic,
                    false,
                    temperature.is_some(),
                    top_k.is_some(),
                    top_p.is_some(),
                    false,
                )
                .unwrap();
                // SAFETY: the same dense spans and flags passed the guarded correctness submission above.
                let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                    kernel.encode(
                        arg(&input),
                        arg(&output),
                        stochastic.then(|| arg(&seed_buffer)),
                        None,
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        vocab,
                        rows as u32,
                        encoding,
                    );
                });
                // SAFETY: every timed submission completed; output guards remain checked.
                assert_eq!(unsafe { KernelFixture::read_guarded(&output, u32::MAX) }, expected);
                eprintln!(
                    "UnifiedSampling F32 V={vocab} rows={rows} options={options:?}: GPU {gpu:?}, wall {wall:?}, CPU wall {:?}",
                    times[times.len() / 2]
                );
            }
        }
    }
    fixture.assert_clean();
}
