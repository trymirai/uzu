use std::time::Duration;

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{Backend, Kernels, kernel::radix_top_k_small::RadixTopKSmall},
        cpu::Cpu,
        vulkan::vk_kernels::RadixTopKSmallVulkanKernel,
    },
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

fn cpu(
    values: &[bf16],
    rows: u32,
    columns: u32,
    k: u32,
    submissions: usize,
) -> (Vec<u32>, Vec<f32>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::RadixTopKSmall::new(&context, columns).unwrap();
    let input = cpu_buffer(&context, values);
    let mut ids = cpu_buffer(&context, &vec![u32::MAX; rows as usize * k as usize]);
    let mut scores = cpu_buffer(&context, &vec![f32::NAN; rows as usize * k as usize]);
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(&input, &mut ids, &mut scores, rows, k, encoding).unwrap();
    });
    (
        buffer_prefix_to_vec::<Cpu, u32>(&ids, rows as usize * k as usize),
        buffer_prefix_to_vec::<Cpu, f32>(&scores, rows as usize * k as usize),
        times,
    )
}

fn check(
    fixture: &KernelFixture,
    values: &[bf16],
    rows: u32,
    columns: u32,
    k: u32,
) {
    let reference = cpu(values, rows, columns, k, 1);
    let mut expected_ids = Vec::new();
    let mut expected_scores = Vec::new();
    for row in values.chunks_exact(columns as usize) {
        let mut columns = (0..row.len()).collect::<Vec<_>>();
        columns.sort_by(|&a, &b| row[b].to_f32().total_cmp(&row[a].to_f32()).then(a.cmp(&b)));
        expected_ids.extend(columns[..k as usize].iter().map(|&i| i as u32));
        expected_scores.extend(columns[..k as usize].iter().map(|&i| row[i].to_f32()));
    }
    assert_eq!(expected_ids, reference.0, "host sort / CPU IDs");
    KernelFixture::assert_bits(&expected_scores, &reference.1, "host sort / CPU scores");
    let sentinel = bf16::from_f32(-31.0);
    let mut input = fixture.guarded(&[vec![sentinel], values.to_vec()].concat(), sentinel);
    input.1.start += 2;
    let ids = fixture.guarded(&vec![u32::MAX; rows as usize * k as usize], u32::MAX);
    let scores = fixture.guarded(&vec![f32::NAN; rows as usize * k as usize], f32::from_bits(0xa5a5_a5a5));
    let kernel = RadixTopKSmallVulkanKernel::new(&fixture.context, columns).unwrap();
    let mut encoding = fixture.encoding();
    // SAFETY: complete disjoint dense spans with 1<=k<=min(512,columns); BF16 input is aligned to two bytes.
    unsafe {
        kernel.encode(arg(&input), arg(&ids), arg(&scores), rows, k, &mut encoding);
    }
    KernelFixture::complete(encoding);
    // SAFETY: the submission completed before reading output, unchanged input and guards.
    unsafe {
        KernelFixture::assert_unchanged(&input, sentinel, values, "top-k input");
        assert_eq!(KernelFixture::read_guarded(&ids, u32::MAX), expected_ids, "rows={rows} columns={columns} k={k}");
        KernelFixture::assert_bits(
            &KernelFixture::read_guarded(&scores, f32::from_bits(0xa5a5_a5a5)),
            &expected_scores,
            "top-k scores",
        );
    }
}

#[uzu_test]
fn matches_cpu_and_full_sort() {
    let fixture = KernelFixture::new();
    for columns in [1u32, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 4097, 131073, 248320] {
        let rows = 3;
        let values = (0..rows as usize * columns as usize)
            .map(|i| {
                let value = (i as u32).wrapping_mul(747_796_405).wrapping_add(289_133_645);
                bf16::from_f32(((value ^ (value >> 16)) % 1024) as f32 / 32.0 - 16.0)
            })
            .collect::<Vec<_>>();
        for k in [1, 31.min(columns), 33.min(columns), 512.min(columns)] {
            check(&fixture, &values, rows, columns, k);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn all_bf16_bits_and_ties() {
    let fixture = KernelFixture::new();
    let all = (0..=u16::MAX).map(bf16::from_bits).collect::<Vec<_>>();
    for k in [1, 31, 33, 128, 512] {
        check(&fixture, &all, 1, 65536, k);
        check(&fixture, &all[32768..], 1, 32768, k);
    }
    for bits in [0u16, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fff, 0xffff] {
        let values = vec![bf16::from_bits(bits); 513];
        for k in [1, 33, 512] {
            check(&fixture, &values, 1, 513, k);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for columns in [65536u32, 131072, 248320] {
        for rows in [1u32, 8] {
            let values = (0..rows as usize * columns as usize)
                .map(|i| bf16::from_f32((i % 8192) as f32 / 512.0))
                .collect::<Vec<_>>();
            for k in [8, 64, 512] {
                check(&fixture, &values, rows, columns, k);
                let (_, _, mut times) = cpu(&values, rows, columns, k, 13);
                times.drain(..3);
                times.sort();
                let input = fixture.guarded(&values, bf16::from_f32(-31.0));
                let ids = fixture.guarded(&vec![u32::MAX; rows as usize * k as usize], u32::MAX);
                let scores = fixture.guarded(&vec![0.0f32; rows as usize * k as usize], -31.0f32);
                let kernel = RadixTopKSmallVulkanKernel::new(&fixture.context, columns).unwrap();
                // SAFETY: the identical dense spans passed the guarded CPU/full-sort comparison above.
                let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                    kernel.encode(arg(&input), arg(&ids), arg(&scores), rows, k, encoding);
                });
                // SAFETY: every timing submission completed before checking guards and the sorted IDs.
                let actual = unsafe { KernelFixture::read_guarded(&ids, u32::MAX) };
                assert_eq!(actual, cpu(&values, rows, columns, k, 1).0);
                unsafe {
                    KernelFixture::read_guarded(&scores, -31.0f32);
                }
                eprintln!(
                    "RadixTopKSmall BF16 C={columns} rows={rows} k={k}: GPU {gpu:?}, wall {wall:?}, CPU wall {:?}",
                    times[times.len() / 2]
                );
            }
        }
    }
    fixture.assert_clean();
}
