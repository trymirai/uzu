use std::time::Duration;

use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::WeaverFrontierInsertChildrenKernel,
        },
        cpu::Cpu,
        vulkan::WeaverFrontierInsertChildrenVulkanKernel,
    },
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

const SENTINEL: u32 = 0xa5a5_a5a5;
const PATTERNS: [u32; 20] = [
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x33800000,
    0xb3800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321, 0x3f800001, 0x00800001,
];

fn cpu(
    data: &[Vec<u32>; 7],
    (capacity, tree_slots, rows, width): (u32, u32, u32, u32),
    submissions: usize,
) -> (Vec<u32>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::WeaverFrontierInsertChildrenKernel::new(&context)
        .expect("CPU WeaverFrontierInsertChildren");
    let reads = [&data[0], &data[1], &data[2], &data[3]].map(|words| cpu_buffer(&context, words));
    let floats = [&data[4], &data[5]].map(|words| cpu_buffer(&context, bytemuck::cast_slice::<u32, f32>(words)));
    let mut out = cpu_buffer(&context, &data[6]);
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(
            &reads[0], &reads[1], &reads[2], &reads[3], &floats[0], &floats[1], &mut out, capacity, tree_slots, rows,
            width, encoding,
        );
    });
    (buffer_prefix_to_vec::<Cpu, u32>(&out, data[6].len()), times)
}

fn case(
    (capacity, tree_slots, rows, width): (u32, u32, u32, u32),
    mode: u32,
) -> [Vec<u32>; 7] {
    let (f, t, n, e) = (capacity as usize, tree_slots as usize, rows as usize, width as usize);
    let mut data = [
        vec![SENTINEL; TreeIdx::COUNT * t],
        vec![SENTINEL; MetadataIdx::COUNT * n],
        vec![1; n],
        vec![0; n * e],
        vec![0; n * e],
        vec![0; n * e],
        vec![SENTINEL; FrontierIdx::COUNT * f],
    ];
    for parent in 0..t {
        data[0][TreeIdx::PathLogprobBits as usize * t + parent] = (parent as f32 * 0.25 - 4.0).to_bits();
        data[0][TreeIdx::Depth as usize * t + parent] = if parent == 0 {
            u32::MAX - 1
        } else {
            parent as u32 % 7
        };
    }
    for row in 0..n {
        data[1][MetadataIdx::TreeSlot as usize * n + row] = match mode {
            0 => row as u32,
            1 => (row % t.max(1)) as u32,
            _ => {
                if row % 3 == 0 {
                    u32::MAX
                } else {
                    (row % t.max(1)) as u32
                }
            },
        };
        data[2][row] = u32::from(mode != 2 || row % 4 != 3);
        for child in 0..e {
            let index = row * e + child;
            data[3][index] = index as u32 ^ 0x7fff_ffff;
            data[4][index] = (index as f32 * -0.125).to_bits();
            data[5][index] = PATTERNS[index % PATTERNS.len()];
        }
    }
    data
}

// Select the last matching valid row for each destination, independently of the forward column writer.
fn membership_oracle(
    data: &[Vec<u32>; 7],
    (capacity, tree_slots, rows, width): (u32, u32, u32, u32),
    actual: &[u32],
    numeric: bool,
) {
    let (f, t, n, e) = (capacity as usize, tree_slots as usize, rows as usize, width as usize);
    for slot in 0..f {
        let winner = (t != 0 && e != 0)
            .then(|| {
                let parent = slot / e;
                (parent < t)
                    .then(|| {
                        (0..n).rev().find(|&row| {
                            data[2][row] != 0 && data[1][MetadataIdx::TreeSlot as usize * n + row] as usize == parent
                        })
                    })
                    .flatten()
            })
            .flatten();
        if let Some(row) = winner {
            let (parent, source) = (slot / e, row * e + slot % e);
            for (field, expected) in [
                (FrontierIdx::TokenId, data[3][source]),
                (FrontierIdx::ParentSlot, parent as u32),
                (FrontierIdx::Depth, data[0][TreeIdx::Depth as usize * t + parent] + 1),
                (FrontierIdx::EdgeLogprobBits, data[5][source]),
                (FrontierIdx::Active, 1),
            ] {
                assert_eq!(actual[field as usize * f + slot], expected, "last valid row: slot {slot}");
            }
            if numeric {
                let sum = (f32::from_bits(data[0][TreeIdx::PathLogprobBits as usize * t + parent])
                    + f32::from_bits(data[4][source]))
                .to_bits();
                let key = if sum & 0x8000_0000 == 0 {
                    sum ^ 0x8000_0000
                } else {
                    !sum
                };
                assert_eq!(actual[FrontierIdx::PathLogprobBits as usize * f + slot], sum, "sum at {slot}");
                assert_eq!(actual[FrontierIdx::PathScoreKey as usize * f + slot], key, "key at {slot}");
            }
        } else {
            for field in 0..FrontierIdx::COUNT {
                assert_eq!(actual[field * f + slot], data[6][field * f + slot], "untouched slot {slot}");
            }
        }
    }
}

fn check(
    fixture: &KernelFixture,
    data: &[Vec<u32>; 7],
    shape: (u32, u32, u32, u32),
    numeric: bool,
    timed: bool,
) -> (Vec<u32>, Option<(Duration, Duration, Duration)>) {
    let (expected, mut cpu_times) = cpu(
        data,
        shape,
        if timed {
            13
        } else {
            1
        },
    );
    membership_oracle(data, shape, &expected, numeric);
    let buffers = data.each_ref().map(|words| fixture.guarded(words, SENTINEL));
    let kernel = WeaverFrontierInsertChildrenVulkanKernel::new(&fixture.context).expect("WeaverFrontierInsertChildren");
    // SAFETY: full declared shapes, aligned/disjoint buffers; parent depths never overflow CPU depth + 1.
    let mut encode = |encoding: &mut _| unsafe {
        kernel.encode(
            arg(&buffers[0]),
            arg(&buffers[1]),
            arg(&buffers[2]),
            arg(&buffers[3]),
            arg(&buffers[4]),
            arg(&buffers[5]),
            arg(&buffers[6]),
            shape.0,
            shape.1,
            shape.2,
            shape.3,
            encoding,
        );
    };
    let times = if timed {
        let (gpu, wall) = fixture.median_times(&mut encode);
        cpu_times.drain(..3);
        cpu_times.sort();
        Some((gpu, wall, cpu_times[cpu_times.len() / 2]))
    } else {
        let mut encoding = fixture.encoding();
        encode(&mut encoding);
        KernelFixture::complete(encoding);
        None
    };
    // SAFETY: every recorded submission completed before these reads.
    let actual = unsafe {
        for index in 0..6 {
            KernelFixture::assert_unchanged(&buffers[index], SENTINEL, &data[index], "insert input");
        }
        KernelFixture::read_guarded::<u32>(&buffers[6], SENTINEL)
    };
    assert_eq!(expected, actual, "CPU / Vulkan: {shape:?}; all fields and float payloads");
    membership_oracle(data, shape, &actual, numeric);
    (actual, times)
}

#[uzu_test]
fn fields_duplicate_order_and_zero_work() {
    let fixture = KernelFixture::new();
    for shape in [
        (1, 1, 1, 1),
        (17, 8, 33, 3),
        (128, 16, 32, 8),
        (4099, 16, 257, 257),
        (4103, 8, 17, 513),
        (5, 17, 5, 33),
        (31, 4, 17, 7),
        (0, 4, 5, 3),
        (17, 0, 5, 3),
        (17, 4, 0, 3),
        (17, 4, 5, 0),
    ] {
        for mode in 0..3 {
            check(&fixture, &case(shape, mode), shape, true, false);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn stored_score_bits_match_cpu() {
    let fixture = KernelFixture::new();
    let shape = (400, 400, 400, 1);
    let mut data = case(shape, 0);
    for row in 0..400 {
        data[0][TreeIdx::PathLogprobBits as usize * 400 + row] = PATTERNS[row / 20];
        data[4][row] = PATTERNS[row % 20];
    }
    let (actual, _) = check(&fixture, &data, shape, false, false);
    // Frozen stored words from the canonical CPU before explicit propagation, not NaN-class equality.
    for (row, sum, key) in [
        (14, 0x7fc12345, 0xffc12345),
        (15, 0xffc54321, 0x003abcde),
        (16, 0x7fc12345, 0xffc12345),
        (17, 0xffc54321, 0x003abcde),
        (253, 0x7fc00000, 0xffc00000),
        (272, 0x7fc00000, 0xffc00000),
        (295, 0xffc54321, 0x003abcde),
        (314, 0x7fc12345, 0xffc12345),
        (337, 0xffc54321, 0x003abcde),
        (356, 0x7fc12345, 0xffc12345),
    ] {
        assert_eq!(actual[FrontierIdx::PathLogprobBits as usize * 400 + row], sum, "stored sum {row}");
        assert_eq!(actual[FrontierIdx::PathScoreKey as usize * 400 + row], key, "stored key {row}");
    }
    fixture.assert_clean();
}

#[uzu_test]
fn dependent_frontier_writes() {
    let fixture = KernelFixture::new();
    let shape = (257, 17, 33, 7);
    let first = case(shape, 1);
    let mut second = case(shape, 2);
    second[6] = cpu(&first, shape, 1).0;
    let expected = cpu(&second, shape, 1).0;
    let first_buffers = first.each_ref().map(|words| fixture.guarded(words, SENTINEL));
    let second_buffers = second[..6].iter().map(|words| fixture.guarded(words, SENTINEL)).collect::<Vec<_>>();
    let kernel = WeaverFrontierInsertChildrenVulkanKernel::new(&fixture.context).expect("WeaverFrontierInsertChildren");
    let mut encoding = fixture.encoding();
    for inputs in [&first_buffers[..6], second_buffers.as_slice()] {
        // SAFETY: valid shapes, distinct inputs/output; the encoder orders both writes to the same frontier range.
        unsafe {
            kernel.encode(
                arg(&inputs[0]),
                arg(&inputs[1]),
                arg(&inputs[2]),
                arg(&inputs[3]),
                arg(&inputs[4]),
                arg(&inputs[5]),
                arg(&first_buffers[6]),
                shape.0,
                shape.1,
                shape.2,
                shape.3,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: both dispatches completed; every input and all output guards remain checked.
    unsafe {
        assert_eq!(KernelFixture::read_guarded::<u32>(&first_buffers[6], SENTINEL), expected, "dependent writes");
        for index in 0..6 {
            KernelFixture::assert_unchanged(&first_buffers[index], SENTINEL, &first[index], "first input");
            KernelFixture::assert_unchanged(&second_buffers[index], SENTINEL, &second[index], "second input");
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for shape in [(2048, 256, 1, 8), (2048, 256, 32, 8), (2048, 128, 32, 16), (4096, 3, 32, 16)] {
        for mode in 0..2 {
            let data = case(shape, mode);
            check(&fixture, &data, shape, true, false);
            let (_, times) = check(&fixture, &data, shape, true, true);
            let (gpu, wall, cpu) = times.expect("timed submissions");
            eprintln!(
                "WeaverInsert capacity={} tree={} rows={} width={} mode={mode}: GPU {gpu:?}, wall {wall:?}, CPU wall {cpu:?}",
                shape.0, shape.1, shape.2, shape.3
            );
        }
    }
    fixture.assert_clean();
}
