use std::{cmp::Reverse, time::Duration};

use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::weaver::{FRONTIER_MAX_SLOTS, FRONTIER_MAX_WIDTH, FrontierIdx, MetadataIdx, TreeIdx},
            kernel::WeaverFrontierSelectKernel,
        },
        cpu::Cpu,
        vulkan::WeaverFrontierSelectVulkanKernel,
    },
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

const SENTINEL: u32 = 0xa5a5_a5a5;
const PAYLOADS: [u32; 6] = [0, 0x80000000, 1, 0x7f812345, 0xffc54321, 0xff854321];

// Shape: frontier, tree, nodes, batch start, ancestor stride, max depth, lookahead, pool depth, pool width.
fn case(
    shape: [u32; 9],
    mode: u32,
) -> [Vec<u32>; 11] {
    let [f, t, n, b, a, _, look, c, w] = shape.map(|x| x as usize);
    let mut data = [
        vec![0; FrontierIdx::COUNT * f],
        vec![SENTINEL; TreeIdx::COUNT * t],
        (0..t * a).map(|i| (i as u32).wrapping_mul(17) ^ 0x1357_2468).collect(),
        vec![SENTINEL; n],
        vec![SENTINEL; MetadataIdx::COUNT * n],
        vec![SENTINEL; n * a],
        vec![SENTINEL; n],
        (0..c * w).map(|i| i as u32 ^ 0x1234_5678).collect(),
        (0..c * w).map(|i| PAYLOADS[i % PAYLOADS.len()]).collect(),
        vec![SENTINEL; n * w],
        vec![SENTINEL; n * w],
    ];
    for slot in 0..f {
        let (token, parent, depth, key, active) = match mode {
            1 => (19, 2, 3, 0xc000_0000 | (slot as u32 % 2), 1),
            2 => (u32::MAX, u32::MAX, look as u32, slot as u32 % 2, 1),
            3 => (
                100 + slot as u32,
                [b as u32 + 1, b as u32, b as u32 + 2, u32::MAX][slot % 4],
                [3, u32::MAX, 2, 5][slot % 4],
                0xffff_fff0u32.wrapping_sub(slot as u32 * 2),
                1,
            ),
            4 => (slot as u32, u32::MAX, 2, 0xffff_ffff, u32::from(slot == 1 || slot + 1 == f)),
            _ => (
                (slot as u32).wrapping_mul(37) % 101,
                (slot % t.max(1)) as u32,
                (slot % 7) as u32,
                (slot as u32).wrapping_mul(0x9e37_79b9),
                u32::from(slot % 3 != 2),
            ),
        };
        for (field, value) in [
            (FrontierIdx::TokenId, token),
            (FrontierIdx::ParentSlot, parent),
            (FrontierIdx::Depth, depth),
            (FrontierIdx::PathLogprobBits, PAYLOADS[slot % PAYLOADS.len()]),
            (FrontierIdx::EdgeLogprobBits, PAYLOADS[(slot + 1) % PAYLOADS.len()]),
            (FrontierIdx::PathScoreKey, key),
            (FrontierIdx::Active, active),
        ] {
            data[0][field as usize * f + slot] = value;
        }
    }
    data
}

fn cpu(
    data: &[Vec<u32>; 11],
    shape: [u32; 9],
    submissions: usize,
) -> ([Vec<u32>; 11], Vec<Duration>) {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::WeaverFrontierSelectKernel::new(&context)
        .expect("CPU WeaverFrontierSelect");
    let [mut frontier, mut tree, mut ancestors, mut tokens, mut metadata, mut node_ancestors, mut valid] =
        std::array::from_fn::<_, 7, _>(|i| cpu_buffer(&context, &data[i]));
    let pool_ids = cpu_buffer(&context, &data[7]);
    let pool_logits = cpu_buffer(&context, bytemuck::cast_slice::<u32, f32>(&data[8]));
    let mut candidate_ids = cpu_buffer(&context, &data[9]);
    let mut candidate_logits = cpu_buffer(&context, bytemuck::cast_slice::<u32, f32>(&data[10]));
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(
            &mut frontier,
            &mut tree,
            &mut ancestors,
            &mut tokens,
            &mut metadata,
            &mut node_ancestors,
            &mut valid,
            &pool_ids,
            &pool_logits,
            &mut candidate_ids,
            &mut candidate_logits,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
            shape[6],
            shape[7],
            shape[8],
            encoding,
        );
    });
    let buffers = [
        &frontier,
        &tree,
        &ancestors,
        &tokens,
        &metadata,
        &node_ancestors,
        &valid,
        &pool_ids,
        &pool_logits,
        &candidate_ids,
        &candidate_logits,
    ];
    (std::array::from_fn(|i| buffer_prefix_to_vec::<Cpu, u32>(buffers[i], data[i].len())), times)
}

// Sort all eligible entries once, rather than repeating the CPU's per-node maximum scan.
fn reference(
    data: &[Vec<u32>; 11],
    shape: [u32; 9],
) -> [Vec<u32>; 11] {
    let [f, t, n, b, a, max_depth, look, c, w] = shape.map(|x| x as usize);
    let mut out = data.clone();
    if f == 0
        || f > FRONTIER_MAX_SLOTS as usize
        || n == 0
        || n > FRONTIER_MAX_WIDTH as usize
        || a == 0
        || max_depth == 0
        || t == 0
        || b + n > t
        || c == 0
        || w == 0
    {
        return out;
    }
    let field = |field: FrontierIdx, slot: usize| data[0][field as usize * f + slot];
    let mut ranked = (0..f)
        .filter_map(|slot| {
            if field(FrontierIdx::Active, slot) == 0 {
                return None;
            }
            let key = (u32::from(field(FrontierIdx::Depth, slot) < look as u32) << 31)
                | (field(FrontierIdx::PathScoreKey, slot) >> 1);
            let parent = field(FrontierIdx::ParentSlot, slot);
            let token = field(FrontierIdx::TokenId, slot);
            (key != 0 || parent != u32::MAX || token != u32::MAX).then_some((Reverse(key), parent, token, slot))
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    for node in 0..n {
        let selected = ranked.get(node);
        let real = selected.is_some();
        let (token, parent, depth, path, edge) =
            selected.map_or((0, u32::MAX, 0, 0, 0), |&(_, parent, token, slot)| {
                out[0][FrontierIdx::Active as usize * f + slot] = 0;
                (
                    token,
                    parent,
                    field(FrontierIdx::Depth, slot),
                    field(FrontierIdx::PathLogprobBits, slot),
                    field(FrontierIdx::EdgeLogprobBits, slot),
                )
            });
        let tree_slot = b + node;
        for (field, value) in [
            (TreeIdx::TokenId, token),
            (TreeIdx::ParentSlot, parent),
            (TreeIdx::Depth, depth),
            (TreeIdx::PathLogprobBits, path),
            (TreeIdx::EdgeLogprobBits, edge),
            (TreeIdx::Valid, u32::from(real)),
        ] {
            out[1][field as usize * t + tree_slot] = value;
        }
        let parent = if real && (parent as usize) < t {
            parent as usize
        } else {
            0
        };
        let row = (0..a)
            .map(|column| {
                if !real || column + 1 > depth as usize {
                    0
                } else if column + 1 == depth as usize {
                    parent as u32
                } else {
                    out[2][parent * a + column]
                }
            })
            .collect::<Vec<_>>();
        out[2][tree_slot * a..(tree_slot + 1) * a].copy_from_slice(&row);
        out[5][node * a..(node + 1) * a].copy_from_slice(&row);
        out[3][node] = token;
        out[4][MetadataIdx::Depth as usize * n + node] = if depth < look as u32 {
            depth
        } else {
            0
        };
        out[4][MetadataIdx::AncestorCount as usize * n + node] = depth;
        out[4][MetadataIdx::TreeSlot as usize * n + node] = tree_slot as u32;
        out[6][node] = u32::from(real && depth < look as u32);
        if (depth as usize) < c {
            for (pool, output) in [(7, 9), (8, 10)] {
                out[output][node * w..(node + 1) * w]
                    .copy_from_slice(&data[pool][depth as usize * w..(depth as usize + 1) * w]);
            }
        }
    }
    out
}

fn check(
    fixture: &KernelFixture,
    data: &[Vec<u32>; 11],
    shape: [u32; 9],
    timed: bool,
) -> Option<(Duration, Duration, Duration)> {
    let expected = reference(data, shape);
    let runs = if timed {
        13
    } else {
        1
    };
    let mut cpu_times = Vec::new();
    for _ in 0..runs {
        let (actual, times) = cpu(data, shape, 1);
        assert_eq!(actual, expected, "actual CPU / sorted reference: {shape:?}");
        cpu_times.extend(times);
    }
    // Each timed invocation starts with its own pristine frontier, never an already-consumed candidate set.
    let buffers = (0..runs).map(|_| data.each_ref().map(|x| fixture.guarded(x, SENTINEL))).collect::<Vec<_>>();
    let kernel = WeaverFrontierSelectVulkanKernel::new(&fixture.context).expect("WeaverFrontierSelect");
    let mut iteration = 0;
    let mut encode = |encoding: &mut _| {
        let buffers = &buffers[iteration];
        iteration += 1;
        // SAFETY: complete, aligned, distinct buffers; batch arithmetic is in the CPU-defined domain.
        unsafe {
            kernel.encode(
                arg(&buffers[0]),
                arg(&buffers[1]),
                arg(&buffers[2]),
                arg(&buffers[3]),
                arg(&buffers[4]),
                arg(&buffers[5]),
                arg(&buffers[6]),
                arg(&buffers[7]),
                arg(&buffers[8]),
                arg(&buffers[9]),
                arg(&buffers[10]),
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                shape[4],
                shape[5],
                shape[6],
                shape[7],
                shape[8],
                encoding,
            );
        }
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
    assert_eq!(iteration, runs);
    for buffers in &buffers {
        for i in 0..11 {
            // SAFETY: every submission has completed; every payload and both guards are checked as raw words.
            let actual = unsafe { KernelFixture::read_guarded::<u32>(&buffers[i], SENTINEL) };
            assert_eq!(actual, expected[i], "CPU / Vulkan buffer {i}: {shape:?}");
        }
    }
    times
}

#[uzu_test]
fn ordering_payloads_and_overlap() {
    let fixture = KernelFixture::new();
    for shape in [
        [1, 2, 1, 0, 1, 2, 4, 3, 1],
        [17, 40, 8, 0, 33, 2, 4, 3, 3],
        [257, 40, 32, 0, 257, 2, 0, 3, 257],
        [2048, 40, 32, 1, 33, 2, 4, 3, 513],
        [513, 33, 32, 1, 1, 1, u32::MAX, 7, 33],
    ] {
        for mode in 0..5 {
            check(&fixture, &case(shape, mode), shape, false);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn zero_work_and_invalid_shapes() {
    let fixture = KernelFixture::new();
    let base = [17, 40, 8, 0, 3, 2, 4, 3, 3];
    for index in [0, 1, 2, 4, 5, 7, 8] {
        let mut shape = base;
        shape[index] = 0;
        check(&fixture, &case(shape, 0), shape, false);
    }
    for (index, value) in [(0, FRONTIER_MAX_SLOTS + 1), (2, FRONTIER_MAX_WIDTH + 1), (3, 40)] {
        let mut shape = base;
        shape[index] = value;
        check(&fixture, &case(shape, 0), shape, false);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn dependent_selection() {
    let fixture = KernelFixture::new();
    let shape = [257, 40, 8, 0, 33, 2, 0, 3, 257];
    let data = case(shape, 3);
    let expected = reference(&reference(&data, shape), shape);
    assert_eq!(cpu(&data, shape, 2).0, expected, "dependent CPU / reference");
    let buffers = data.each_ref().map(|x| fixture.guarded(x, SENTINEL));
    let kernel = WeaverFrontierSelectVulkanKernel::new(&fixture.context).expect("WeaverFrontierSelect");
    let mut encoding = fixture.encoding();
    for _ in 0..2 {
        // SAFETY: valid full shapes; the encoder orders both RAW/WAW accesses to these exact ranges.
        unsafe {
            kernel.encode(
                arg(&buffers[0]),
                arg(&buffers[1]),
                arg(&buffers[2]),
                arg(&buffers[3]),
                arg(&buffers[4]),
                arg(&buffers[5]),
                arg(&buffers[6]),
                arg(&buffers[7]),
                arg(&buffers[8]),
                arg(&buffers[9]),
                arg(&buffers[10]),
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                shape[4],
                shape[5],
                shape[6],
                shape[7],
                shape[8],
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    for i in 0..11 {
        // SAFETY: both dispatches completed before all buffer and guard reads.
        let actual = unsafe { KernelFixture::read_guarded::<u32>(&buffers[i], SENTINEL) };
        assert_eq!(actual, expected[i], "dependent CPU / Vulkan buffer {i}");
    }
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for shape in [
        [32, 40, 1, 0, 8, 8, 4, 4, 32],
        [257, 40, 8, 0, 33, 8, 4, 4, 257],
        [2048, 40, 32, 1, 33, 8, 4, 4, 512],
        [2048, 40, 32, 0, 257, 8, 0, 4, 512],
    ] {
        for mode in [0, 3] {
            let data = case(shape, mode);
            check(&fixture, &data, shape, false);
            let (gpu, wall, cpu) = check(&fixture, &data, shape, true).expect("timed submissions");
            eprintln!("WeaverSelect shape={shape:?} mode={mode}: GPU {gpu:?}, wall {wall:?}, CPU wall {cpu:?}");
        }
    }
    fixture.assert_clean();
}
