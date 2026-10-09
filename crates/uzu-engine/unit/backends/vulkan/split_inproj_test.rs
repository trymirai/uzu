use std::{fmt::Debug, mem::size_of, time::Duration};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{arg, assert_same_bits, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture, specials};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, kernel::SplitInProjKernel},
        cpu::Cpu,
        vulkan::{SplitInProjVulkanKernel, VkCommandBufferEncoding},
    },
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// (suffix_length, total_dim, conv_dim, inner_dim, num_heads): one element; an exact fit; segment ends inside 256-column
/// groups and a partial last group; total_dim ending inside conv, z and dt; 65 ignored extra columns; each segment empty;
/// no visited column, no rows and no columns; segment sums past u32::MAX; a Mamba2 prefill shape.
const SHAPES: [(u32, u32, u32, u32, u32); 17] = [
    (1, 1, 1, 0, 0),
    (3, 14, 8, 4, 2),
    (2, 517, 255, 257, 5),
    (2, 100, 131, 4, 3),
    (3, 200, 131, 97, 7),
    (4, 230, 131, 97, 7),
    (3, 300, 131, 97, 7),
    (2, 9, 0, 5, 4),
    (2, 9, 5, 0, 4),
    (2, 9, 5, 4, 0),
    (3, 5, 0, 0, 0),
    (0, 9, 5, 2, 2),
    (4, 0, 5, 2, 2),
    (1, 7, u32::MAX - 1, 3, 3),
    (1, 7, 2, u32::MAX, u32::MAX),
    (1, 9, 2, 3, u32::MAX),
    (17, 6448, 3328, 3072, 48),
];

/// No work at u32::MAX scalars, where the CPU loops would take seconds: no dispatch is recorded and nothing changes.
const IDLE: [(u32, u32, u32, u32, u32); 3] = [(u32::MAX, 0, 5, 5, 5), (0, u32::MAX, 5, 5, 5), (u32::MAX, 5, 0, 0, 0)];

/// Elements of the valid spans [input, conv_out, z_out, dt_out, z_bias]: each row visits its leading
/// min(total_dim, conv_dim + inner_dim + num_heads) columns.
fn spans((suffix, total, conv, inner, heads): (u32, u32, u32, u32, u32)) -> [usize; 5] {
    let [suffix, total, conv, inner, heads] = [suffix, total, conv, inner, heads].map(u64::from);
    let columns = total.min(conv + inner + heads);
    let (conv_width, z_width) = (conv.min(columns), inner.min(columns - conv.min(columns)));
    let span = |stride: u64, width: u64| match suffix == 0 || width == 0 {
        true => 0,
        false => ((suffix - 1) * stride + width) as usize,
    };
    let dt = span(heads, columns - conv_width - z_width);
    [span(total, columns), span(conv, conv_width), span(inner, z_width), dt, z_width as usize]
}

/// Raw input bits with the specials at every fifth element, except finite eighths in z columns; z_bias lanes are
/// distinct nonzero eighths, so any misplaced lane changes a finite sum.
fn payload<T: ArrayElement + Float>(dims: (u32, u32, u32, u32, u32)) -> (Vec<T>, Vec<T>) {
    let (_, total, conv, inner, _) = dims;
    let [input_len, .., bias_len] = spans(dims);
    let specials = specials::<T>();
    let input = (0..input_len)
        .map(|i| {
            let column = i as u64 % u64::from(total);
            if column >= u64::from(conv) && column - u64::from(conv) < u64::from(inner) {
                T::from(((i * 37) % 61) as f32 / 8.0 - 3.75).unwrap()
            } else if i % 5 == 0 {
                specials[i / 5 % specials.len()]
            } else {
                let hashed = (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(13).to_ne_bytes();
                bytemuck::pod_read_unaligned(&hashed[..size_of::<T>()])
            }
        })
        .collect();
    let bias = (0..bias_len).map(|j| T::from((3 * (j % 13) + 1) as f32 / 8.0).unwrap()).collect();
    (input, bias)
}

/// z witnesses (input, z_bias, sum rounded to T): cancellation to +0; -0 + -0; the smallest subnormal doubled; the
/// smallest normal minus the smallest subnormal; inf - inf; overflow to inf; ties to even down and up; a NaN operand.
fn witnesses<T: ArrayElement + Float + Debug>() -> [[T; 3]; 9] {
    let (one, two, epsilon, normal) = (T::one(), T::from(2).unwrap(), T::epsilon(), T::min_positive_value());
    let (tiny, half_ulp) = (normal * epsilon, epsilon / two);
    assert!(tiny != T::zero() && half_ulp != T::zero(), "{:?}: tiny {tiny:?}, half ulp {half_ulp:?}", T::data_type());
    let (infinity, nan) = (T::infinity(), T::nan());
    [
        [one, -one, T::zero()],
        [T::neg_zero(), T::neg_zero(), T::neg_zero()],
        [tiny, tiny, tiny * two],
        [normal, -tiny, normal * (one - epsilon)],
        [infinity, -infinity, nan],
        [T::max_value(), T::max_value(), infinity],
        [one, half_ulp, one],
        [one + epsilon, half_ulp, one + epsilon * two],
        [nan, one, nan],
    ]
}

/// The CPU kernel through the shared traits in `submissions` timed submissions, its outputs starting as sentinels.
/// Returns [conv_out, z_out, dt_out] and each submission's wall time.
fn cpu_split<T: ArrayElement + Float + Default>(
    dims: (u32, u32, u32, u32, u32),
    input: &[T],
    bias: &[T],
    submissions: usize,
) -> ([Vec<T>; 3], Vec<Duration>) {
    let (suffix, total, conv, inner, heads) = dims;
    let [_, conv_len, z_len, dt_len, _] = spans(dims);
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::SplitInProjKernel::new(&context, T::data_type())
        .expect("CPU SplitInProj");
    let (input_buffer, bias_buffer) = (cpu_buffer(&context, input), cpu_buffer(&context, bias));
    let mut outputs = [conv_len, z_len, dt_len].map(|len| cpu_buffer(&context, &vec![T::from(-7.0).unwrap(); len]));
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        let [conv_out, z_out, dt_out] = &mut outputs;
        kernel.encode(
            &input_buffer,
            conv_out,
            z_out,
            dt_out,
            &bias_buffer,
            suffix,
            total,
            conv,
            inner,
            heads,
            command_buffer,
        );
    });
    let lens = [conv_len, z_len, dt_len];
    (std::array::from_fn(|i| buffer_prefix_to_vec::<Cpu, T>(&outputs[i], lens[i])), times)
}

/// The Vulkan counterpart of `cpu_split` over guarded ranges of exactly the valid spans, recorded once or, when `timed`,
/// in `median_times` submissions. When `chained`, a first dispatch in the same command buffer copies the input through
/// conv_out (conv_dim = total_dim) into the buffer the second reads. Asserts input, z_bias, the copy and every guard.
fn gpu_split<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &SplitInProjVulkanKernel,
    dims: (u32, u32, u32, u32, u32),
    input: &[T],
    bias: &[T],
    chained: bool,
    timed: bool,
) -> ([Vec<T>; 3], Option<(Duration, Duration)>) {
    let (suffix, total, conv, inner, heads) = dims;
    let [_, conv_len, z_len, dt_len, _] = spans(dims);
    let s = T::from(-7.0).unwrap();
    let [input_buffer, bias_buffer, copy, conv_out, z_out, dt_out, empty] =
        [input, bias, &vec![s; input.len()], &vec![s; conv_len], &vec![s; z_len], &vec![s; dt_len], &[]]
            .map(|values| fixture.guarded(values, s));
    // SAFETY: each range holds exactly the aligned valid span of its argument; written ranges alias nothing (the empty
    // ones hold no element), and the copy covers every column of the input.
    let mut record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let source = match chained {
            true => {
                let (input, empty) = (arg(&input_buffer), arg(&empty));
                kernel.encode(
                    input,
                    arg(&copy),
                    empty.clone(),
                    empty.clone(),
                    empty,
                    suffix,
                    total,
                    total,
                    0,
                    0,
                    encoding,
                );
                &copy
            },
            false => &input_buffer,
        };
        let (outputs, bias) = ([&conv_out, &z_out, &dt_out].map(arg), arg(&bias_buffer));
        let [conv_out, z_out, dt_out] = outputs;
        kernel.encode(arg(source), conv_out, z_out, dt_out, bias, suffix, total, conv, inner, heads, encoding);
    };
    let times = match timed {
        true => Some(fixture.median_times(&mut record)),
        false => {
            let mut encoding = fixture.encoding();
            record(&mut encoding);
            KernelFixture::complete(encoding);
            None
        },
    };
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&input_buffer, s, input, "input");
        KernelFixture::assert_unchanged(&bias_buffer, s, bias, "z_bias");
        KernelFixture::assert_unchanged(&empty, s, &[], "empty");
        if chained {
            assert_same_bits(input, &KernelFixture::read_guarded(&copy, s), "chained copy");
        }
        ([&conv_out, &z_out, &dt_out].map(|output| KernelFixture::read_guarded(output, s)), times)
    }
}

/// Vulkan alone and, where every column is visited, chained after a copy, against the CPU: copies and every sentinel
/// slot of the valid spans bit for bit with NaN payloads, z bit for bit up to the NaN payload. Returns the CPU z_out.
fn check<T: ArrayElement + Float + Debug + Default>(
    fixture: &KernelFixture,
    kernel: &SplitInProjVulkanKernel,
    dims: (u32, u32, u32, u32, u32),
    input: &[T],
    bias: &[T],
) -> Vec<T> {
    let ([conv_out, z_out, dt_out], _) = cpu_split(dims, input, bias, 1);
    let every_column = spans(dims)[0] == dims.0 as usize * dims.1 as usize;
    for chained in [false, true].into_iter().filter(|&chained| !chained || every_column) {
        let label = format!("{:?} {dims:?} chained {chained}", T::data_type());
        let ([conv, z, dt], _) = gpu_split(fixture, kernel, dims, input, bias, chained, false);
        assert_same_bits(&conv_out, &conv, &format!("{label} conv_out"));
        KernelFixture::assert_bits(&z_out, &z, &format!("{label} z_out"));
        assert_same_bits(&dt_out, &dt, &format!("{label} dt_out"));
    }
    z_out
}

fn matches_cpu<T: ArrayElement + Float + Debug + Default>() {
    let fixture = KernelFixture::new();
    let kernel = SplitInProjVulkanKernel::new(&fixture.context, T::data_type()).expect("Vulkan SplitInProj");
    for dims in SHAPES {
        let (input, bias) = payload::<T>(dims);
        check(&fixture, &kernel, dims, &input, &bias);
    }
    for dims in IDLE {
        gpu_split::<T>(&fixture, &kernel, dims, &[], &[], false, false);
    }
    // Both rows hold the witnesses in their z columns.
    let witnesses = witnesses::<T>();
    let dims = (2, 14, 3, 9, 2);
    let (mut input, _) = payload::<T>(dims);
    for (i, [a, ..]) in witnesses.iter().chain(&witnesses).enumerate() {
        input[i / 9 * 14 + 3 + i % 9] = *a;
    }
    let bias = witnesses.iter().map(|[_, b, _]| *b).collect::<Vec<_>>();
    let sums = witnesses.iter().chain(&witnesses).map(|[.., sum]| *sum).collect::<Vec<_>>();
    let z_out = check(&fixture, &kernel, dims, &input, &bias);
    KernelFixture::assert_bits(&sums, &z_out, &format!("{:?} z witnesses", T::data_type()));
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_all_types() {
    matches_cpu::<f32>();
    matches_cpu::<f16>();
    matches_cpu::<bf16>();
}

/// Run alone, without sync validation: `... split_inproj_test::throughput -- --ignored --nocapture`. At a Mamba2 shape
/// (128 heads of 64, 8 groups of state 128) for decode and a 512-token prefill, prints the GPU and wall medians of 10
/// Vulkan submissions after 3 warm-up ones, the CPU kernel's wall median, and the logical bytes the shader loads and
/// stores: each input element once, each output element once, and z_bias once per dispatched row group.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + Debug + Default>(fixture: &KernelFixture) {
        let kernel = SplitInProjVulkanKernel::new(&fixture.context, T::data_type()).expect("Vulkan SplitInProj");
        let row_group_limit = fixture.context.physical_device().properties.limits.max_compute_work_group_count[1];
        for suffix in [1, 512] {
            let dims = (suffix, 18560, 10240, 8192, 128);
            let (input, bias) = payload::<T>(dims);
            let (_, times) = gpu_split(fixture, &kernel, dims, &input, &bias, false, true);
            let (gpu, wall) = times.expect("timed");
            let mut cpu = cpu_split(dims, &input, &bias, 13).1.split_off(3);
            cpu.sort();
            let row_groups = suffix.min(row_group_limit) as usize;
            let bytes = (2 * input.len() + row_groups * bias.len()) * size_of::<T>();
            let rate = bytes as f64 / gpu.as_secs_f64() / 1e9;
            let cpu = cpu[cpu.len() / 2];
            eprintln!(
                "SplitInProj {:?} {dims:?}: {bytes} logical B; GPU {gpu:?} ({rate:.1} logical GB/s), wall {wall:?}; CPU \
                 wall {cpu:?}",
                T::data_type()
            );
        }
    }
    let fixture = KernelFixture::new();
    // Round 0 settles the GPU clocks; compare round 1.
    for round in 0..2 {
        eprintln!("SplitInProj throughput round {round}");
        measure::<f32>(&fixture);
        measure::<bf16>(&fixture);
    }
    fixture.assert_clean();
}
