use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{arg, conv1d_values, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{Backend, Kernels, kernel::SeparableCausalConvKernel},
        cpu::Cpu,
        vulkan::{Error, SeparableCausalConvVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

/// [C, K, G, Q, R] (model_dim, kernel_size, group_size, sequence_length, coefficient_row_stride): the common test's
/// shape with slack coefficients; one element; a partial group whose last channels read into the next tap's row; C below
/// G, so every tap reads its row's first coefficient, in overlapping rows; a channel tail past one workgroup with C and G
/// not multiples of 4; no taps; K past Q; one shared coefficient row; more tokens than 65535 workgroups; no tokens and
/// no channels at large unused K and R, where nothing is addressed.
const SHAPES: [[u32; 5]; 11] = [
    [16, 2, 16, 2, 4],
    [1, 1, 1, 1, 1],
    [10, 3, 4, 5, 6],
    [3, 4, 8, 6, 1],
    [67, 4, 5, 7, 15],
    [9, 0, 3, 3, 0],
    [16, 8, 4, 3, 16],
    [8, 3, 2, 4, 0],
    [1, 2, 1, 70000, 2],
    [16, 8, 4, 0, u32::MAX],
    [0, 8, 4, 5, u32::MAX],
];

/// Index witnesses WA to WD with their coefficient_deltas allocations, which also hold every element a ceil(C / G) row
/// would address.
const INDEX_WITNESSES: [([u32; 5], usize); 4] =
    [([10, 3, 4, 5, 6], 33), ([3, 4, 8, 6, 1], 9), ([8, 3, 2, 4, 0], 12), ([10, 2, 3, 3, 8], 24)];

const NAMES: [&str; 4] = ["input", "coefficient_deltas", "weights", "bias"];
const PRESENCE: &str = "SeparableCausalConv: argument 'bias' must be present exactly when has_bias";
const PRECONDITION: &str = "SeparableCausalConv: precondition group_size > 0 violated";
const CPU_FAILURE: &str = "called `Result::unwrap()` on an `Err` value: CommandBufferExecutionFailed(RecvError)";

fn fill() -> bf16 {
    bf16::from_f32(-7.0)
}

fn bits(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|&value| bf16::from_f32(value)).collect()
}

/// Elements of [input, coefficient_deltas, weights, bias, output] the kernel reads or writes: the inputs only for taps
/// that exist, the bias only for outputs that exist.
fn extents(
    shape: [u32; 5],
    has_bias: bool,
) -> [usize; 5] {
    let [c, k, g, q, r] = shape.map(u64::from);
    let [input, deltas, weights] = match q != 0 && c != 0 && k != 0 {
        true => [q * c, (q - 1) * r + (k.min(q) - 1) * (c / g) + (c - 1) / g + 1, c * k],
        false => [0; 3],
    };
    let bias = match has_bias && q * c != 0 {
        true => c,
        false => 0,
    };
    [input, deltas, weights, bias, q * c].map(|len| len as usize)
}

/// Finite eighths over the extents, with specials at every `every`-th input, coefficient and weight element when
/// `every` > 0.
fn inputs(
    shape: [u32; 5],
    has_bias: bool,
    every: usize,
) -> [Vec<bf16>; 4] {
    let lens = extents(shape, has_bias);
    std::array::from_fn(|i| {
        conv1d_values(
            lens[i],
            i,
            if i < 3 {
                every
            } else {
                0
            },
        )
    })
}

/// The CPU kernel on its own fresh context in `submissions` timed submissions, the output starting as sentinels: the
/// output and the wall times.
fn cpu_run(
    shape: [u32; 5],
    has_bias: bool,
    inputs: &[Vec<bf16>; 4],
    submissions: usize,
) -> (Vec<bf16>, Vec<Duration>) {
    let [c, k, g, q, r] = shape;
    let len = q as usize * c as usize;
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::SeparableCausalConvKernel::new(
        &context,
        DataType::BF16,
        c,
        k,
        g,
        has_bias,
    )
    .expect("CPU SeparableCausalConv");
    let [input, deltas, weights, bias] = inputs.each_ref().map(|values| cpu_buffer(&context, values));
    let mut output = cpu_buffer(&context, &vec![fill(); len]);
    let times = cpu_submissions(&context, submissions, |command_buffer| {
        kernel.encode(&input, &deltas, &weights, has_bias.then_some(&bias), &mut output, q, r, command_buffer);
    });
    (buffer_prefix_to_vec::<Cpu, bf16>(&output, len), times)
}

/// Asserts that every input range and its guards still hold their payload and sentinels.
///
/// # Safety
/// Every command buffer using the buffers has completed.
unsafe fn assert_inputs(
    buffers: &[(Arc<VkBuffer>, Range<u64>); 4],
    inputs: &[Vec<bf16>; 4],
) {
    for ((name, guarded), values) in NAMES.into_iter().zip(buffers).zip(inputs) {
        unsafe { KernelFixture::assert_unchanged(guarded, fill(), values, name) };
    }
}

/// The Vulkan counterpart of `cpu_run` over guarded ranges of exactly `inputs` and the output, recorded once or, when
/// `timed`, in `median_times` submissions; asserts every input and guard unchanged.
fn gpu_run(
    fixture: &KernelFixture,
    kernel: &SeparableCausalConvVulkanKernel,
    shape: [u32; 5],
    has_bias: bool,
    inputs: &[Vec<bf16>; 4],
    timed: bool,
) -> (Vec<bf16>, Option<(Duration, Duration)>) {
    let [c, _, _, q, r] = shape;
    let buffers = inputs.each_ref().map(|values| fixture.guarded(values, fill()));
    let output = fixture.guarded(&vec![fill(); q as usize * c as usize], fill());
    // SAFETY: each range holds every element the shape addresses, aligned, and the output aliases nothing.
    let mut record = |encoding: &mut VkCommandBufferEncoding| unsafe {
        let [input, deltas, weights, bias] = buffers.each_ref().map(arg);
        kernel.encode(input, deltas, weights, has_bias.then_some(bias), arg(&output), q, r, encoding)
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
        assert_inputs(&buffers, inputs);
        (KernelFixture::read_guarded(&output, fill()), times)
    }
}

/// [CPU, Vulkan] outputs of one case.
fn run(
    fixture: &KernelFixture,
    shape: [u32; 5],
    has_bias: bool,
    inputs: &[Vec<bf16>; 4],
) -> [Vec<bf16>; 2] {
    let [c, k, g, ..] = shape;
    let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, c, k, g, has_bias)
        .expect("Vulkan SeparableCausalConv");
    [cpu_run(shape, has_bias, inputs, 1).0, gpu_run(fixture, &kernel, shape, has_bias, inputs, false).0]
}

/// A hand-built case whose CPU and Vulkan outputs must both be `expected`, bit for bit up to NaN payloads.
fn witness(
    fixture: &KernelFixture,
    label: &str,
    shape: [u32; 5],
    has_bias: bool,
    inputs: [Vec<bf16>; 4],
    expected: &[bf16],
) {
    let [cpu, gpu] = run(fixture, shape, has_bias, &inputs);
    KernelFixture::assert_bits(expected, &cpu, &format!("{label} CPU"));
    KernelFixture::assert_bits(expected, &gpu, &format!("{label} Vulkan"));
}

/// Every shape with and without a bias over exactly its addressed extents: Vulkan's full output is the CPU's bit for
/// bit up to NaN payloads, and every input and guard is unchanged.
#[uzu_test]
fn matches_cpu() {
    let fixture = KernelFixture::new();
    for shape in SHAPES {
        for has_bias in [false, true] {
            let [cpu, gpu] = run(&fixture, shape, has_bias, &inputs(shape, has_bias, 5));
            KernelFixture::assert_bits(&cpu, &gpu, &format!("{shape:?} bias {has_bias}"));
        }
    }
    fixture.assert_clean();
}

/// Exact outputs derived here, independently of the CPU code, which the CPU and Vulkan must both produce. Index sums:
/// with unit inputs, zero weights and coefficient k + 1 at element k, an output is the sum over its taps of
/// t R + j (C / G) + c / G + 1, below 256 and so exact. Order: the products 2^24, 1 and -2^24 cancel to +0 only in
/// ascending order. Separately rounded: -(1 + 2^-7 + 2^-17), then the product (1 + 2^-7)(1 + 2^-17) rounding to
/// 1 + 2^-7 + 2^-17, give +0, where a fused one gives 2^-24. Bias first: 2^24 + 1 rounds back to 2^24 before -2^24
/// cancels it. Signed zeros, subnormal products and sums, infinities and NaNs.
#[uzu_test]
fn exact_witnesses() {
    let fixture = KernelFixture::new();
    let (one, zero, big) = (bf16::ONE, bf16::ZERO, 2f32.powi(24));
    for (shape, allocation) in INDEX_WITNESSES {
        let [c, k, g, q, r] = shape.map(|n| n as usize);
        let expected: Vec<bf16> = (0..q * c)
            .map(|i| {
                let (t, channel) = (i / c, i % c);
                let sum: usize = (0..k.min(t + 1)).map(|j| t * r + j * (c / g) + channel / g + 1).sum();
                bf16::from_f32(sum as f32)
            })
            .collect();
        let deltas = (1..=allocation).map(|element| bf16::from_f32(element as f32)).collect();
        let inputs = [vec![one; q * c], deltas, vec![zero; c * k], vec![]];
        witness(&fixture, &format!("index {shape:?}"), shape, false, inputs, &expected);
    }

    let order = [bits(&[1.0; 3]), vec![zero; 5], bits(&[-big, 1.0, big]), vec![]];
    witness(&fixture, "order", [1, 3, 1, 3, 1], false, order, &bits(&[big, big, 0.0]));

    let (step, tiny) = (1.0 + 2f32.powi(-7), 2f32.powi(-17));
    let unfused = [bits(&[step, -1.0]), bits(&[0.0, 0.0, tiny, tiny]), bits(&[1.0, step]), vec![]];
    witness(&fixture, "unfused", [1, 2, 1, 2, 2], false, unfused, &bits(&[1.0 + 2f32.powi(-6), 0.0]));

    // Channel 0 starts from 2^24, channel 1 from -0 with -0 products.
    let bias_first = [bits(&[1.0, -0.0, 1.0, -0.0]), vec![zero; 8], bits(&[-big, 1.0, 1.0, 1.0]), bits(&[big, -0.0])];
    witness(&fixture, "bias first", [2, 2, 1, 2, 4], true, bias_first, &bits(&[big, -0.0, 0.0, -0.0]));
    let no_taps = [vec![], vec![], vec![], bits(&[-0.0])];
    witness(&fixture, "-0 bias without taps", [1, 0, 1, 1, 0], true, no_taps, &bits(&[-0.0]));

    // Per channel: a subnormal input; a product below the normal range; a sum of subnormals; +0 plus a -0 product;
    // infinities; infinity times zero; an infinite and opposite coefficient sum; a signalling NaN input.
    let (inf, sub, signalling) = (f32::INFINITY, bf16::from_bits(0x0001), bf16::from_bits(0x7F81));
    let mut x = bits(&[0.0, 2f32.powi(-65), 2f32.powi(60), -0.0, inf, -inf, inf, 1.0, 0.0]);
    let mut w = bits(&[1.5, 2f32.powi(-65), 0.0, 1.0, 1.0, 1.0, 0.0, inf, 1.0]);
    let mut d = bits(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, 0.0]);
    (x[0], x[8], w[2], d[2]) = (sub, signalling, sub, sub);
    let mut expected = bits(&[0.0, 0.0, 2f32.powi(-72), 0.0, inf, -inf, f32::NAN, f32::NAN, f32::NAN]);
    (expected[0], expected[1]) = (bf16::from_bits(0x0002), bf16::from_bits(0x0008));
    witness(&fixture, "classes", [9, 1, 1, 1, 9], false, [x, d, w, vec![]], &expected);
    fixture.assert_clean();
}

/// The DSL's presence check, an API invariant of the generated binding rather than of the CPU kernel (which ignores an
/// unused bias and reads a missing one only when it has work): a missing active bias and a present inactive one fail its
/// assert_eq, with the full diagnostic of both booleans, before anything is recorded, also without work, while an active
/// bias without work may be an empty range.
#[uzu_test]
fn optional_bias_presence() {
    let fixture = KernelFixture::new();
    let values = inputs([4, 2, 2, 3, 2], true, 0);
    let buffers = values.each_ref().map(|values| fixture.guarded(values, fill()));
    let (output, empty) = (fixture.guarded(&[fill(); 12], fill()), fixture.guarded::<bf16>(&[], fill()));
    for (has_bias, [c, q]) in [(true, [4, 3]), (true, [4, 0]), (true, [0, 3]), (false, [4, 3]), (false, [4, 0])] {
        let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, c, 2, 2, has_bias)
            .expect("SeparableCausalConv");
        let [input, deltas, weights, bias] = buffers.each_ref().map(arg);
        let mut encoding = fixture.encoding();
        // SAFETY: the presence check panics before anything is recorded.
        let encoded = catch_unwind(AssertUnwindSafe(|| unsafe {
            kernel.encode(input, deltas, weights, (!has_bias).then_some(bias), arg(&output), q, 2, &mut encoding)
        }));
        let message = encoded.expect_err("wrong bias presence encoded").downcast::<String>().expect("panic message");
        let expected =
            format!("assertion `left == right` failed: {PRESENCE}\n  left: {}\n right: {has_bias}", !has_bias);
        assert_eq!(*message, expected, "bias {has_bias}, C {c}, Q {q}");
        KernelFixture::complete(encoding);
    }
    let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, 4, 2, 2, true)
        .expect("SeparableCausalConv");
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    // SAFETY: without work nothing is indexed or recorded.
    unsafe { kernel.encode(e(), e(), e(), Some(e()), e(), 0, 2, &mut encoding) };
    KernelFixture::complete(encoding);
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&buffers, &values);
        KernelFixture::assert_unchanged(&output, fill(), &[fill(); 12], "output");
        KernelFixture::assert_unchanged(&empty, fill(), &[], "empty");
    }
    fixture.assert_clean();
}

/// Group size 0, with the bias present exactly when active: the CPU divides C by it before any work, failing also
/// without work as the failed wait of its own fresh context, and Vulkan's encode precondition fails the same shapes
/// before anything is recorded. Only BF16 exists.
#[uzu_test]
fn rejects_invalid_configurations() {
    let fixture = KernelFixture::new();
    let values = inputs([4, 2, 2, 3, 2], true, 0);
    let buffers = values.each_ref().map(|values| fixture.guarded(values, fill()));
    let output = fixture.guarded(&[fill(); 12], fill());
    for (has_bias, [c, q]) in [(false, [4, 3]), (true, [4, 3]), (false, [4, 0]), (true, [0, 3])] {
        let shape = [c, 2, 0, q, 2];
        let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, c, 2, 0, has_bias)
            .expect("SeparableCausalConv");
        let [input, deltas, weights, bias] = buffers.each_ref().map(arg);
        let mut encoding = fixture.encoding();
        // SAFETY: the precondition panics before anything is recorded.
        let encoded = catch_unwind(AssertUnwindSafe(|| unsafe {
            kernel.encode(input, deltas, weights, has_bias.then_some(bias), arg(&output), q, 2, &mut encoding)
        }));
        let message = encoded.expect_err("group size 0 encoded").downcast::<String>().expect("panic message");
        assert_eq!(*message, PRECONDITION, "{shape:?} bias {has_bias}");
        KernelFixture::complete(encoding);
        let cpu = catch_unwind(AssertUnwindSafe(|| cpu_run(shape, has_bias, &values, 1)));
        let message = cpu.expect_err("CPU accepted group size 0").downcast::<String>().expect("panic message");
        assert_eq!(*message, CPU_FAILURE, "{shape:?} bias {has_bias}");
    }
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        assert_inputs(&buffers, &values);
        KernelFixture::assert_unchanged(&output, fill(), &[fill(); 12], "output");
    }
    let f32_kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::F32, 4, 2, 2, false);
    assert!(matches!(
        f32_kernel,
        Err(Error::KernelVariant {
            kernel: "SeparableCausalConv",
            ..
        })
    ));
    fixture.assert_clean();
}

/// No work at u32::MAX scalars, where the kernel would otherwise loop for long: no tokens, no channels or neither, with
/// and without a bias (an empty range when active). Nothing is recorded or changed.
#[uzu_test]
fn zero_work_records_nothing() {
    let fixture = KernelFixture::new();
    let (empty, m) = (fixture.guarded::<bf16>(&[], fill()), u32::MAX);
    let e = || arg(&empty);
    let mut encoding = fixture.encoding();
    for [c, q] in [[m, 0], [0, m], [0, 0]] {
        for has_bias in [false, true] {
            let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, c, m, 4, has_bias)
                .expect("SeparableCausalConv");
            // SAFETY: without work nothing is indexed or recorded.
            unsafe { kernel.encode(e(), e(), e(), has_bias.then(e), e(), q, m, &mut encoding) };
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using the buffer has completed.
    unsafe { KernelFixture::assert_unchanged(&empty, fill(), &[], "empty") };
    fixture.assert_clean();
}

/// Run alone, without sync validation: `... separable_causal_conv_test::throughput -- --ignored --nocapture`. A synthetic
/// family, not model shapes: [C, K, G, Q, R] = [4096, 4, 128, Q, 128] with a bias for Q 1, 64 and 1024, in two rounds
/// of opposite order, from finite eighths. Prints the GPU and wall medians of 10 Vulkan submissions after 3 warm-up
/// ones and the CPU kernel's wall median, after the full output matched the CPU's after one submission and after the
/// last timed one. The bytes are the logical traffic, three BF16 reads per executed tap and a bias read and an output
/// write per output, not measured memory bandwidth.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let kernel = SeparableCausalConvVulkanKernel::new(&fixture.context, DataType::BF16, 4096, 4, 128, true)
        .expect("SeparableCausalConv");
    for round in 0..2 {
        eprintln!("SeparableCausalConv throughput round {round}");
        let mut lengths = [1, 64, 1024];
        if round == 1 {
            lengths.reverse();
        }
        for q in lengths {
            let shape = [4096, 4, 128, q, 128];
            let values = inputs(shape, true, 0);
            let (cpu, mut cpu_times) = cpu_run(shape, true, &values, 13);
            let label = format!("{shape:?}");
            KernelFixture::assert_bits(&cpu, &gpu_run(&fixture, &kernel, shape, true, &values, false).0, &label);
            let (output, times) = gpu_run(&fixture, &kernel, shape, true, &values, true);
            KernelFixture::assert_bits(&cpu, &output, &format!("{label} after timing"));
            let (gpu, wall) = times.expect("timed");
            cpu_times.drain(..3);
            cpu_times.sort();
            let (c, k, q) = (4096u64, 4u64, u64::from(q));
            let m = k.min(q);
            let bytes = 2 * c * (3 * (m * (m + 1) / 2 + (q - m) * m) + 2 * q);
            eprintln!(
                "SeparableCausalConv {label}: {bytes} B logical traffic ({:.1} GB/s effective); GPU {gpu:?}, wall \
                 {wall:?}; CPU wall {:?}",
                bytes as f64 / gpu.as_secs_f64() / 1e9,
                cpu_times[cpu_times.len() / 2]
            );
        }
    }
    fixture.assert_clean();
}
