use std::{
    fmt::Debug,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use bytemuck::NoUninit;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{check_bounds, kernel_fixture::KernelFixture, round32, silu_oracle, to};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::SigmoidGateKernel},
        cpu::Cpu,
        vulkan::{Error, SigmoidGateVulkanKernel, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// Output elements past the batch, which every dispatch must leave untouched.
const TAIL: usize = 5;

fn kernel<T: ArrayElement>(fixture: &KernelFixture) -> SigmoidGateVulkanKernel {
    SigmoidGateVulkanKernel::new(&fixture.context, T::data_type()).expect("SigmoidGate")
}

/// The CPU kernel through the shared trait on one dispatch: gate rows of `stride`, the in-place output and
/// `(gate_dim, batch_dim, stride)`. CPU buffers cannot be empty.
fn cpu_output<T: ArrayElement + Float>(
    (gate, output, (gate_dim, batch, stride)): &(Vec<T>, Vec<T>, (u32, u32, u32))
) -> Vec<T> {
    if output.is_empty() {
        return Vec::new();
    }
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::SigmoidGateKernel::new(&context, T::data_type())
        .expect("CPU SigmoidGate");
    let gate = create_buffer_with_data::<Cpu, T>(&context, gate);
    let mut result = create_buffer_with_data::<Cpu, T>(&context, output);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(&gate, &mut result, *gate_dim, *batch, *stride, &mut command_buffer);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&result)
}

/// Records every dispatch into `encoding` over guarded ranges, the outputs extending `TAIL` elements past the batch, and
/// completes it. Returns each output after checking the guards, the tail and the unchanged gate with its padding.
fn gpu_outputs<T: ArrayElement + Float + NoUninit>(
    fixture: &KernelFixture,
    kernel: &SigmoidGateVulkanKernel,
    dispatches: &[(Vec<T>, Vec<T>, (u32, u32, u32))],
    mut encoding: VkCommandBufferEncoding,
) -> Vec<Vec<T>> {
    let sentinel = T::from(-7.0).unwrap();
    let buffers = dispatches
        .iter()
        .map(|(gate, output, _)| {
            let output = fixture.guarded(&[output.clone(), vec![sentinel; TAIL]].concat(), sentinel);
            (fixture.guarded(gate, sentinel), output)
        })
        .collect::<Vec<_>>();
    for ((_, _, (gate_dim, batch, stride)), (gate, output)) in dispatches.iter().zip(&buffers) {
        // SAFETY: the gate holds `batch` rows of `stride`, the output `batch * gate_dim` elements plus the tail, both
        // aligned; the output aliases nothing.
        unsafe {
            kernel.encode(
                (&gate.0, gate.1.clone()),
                (&output.0, output.1.clone()),
                *gate_dim,
                *batch,
                *stride,
                &mut encoding,
            )
        };
    }
    KernelFixture::complete(encoding);
    dispatches
        .iter()
        .zip(&buffers)
        .map(|((gate_values, output_values, _), (gate, output))| {
            // SAFETY: the only command buffer using these buffers has completed.
            let values = unsafe {
                KernelFixture::assert_unchanged(gate, sentinel, gate_values, "gate");
                KernelFixture::read_guarded(output, sentinel)
            };
            let tail = bytemuck::cast_slice::<T, u8>(&values[output_values.len()..]);
            assert_eq!(tail, bytemuck::cast_slice::<T, u8>(&[sentinel; TAIL]), "an element past the batch was written");
            values[..output_values.len()].to_vec()
        })
        .collect()
}

/// The CPU's staging of one element as bounds rounded to T: the stored FP32 sigmoid is SiLU of 1 with slope g, whose
/// bounds `silu_oracle` owns (Vulkan's exp and division errors, the tiny gates' exact 1/2), then its correctly rounded
/// FP32 product with the output, monotonic in the sigmoid.
fn oracle<T: Float>(
    gate: T,
    output: T,
) -> ((f64, f64), f64) {
    let ((lo, hi), center) = silu_oracle(1.0, gate.to_f32().unwrap());
    let product = |sigmoid: f64| to::<T>(round32(output.to_f64().unwrap() * sigmoid));
    let (a, b) = (product(lo), product(hi));
    ((a.min(b), a.max(b)), product(center))
}

/// Checks CPU and Vulkan against the staged oracle.
fn check<T: ArrayElement + Float + NoUninit + Debug>(
    fixture: &KernelFixture,
    dispatches: &[(Vec<T>, Vec<T>, (u32, u32, u32))],
    encoding: VkCommandBufferEncoding,
) {
    let outputs = gpu_outputs(fixture, &kernel::<T>(fixture), dispatches, encoding);
    for (dispatch, gpu) in dispatches.iter().zip(outputs) {
        let (gate, output, (gate_dim, batch, stride)) = dispatch;
        let (gate_dim, stride) = (*gate_dim as usize, *stride as usize);
        let bounds = (0..output.len()).map(|i| oracle(gate[i / gate_dim * stride + i % gate_dim], output[i]));
        let case = format!("SigmoidGate {:?} {gate_dim}x{batch} stride {stride}", T::data_type());
        check_bounds(&bounds.collect::<Vec<_>>(), &cpu_output(dispatch), &gpu, &case);
    }
}

/// `batch` rows of `gate_dim` gates in [-12, 12] padded to `stride` with NaN, which the kernel must never read, and
/// outputs in [-4, 4].
fn ordinary<T: Float>(
    (gate_dim, batch, stride): (u32, u32, u32),
    seed: usize,
) -> (Vec<T>, Vec<T>, (u32, u32, u32)) {
    let gate = (0..(batch * stride) as usize)
        .map(|i| match i % stride as usize >= gate_dim as usize {
            true => T::nan(),
            false => T::from(((i * 29 + seed) % 241) as f32 / 10.0 - 12.0).unwrap(),
        })
        .collect();
    let output = (0..(batch * gate_dim) as usize)
        .map(|i| T::from(((i * 13 + seed) % 97) as f32 / 12.0 - 4.0).unwrap())
        .collect();
    (gate, output, (gate_dim, batch, stride))
}

/// Strided rows of odd and tail widths, a single element and rows of whole workgroups, one and three rows each; an
/// empty batch records nothing.
fn matches_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let mut dispatches = Vec::new();
    for (gate_dim, stride) in [(1, 1), (7, 9), (255, 256), (256, 256), (257, 600), (1000, 1031)] {
        for batch in [1, 3] {
            dispatches.push(ordinary::<T>((gate_dim, batch, stride), gate_dim as usize + batch as usize));
        }
    }
    dispatches.push(ordinary((64, 0, 64), 1));
    check(&fixture, &dispatches, fixture.encoding());
    fixture.assert_clean();
}

#[uzu_test]
fn matches_oracle_all_types() {
    matches_oracle::<f32>();
    matches_oracle::<bf16>();
}

/// Every pair of special gates and outputs: NaN, infinities, signed zeros, subnormals, gates up to 2^-26 whose sigmoid
/// is exactly 1/2, the negative tail whose sigmoid is subnormal (from -87.3) or zero once e^-g overflows, times large
/// outputs that bring subnormal sigmoids back to normal results. Infinite outputs skip the gates where the exponential's
/// bounds straddle FP32 overflow, so the sigmoid may be zero or not (the product NaN or infinite).
fn special_values_match_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let gates = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8040_0000),
        2f32.powi(-27),
        -2f32.powi(-26),
        -20.0,
        -87.0,
        -87.5,
        -88.5,
        -95.0,
        -100.0,
        -103.5,
        -110.0,
        -200.0,
        30.0,
        100.0,
        f32::MAX,
        -f32::MAX,
    ];
    let outputs = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::from_bits(0x0000_0003),
        f32::from_bits(0x8070_0000),
        1.5,
        -3.25,
        2f32.powi(100),
        -2f32.powi(110),
        2f32.powi(120),
    ];
    let straddles = |gate: f32| (-88.73..=-88.0).contains(&gate);
    let pairs = gates
        .iter()
        .flat_map(|&gate| outputs.iter().map(move |&output| (gate, output)))
        .filter(|&(gate, output)| !(output.is_infinite() && straddles(gate)))
        .map(|(gate, output)| (T::from(gate).unwrap(), T::from(output).unwrap()))
        .collect::<Vec<_>>();
    let (gate, output): (Vec<T>, Vec<T>) = pairs.into_iter().unzip();
    let gate_dim = gate.len() as u32;
    check(&fixture, &[(gate, output, (gate_dim, 1, gate_dim))], fixture.encoding());
    fixture.assert_clean();
}

#[uzu_test]
fn special_values_match_oracle_all_types() {
    special_values_match_oracle::<f32>();
    special_values_match_oracle::<bf16>();
}

/// The stored sigmoid stage: from -87.34 until e^-g overflows at -88.72 the CPU's sigmoid is a subnormal k 2^-149 of
/// at most 23 bits, so its FP32 product with 2^120 is exactly k 2^-29, while flushing the subnormal gives zero and fusing
/// the output into the division (2^120 / (1 + e^-g)) rounds on the finer FP32 grid of the result, off k 2^-29 for these
/// gates. Each case first asserts the fused value is off the grid, then CPU and Vulkan must lie on it within the staged
/// oracle. BF16 outputs keep too few bits to show the grid.
#[uzu_test]
fn sigmoid_stage_is_stored() {
    let fixture = KernelFixture::new();
    let gates = [-87.5f32, -87.75, -88.0, -88.2];
    let scale = 2f64.powi(120);
    let grid = 2f64.powi(120 - 149);
    for &gate in &gates {
        let fused = to::<f32>(scale / (1.0 + (-f64::from(gate)).exp()));
        assert!((fused / grid).fract() != 0.0, "gate {gate}: fusing lands on the grid");
    }
    let dispatch = (gates.to_vec(), vec![scale as f32; gates.len()], (gates.len() as u32, 1, gates.len() as u32));
    let gpu = gpu_outputs(&fixture, &kernel::<f32>(&fixture), std::slice::from_ref(&dispatch), fixture.encoding());
    let gpu = gpu.into_iter().next().expect("one output");
    let cpu = cpu_output(&dispatch);
    for (side, values) in [("CPU", &cpu), ("Vulkan", &gpu)] {
        for (&gate, value) in gates.iter().zip(values) {
            let value = f64::from(*value);
            assert!(value > 0.0 && (value / grid).fract() == 0.0, "{side} gate {gate}: {value:e} is off the grid");
        }
    }
    check(&fixture, &[dispatch], fixture.encoding());
    fixture.assert_clean();
}

/// Construction rejects F16 and I32. `encode` rejects an empty gate row and a stride below it, also for empty batches,
/// before recording anything; the same command buffer then completes valid work and rejected outputs stay untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for data_type in [DataType::F16, DataType::I32] {
        assert!(
            matches!(
                SigmoidGateVulkanKernel::new(&fixture.context, data_type),
                Err(Error::KernelVariant {
                    kernel: "SigmoidGate",
                    ..
                })
            ),
            "{data_type:?}"
        );
    }
    let kernel = kernel::<f32>(&fixture);
    let untouched = fixture.buffer(&[0u32; 1024]);
    let mut encoding = fixture.encoding();
    for (gate_dim, stride, batch) in [(0, 0, 0), (0, 4, 2), (8, 7, 0), (8, 7, 2), (u32::MAX, u32::MAX - 1, 1)] {
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the precondition fails before recording.
            kernel.encode((&untouched, 0..2048), (&untouched, 2048..4096), gate_dim, batch, stride, &mut encoding)
        }));
        let payload = result.expect_err("encode accepted");
        let message = payload.downcast_ref::<String>().expect("precondition message");
        assert!(message.contains("SigmoidGate: precondition"), "{gate_dim} {stride} {batch}: {message}");
    }
    check(&fixture, &[ordinary::<f32>((33, 2, 40), 3)], encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    assert!(unsafe { KernelFixture::read::<u32>(&untouched) }.iter().all(|&word| word == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// Run alone: `cargo test ... sigmoid_gate_test::throughput -- --ignored --nocapture`. Construction cost, then decode
/// and prefill batches of 32 gated heads of 128 from packed rows of a gated attention projection.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + NoUninit>(fixture: &KernelFixture) {
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                kernel::<T>(fixture);
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!("SigmoidGate {:?} construction: first {first:?}, median of 11 {:?}", T::data_type(), construction[5]);
        let kernel = kernel::<T>(fixture);
        let (gate_dim, stride) = (4096u32, 2 * 4096 + 2 * 8 * 128);
        for batch in [1u32, 128, 1024] {
            let (gate, output, _) = ordinary::<T>((gate_dim, batch, stride), 1);
            let (gate, output) = (fixture.buffer(&gate), fixture.buffer(&output));
            let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                // SAFETY: the gate holds `batch` rows of `stride`, the output `batch * gate_dim`; they do not alias.
                kernel.encode((&gate, 0..gate.size()), (&output, 0..output.size()), gate_dim, batch, stride, encoding);
            });
            let bytes = 3 * u64::from(batch * gate_dim) * size_of::<T>() as u64;
            eprintln!(
                "MEASURE SigmoidGate {:?} {batch}x{gate_dim}: {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?}",
                T::data_type(),
                bytes as f64 / gpu.as_secs_f64() / 1e9
            );
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
