use std::{fmt::Debug, time::Duration};

use bytemuck::Pod;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{kernel_fixture::KernelFixture, sentinel};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, Context, Kernels,
            kernel::{
                DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel,
                delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            },
        },
        cpu::Cpu,
        vulkan::{VkCommandBufferEncoding, Vulkan},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, submit_command_buffer},
};

const HEAD_K_DIM: u32 = 128;
const CHUNK: usize = 64;

/// [H, G, Dv, Q] (num_heads, num_groups, value_head_dim, suffix_len): four full chunks at Dv 32; three full chunks and a
/// one-token tail at Dv 64; four v heads per k head over two chunks and a one-token tail.
const SHAPES: [[u32; 4]; 3] = [[4, 2, 32, 256], [4, 2, 64, 257], [8, 2, 64, 129]];

/// The Metal chunked test's output and state tolerances, as (absolute, relative): the chunked form reassociates the
/// recurrence, so comparison with the sequential CPU allows for reassociation and output rounding.
const OUTPUT_TOLERANCE: (f32, f32) = (5e-3, 1e-3);
const STATE_TOLERANCE: (f32, f32) = (2e-4, 1e-3);

/// A deterministic multiple of 1/64 in [-1, 1], exact in BF16.
fn grid(
    i: usize,
    seed: u32,
) -> f32 {
    let x = (i as u32).wrapping_mul(0x9E37_79B1) ^ seed.wrapping_mul(0x85EB_CA6B);
    ((x >> 15) % 129) as f32 / 64.0 - 1.0
}

/// in_proj rows [q, k, v, z, beta_raw, a_raw] and [a_log, dt_bias, state]. Every 37th token from the 5th has zero q and k
/// rows in k head 0. beta is near 1 or near 0, flipping per v head and per chunk. a_log 1 decays strongly, -6 barely; a_raw
/// 16 on each chunk's last token and on token 64 resets the strong heads, across and right after a chunk boundary.
fn inputs(
    [h, g, dv, q]: [u32; 4],
    zero: bool,
) -> (Vec<f32>, [Vec<f32>; 3]) {
    let [h, g, dv, q] = [h, g, dv, q].map(|n| n as usize);
    let (key, value, row) = (g * 128, h * dv, 2 * (g * 128 + h * dv + h));
    let projection = (0..q * row).map(|i| {
        let (t, c) = (i / row, i % row);
        match c {
            _ if c < 2 * key => f32::from(u8::from(t % 37 != 5 || c % key >= 128)) * grid(i, 0),
            _ if c < 2 * (key + value) => grid(i, 1) / 2.0,
            _ if c < 2 * (key + value) + h => {
                let strong = (c - 2 * (key + value) + t / CHUNK).is_multiple_of(2);
                (if strong {
                    8.0
                } else {
                    -8.0
                }) + grid(i, 2) / 4.0
            },
            _ if t % CHUNK == CHUNK - 1 || t == CHUNK => 16.0,
            _ => grid(i, 3) / 2.0,
        }
    });
    let a_log = (0..h).map(|hv| [1.0, -6.0, -1.0, 0.5][hv % 4]).collect();
    let dt_bias = (0..h).map(|hv| [0.25, -0.5, 0.0, 0.5][hv % 4]).collect();
    let state = (0..h * dv * 128).map(|i| grid(i, 4) / 4.0);
    match zero {
        true => (vec![0.0; q * row], [a_log, dt_bias, vec![0.0; h * dv * 128]]),
        false => (projection.collect(), [a_log, dt_bias, state.collect()]),
    }
}

/// The engine's sequential path on the CPU: Prep to F32 q, k, beta and decay, then Prefill: [delta_output, state].
fn cpu_reference<T: ArrayElement>(
    [h, g, dv, q]: [u32; 4],
    in_proj: &[T],
    [a_log, dt_bias, state]: &[Vec<f32>; 3],
) -> (Vec<T>, Vec<f32>) {
    let (key, value) = (g * HEAD_K_DIM, h * dv);
    let context = create_context::<Cpu>();
    let prep = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        T::data_type(),
        DataType::F32,
        HEAD_K_DIM,
        false,
        false,
    )
    .expect("CPU DeltaNetPrefillPrep");
    let prefill =
        <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(&context, T::data_type(), HEAD_K_DIM)
            .expect("CPU DeltaNetPrefill");
    let projection = create_buffer_with_data::<Cpu, T>(&context, in_proj);
    let [a_log, dt_bias] = [a_log, dt_bias].map(|values| create_buffer_with_data::<Cpu, f32>(&context, values));
    let [mut q_norm, mut k_norm] = [0; 2].map(|_| create_buffer::<Cpu, f32>(&context, (q * key) as usize));
    let [mut beta, mut decay] = [0; 2].map(|_| create_buffer::<Cpu, f32>(&context, (q * h) as usize));
    let mut ssm_state = create_buffer_with_data::<Cpu, f32>(&context, state);
    let mut out = create_buffer::<Cpu, T>(&context, (q * value) as usize);
    let mut encoding = context.create_command_buffer(None, None).expect("CPU command buffer");
    prep.encode(
        &projection,
        &a_log,
        &dt_bias,
        &mut q_norm,
        &mut k_norm,
        None::<&mut <Cpu as Backend>::GlobalBuffer>,
        &mut beta,
        &mut decay,
        h,
        g,
        key,
        value,
        q,
        &mut encoding,
    );
    prefill.encode(
        &q_norm,
        &k_norm,
        &beta,
        &decay,
        &projection,
        &mut ssm_state,
        &mut out,
        h,
        g,
        dv,
        key,
        value,
        q,
        dv.div_ceil(16),
        &mut encoding,
    );
    submit_command_buffer(encoding);
    (buffer_to_vec::<Cpu, T>(&out), buffer_to_vec::<Cpu, f32>(&ssm_state))
}

/// A scratch allocation holding `values`, the rest of its pool page `fill`.
fn scratch<T: Pod>(
    encoding: &mut VkCommandBufferEncoding,
    values: &[T],
    fill: T,
) -> <Vulkan as Backend>::ScratchBuffer {
    let scratch = encoding.allocate_scratch(size_of_val(values)).expect("scratch");
    let page = scratch.page();
    let mut elements = vec![fill; page.size() as usize / size_of::<T>()];
    elements[..values.len()].copy_from_slice(values);
    let bytes = bytemuck::cast_slice::<T, u8>(&elements);
    // SAFETY: no submitted command buffer uses the fresh page, and the copy stays inside it.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page.cpu_ptr().as_ptr().cast::<u8>(), bytes.len()) };
    scratch
}

/// The first `len` elements of a scratch allocation, after asserting the rest of its page still holds sentinels.
///
/// # Safety
/// Every command buffer using the page has completed.
unsafe fn read_scratch<T: Pod + Float>(
    scratch: &<Vulkan as Backend>::ScratchBuffer,
    len: usize,
    label: &str,
) -> Vec<T> {
    let mut values = bytemuck::pod_collect_to_vec::<u8, T>(&unsafe { scratch.page().get_bytes() });
    let slack = values.split_off(len);
    assert!(
        slack.iter().all(|v| bytemuck::bytes_of(v) == bytemuck::bytes_of(&sentinel::<T>())),
        "{label}: slack written"
    );
    values
}

/// The Vulkan chunked prefill through the common trait, delta_output starting as sentinels: [delta_output, state] after
/// asserting in_proj, a_log, dt_bias and the scratch slack unchanged.
fn vulkan<T: ArrayElement + Pod + Float + Debug>(
    fixture: &KernelFixture,
    [h, g, dv, q]: [u32; 4],
    in_proj: &[T],
    [a_log, dt_bias, state]: &[Vec<f32>; 3],
    label: &str,
    chunked: bool,
) -> (Vec<T>, Vec<f32>, Duration) {
    let context = &*fixture.context;
    let kernel =
        <<Vulkan as Backend>::Kernels as Kernels>::DeltaNetChunkedPrefill::new(context, T::data_type(), HEAD_K_DIM)
            .expect("DeltaNetChunkedPrefill")
            .expect("DeltaNetChunkedPrefill supported on this Vulkan context");
    let mut encoding = context.create_command_buffer(None, None).expect("command buffer");
    let out_len = (q * h * dv) as usize;
    let projection = scratch(&mut encoding, in_proj, sentinel::<T>());
    let mut out = scratch(&mut encoding, &vec![sentinel::<T>(); out_len], sentinel::<T>());
    let [a_log_buffer, dt_bias_buffer] = [a_log, dt_bias].map(|v| create_buffer_with_data::<Vulkan, f32>(context, v));
    let mut ssm_state = create_buffer_with_data::<Vulkan, f32>(context, state);
    let args = DeltaNetChunkedPrefillArgs {
        in_projected: &projection,
        a_log: &a_log_buffer,
        dt_bias: &dt_bias_buffer,
        ssm_state: &mut ssm_state,
        delta_output: &mut out,
        num_heads: h,
        num_groups: g,
        value_head_dim: dv,
        key_dim: g * HEAD_K_DIM,
        value_dim: h * dv,
        suffix_len: q,
    };
    if chunked {
        kernel.encode(args, &mut encoding).expect("chunked encode");
    } else {
        let prep = <<Vulkan as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            &fixture.context,
            T::data_type(),
            DataType::F32,
            HEAD_K_DIM,
            false,
            false,
        )
        .expect("recurrent prep");
        let prefill = <<Vulkan as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
            &fixture.context,
            T::data_type(),
            HEAD_K_DIM,
        )
        .expect("recurrent prefill");
        let [mut q_norm, mut k_norm] =
            [0; 2].map(|_| encoding.allocate_scratch_for_shape(&[q, g * HEAD_K_DIM], DataType::F32).unwrap());
        let [mut beta, mut decay] =
            [0; 2].map(|_| encoding.allocate_scratch_for_shape(&[q, h], DataType::F32).unwrap());
        DeltaNetPrefillPrepKernel::encode(
            &prep,
            args.in_projected,
            args.a_log,
            args.dt_bias,
            &mut q_norm,
            &mut k_norm,
            None::<&mut <Vulkan as Backend>::ScratchBuffer>,
            &mut beta,
            &mut decay,
            h,
            g,
            g * HEAD_K_DIM,
            h * dv,
            q,
            &mut encoding,
        );
        DeltaNetPrefillKernel::encode(
            &prefill,
            &q_norm,
            &k_norm,
            &beta,
            &decay,
            args.in_projected,
            args.ssm_state,
            args.delta_output,
            h,
            g,
            dv,
            g * HEAD_K_DIM,
            h * dv,
            q,
            dv.div_ceil(16),
            &mut encoding,
        );
    }
    let completed = KernelFixture::complete(encoding);
    let time = completed.gpu_execution_time();
    // SAFETY: the only command buffer using these pages has completed.
    let (read, out) =
        unsafe { (read_scratch::<T>(&projection, in_proj.len(), label), read_scratch::<T>(&out, out_len, label)) };
    KernelFixture::assert_bits(in_proj, &read, &format!("{label}: in_proj"));
    KernelFixture::assert_bits(a_log, &buffer_to_vec::<Vulkan, f32>(&a_log_buffer), &format!("{label}: a_log"));
    KernelFixture::assert_bits(dt_bias, &buffer_to_vec::<Vulkan, f32>(&dt_bias_buffer), &format!("{label}: dt_bias"));
    (out, buffer_to_vec::<Vulkan, f32>(&ssm_state), time)
}

/// Every element within max(absolute, relative max(|e|, |a|)) of the CPU; NaN never is.
fn assert_close(
    expected: &[f32],
    actual: &[f32],
    (absolute, relative): (f32, f32),
    label: &str,
) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    let (mut max, mut outside) = (0f32, 0);
    for (&e, &a) in expected.iter().zip(actual) {
        let error = (a - e).abs();
        max = max.max(error);
        outside += usize::from(!error.is_finite() || error > absolute.max(relative * e.abs().max(a.abs())));
    }
    eprintln!("{label}: max absolute error {max:.3e}, {outside} outside");
    assert_eq!(outside, 0, "{label}: elements outside the tolerance");
}

fn case<T: ArrayElement + Pod + Float + Debug>(
    fixture: &KernelFixture,
    shape: [u32; 4],
    zero: bool,
) {
    let label = format!("{:?} {shape:?} zero {zero}", T::data_type());
    let (projection, f32s) = inputs(shape, zero);
    let in_proj = projection.iter().map(|&x| T::from(x).unwrap()).collect::<Vec<_>>();
    let (cpu_out, cpu_state) = cpu_reference(shape, &in_proj, &f32s);
    let (gpu_out, gpu_state, _) = vulkan(fixture, shape, &in_proj, &f32s, &label, true);
    let [cpu_out, gpu_out] = [cpu_out, gpu_out].map(|v| v.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<_>>());
    match zero {
        true => {
            let all_zero = [&cpu_out, &gpu_out, &cpu_state, &gpu_state].iter().all(|v| v.iter().all(|&x| x == 0.0));
            assert!(all_zero, "{label}: nonzero result from zero inputs and state");
        },
        // The tolerance's premise: below 1 one BF16 step is at most 2^-8, within the absolute bound.
        false => {
            assert!(cpu_out.iter().all(|x| x.abs() < 1.0), "{label}: CPU output reaches 1");
            assert_ne!(cpu_state, f32s[2], "{label}: CPU state unchanged");
        },
    }
    assert_close(&cpu_out, &gpu_out, OUTPUT_TOLERANCE, &format!("{label} delta_output"));
    assert_close(&cpu_state, &gpu_state, STATE_TOLERANCE, &format!("{label} state"));
}

/// Every shape in F32 and BF16 storage from a nonzero state: the Vulkan chunked prefill's delta_output and state within
/// the Metal chunked tolerances of the engine's sequential CPU Prep and Prefill, read-only inputs and slack unchanged.
#[uzu_test]
fn chunked_prefill_matches_cpu_recurrence() {
    let fixture = KernelFixture::new();
    for shape in SHAPES {
        case::<f32>(&fixture, shape, false);
        case::<bf16>(&fixture, shape, false);
    }
    fixture.assert_clean();
}

/// All-zero in_proj and state over a tail chunk: exactly zero output and state on the CPU and Vulkan, so zero q and k
/// rows normalize to zero rather than NaN.
#[uzu_test]
fn chunked_prefill_zero_inputs() {
    let fixture = KernelFixture::new();
    case::<f32>(&fixture, SHAPES[1], true);
    case::<bf16>(&fixture, SHAPES[1], true);
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for shape in [[4, 2, 32, 129], [4, 2, 64, 256], [16, 8, 128, 512], [16, 8, 128, 1024]] {
        let (projection, params) = inputs(shape, false);
        let projection = projection.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
        let (expected_output, expected_state) = cpu_reference(shape, &projection, &params);
        let expected_output = expected_output.into_iter().map(bf16::to_f32).collect::<Vec<_>>();
        let mut medians = [Duration::ZERO; 2];
        for (index, chunked) in [false, true].into_iter().enumerate() {
            let mut times = Vec::new();
            for sample in 0..15 {
                let (output, state, time) = vulkan(&fixture, shape, &projection, &params, "timing", chunked);
                if sample == 0 {
                    let output = output.into_iter().map(bf16::to_f32).collect::<Vec<_>>();
                    assert_close(&expected_output, &output, OUTPUT_TOLERANCE, "timing output");
                    assert_close(&expected_state, &state, STATE_TOLERANCE, "timing state");
                }
                if sample >= 2 {
                    times.push(time);
                }
            }
            times.sort_unstable();
            medians[index] = times[times.len() / 2];
        }
        eprintln!(
            "BF16 {shape:?}: recurrent {:?}, chunked {:?}, speedup {:.3}",
            medians[0],
            medians[1],
            medians[0].as_secs_f64() / medians[1].as_secs_f64()
        );
    }
    fixture.assert_clean();
}
