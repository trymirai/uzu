//! What the Neural Engine executors of dense MLPs ([`super::ffn`]) and of single linear layers
//! ([`super::linear`]) have in common: the GPU staging, pack and unpack path around an ANE
//! program, the hand-off of the timeline to the ANE, and the health of the engine.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use half::f16;
use metal::{MTLSharedEventExt, MTLSharedEventListener, MTLSharedEventNotificationBlock};
use objc2::rc::Retained;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
            gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
            kernel::matmul::{MatmulB, QuantizedCorrection},
        },
        metal::{
            Metal,
            ane::{
                AneBinding, AneElement, AneError, AneProgram, AneSurface,
                trace::{self, Mark},
            },
            command_buffer::{Handoffs, MetalCommandBufferEncoding},
            context::{MetalContext, MetalLane},
            error::MetalError,
            kernel::{
                AnePackMetalKernel, AneStageUpRowsMetalKernel, AneTransposeCodesMetalKernel, AneUnpackMetalKernel,
            },
        },
    },
    encodable_block::{linear::LinearWeights, weight_matrix::WeightMatrix},
};

pub type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;

/// The width of the activations the GPU kernels pack and the ANE programs read (the GPU kernels hold
/// a token, and a staged row, as 4096-value chunks in registers).
pub const CHUNK: u32 = 4096;
/// Smaller batches stay on the GPU: the ANE's fixed cost per call dominates them.
pub const MIN_TOKENS: u32 = 128;
/// The row counts the programs are compiled for.
pub const ROWS: [u32; 4] = [256, 512, 768, 1024];
/// Width of each matmul's reduction segment in the programs.
pub const SEGMENT: u32 = 2048;
const SIGN_SEED: u64 = 0x5eed_a11e_2026_1103;

/// The error a prefill fails with when an ANE request of it failed: its output would be wrong.
/// The engine has retired by then, so running the prefill again keeps every call on the GPU.
const FAILED: &str = "ANE request failed during prefill (the Neural Engine is retired; retry runs on the GPU)";

/// Test-only failure injection, `UZU_ANE_FAULT=enqueue:K` or `completion:K` (K counts from 1): the
/// K-th ANE request fails to enqueue, or enqueues and reports failure.
#[derive(Clone, Copy, PartialEq)]
enum FaultKind {
    Enqueue,
    Completion,
}

struct Fault {
    kind: FaultKind,
    at: usize,
}

impl Fault {
    fn from_env() -> Option<Self> {
        let value = std::env::var("UZU_ANE_FAULT").ok()?;
        let (kind, at) = value.split_once(':')?;
        let kind = match kind {
            "enqueue" => FaultKind::Enqueue,
            "completion" => FaultKind::Completion,
            _ => return None,
        };
        Some(Self {
            kind,
            at: at.parse().ok()?,
        })
    }
}

/// Whether the Neural Engine still runs calls: every executor of a device shares it, and the
/// first failed request retires all of them.
pub struct AneHealth {
    healthy: AtomicBool,
    fault: Option<Fault>,
    requests: AtomicUsize,
    /// Listens for the ANE's `done` signals when tracing (`UZU_ANE_TRACE`).
    trace_listener: Option<Retained<MTLSharedEventListener>>,
}

impl Default for AneHealth {
    fn default() -> Self {
        Self {
            healthy: AtomicBool::new(true),
            fault: Fault::from_env(),
            requests: AtomicUsize::new(0),
            trace_listener: trace::enabled().then(MTLSharedEventListener::new),
        }
    }
}

impl AneHealth {
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn retire(&self) {
        self.healthy.store(false, Ordering::Release);
    }

    /// Counts a request, returning the fault injected into it, if any.
    fn begin_request(&self) -> Option<FaultKind> {
        let request = self.requests.fetch_add(1, Ordering::AcqRel) + 1;
        self.fault.as_ref().filter(|fault| fault.at == request).map(|fault| fault.kind)
    }
}

/// Deterministic ±1 signs.
pub fn random_signs(
    count: usize,
    salt: u64,
) -> Vec<f32> {
    let mut state = SIGN_SEED ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// `out[c][i] = sign[c] * H_block[c % block][i]` (orthonormal), the grouped-conv form of `D H_block`.
pub fn signed_hadamard_blocks(
    signs: &[f32],
    block: u32,
) -> Vec<f16> {
    let block = block as usize;
    let norm = 1.0 / (block as f32).sqrt();
    let mut out = Vec::with_capacity(signs.len() * block);
    for (channel, sign) in signs.iter().enumerate() {
        let row = channel % block;
        for column in 0..block {
            let hadamard = if (row & column).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            out.push(f16::from_f32(sign * hadamard * norm));
        }
    }
    out
}

pub fn read_signs(buffer: &GpuBuffer) -> Vec<f32> {
    buffer.as_slice::<i32>().iter().map(|sign| *sign as f32).collect()
}

pub fn upload(
    context: &MetalContext,
    values: &[f32],
) -> Result<GpuBuffer, MetalError> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(values))?;
    buffer.as_slice_mut::<f32>().copy_from_slice(values);
    Ok(buffer.handoff())
}

/// Uploads a +-1 sign vector as a bit mask (bit `i % 16` of u16 `i / 16` set where `signs[i]` is negative),
/// the form `AneStageDownRows` reads: 32x less cache traffic per staged row than the float signs.
pub fn upload_sign_bits(
    context: &MetalContext,
    signs: &[f32],
) -> Result<GpuBuffer, MetalError> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(signs) / 32 * 2)?;
    for (word, group) in buffer.as_slice_mut::<u16>().iter_mut().zip(signs.chunks(16)) {
        *word = group.iter().enumerate().fold(0u16, |bits, (index, sign)| bits | (u16::from(*sign < 0.0) << index));
    }
    Ok(buffer.handoff())
}

/// Whether a projection is mirai W4 (asymmetric, group 32) in the plain row-major layout the
/// executors stage from, with input and output RHT signs and no biases.
pub fn stageable(weights: &LinearWeights<'_, Metal>) -> bool {
    let (Some(info), Some(_), Some(_)) =
        (weights.matrix.quantization(), weights.input_rht_signs, weights.output_rht_signs)
    else {
        return false;
    };
    let MatmulB::Quantized(quantized) = weights.matrix.matmul_b() else {
        return false;
    };
    let zero_points = matches!(quantized.correction, QuantizedCorrection::ZeroPoints(_));
    info.mode == QuantizationMode::U4
        && info.method == QuantizationMethod::ScaleZeroPoint
        && info.group_size == 32
        && zero_points
        && !quantized.signed_codes
        && quantized.weights_layout == Default::default()
        && !weights.has_biases
}

/// Elements between consecutive groups of a W4 matrix's scales (its padded output dimension).
pub fn scale_stride(matrix: &WeightMatrix<Metal>) -> u32 {
    let MatmulB::Quantized(quantized) = matrix.matmul_b() else {
        unreachable!("staged weights are quantized")
    };
    quantized.params.scale_shape()[1]
}

/// The GPU kernels around an ANE program: activation packing, row staging and output unpacking.
pub struct AneKernels {
    activations: AneElement,
    pack: AnePackMetalKernel,
    transpose: AneTransposeCodesMetalKernel,
    unpack: AneUnpackMetalKernel,
    stage_up: AneStageUpRowsMetalKernel,
}

impl AneKernels {
    /// `scale_row`: whether the program's output carries a per-token scale in the row after its
    /// channels (the FFN's intermediate scale), which unpacking applies. `activations`: the element
    /// type of the program's input `x`.
    pub fn new(
        context: &MetalContext,
        scale_row: bool,
        activations: AneElement,
    ) -> Result<Self, AneError> {
        let half_values = activations == AneElement::Float16;
        let kernel_error = |error: MetalError| AneError::Unavailable(error.to_string());
        Ok(Self {
            pack: AnePackMetalKernel::new(context, half_values).map_err(kernel_error)?,
            transpose: AneTransposeCodesMetalKernel::new(context, half_values).map_err(kernel_error)?,
            activations,
            unpack: AneUnpackMetalKernel::new(context, scale_row).map_err(kernel_error)?,
            stage_up: AneStageUpRowsMetalKernel::new(context).map_err(kernel_error)?,
        })
    }

    /// Packs `tokens` rows of `input` (token-major bf16 `[tokens, CHUNK]`) as `a = H(signs * x)`
    /// into the program's input `x` (int8, channel-major) and its token scales `tx`.
    pub fn pack(
        &self,
        input: &<Metal as Backend>::ScratchBuffer,
        signs: &GpuBuffer,
        tokens: u32,
        x: &mut AneSurface,
        tx: &mut AneSurface,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let mut codes = command_buffer.allocate_scratch((tokens * CHUNK) as usize * self.activations.bytes())?;
        self.pack.encode(input, signs, &mut codes, tx.buffer_mut(), CHUNK, tokens, command_buffer);
        let x_row_stride = x.row_stride_elements() as u32;
        self.transpose.encode(&codes, x.buffer_mut(), CHUNK, tokens, x_row_stride, command_buffer);
        Ok(())
    }

    /// Stages `rows` rows of a W4 projection, from `row_offset`, into the int8 weight rows `staged`
    /// and their scales `scales` (see `AneStageUpRows`): row `r' = H4096(pre_signs * H32(r))`.
    pub fn stage_rows(
        &self,
        matrix: &WeightMatrix<Metal>,
        pre_signs: &GpuBuffer,
        staged: &mut AneSurface,
        scales: &mut AneSurface,
        row_offset: u32,
        rows: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) {
        let staged_row_stride = staged.row_stride_bytes() as u32;
        let row_scale_stride = scales.row_stride_elements() as u32;
        self.stage_up.encode(
            matrix.values(),
            matrix.scales().unwrap(),
            matrix.zero_points().unwrap(),
            pre_signs,
            staged.buffer_mut(),
            scales.buffer_mut(),
            row_offset,
            rows,
            scale_stride(matrix),
            staged_row_stride,
            row_scale_stride,
            command_buffer,
        );
    }

    /// Unpacks the program's output `y` (`channels` rows, channel-major fp16) of `tokens` tokens
    /// into token-major bf16 `[tokens, channels]`: scales it by the token scales `tx`, and applies
    /// the projection's output rotation `signs * H32`.
    pub fn unpack(
        &self,
        y: &AneSurface,
        tx: &AneSurface,
        signs: &GpuBuffer,
        channels: u32,
        tokens: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let mut output = command_buffer.allocate_scratch((tokens * channels * 2) as usize)?;
        self.unpack.encode(
            y.buffer(),
            tx.buffer(),
            signs,
            &mut output,
            channels,
            tokens,
            y.row_stride_elements() as u32,
            command_buffer,
        );
        Ok(output)
    }
}

/// An ANE program with its bindings and event listeners, run by [`submit`].
pub trait AneExecutor: Send + Sync + 'static {
    fn health(&self) -> &AneHealth;

    fn program(&self) -> &AneProgram;

    /// The binding of the function of `rows` rows on a lane's surfaces.
    fn binding(
        &self,
        lane: usize,
        rows: u32,
    ) -> &AneBinding;

    /// Queues the lane's requests once the GPU work before them is done, so requests reach the
    /// ANE (which serves them in order) in the order they become ready, not the order they were
    /// encoded in.
    fn listener(
        &self,
        lane: usize,
    ) -> &MTLSharedEventListener;
}

/// Hands the timeline to the ANE: everything encoded since the last hand-off, `order`'s mixer
/// segment included, runs after the previous prefill chunk's segment of the same `order` (a number
/// unique to the call site, see [`MetalCommandBufferEncoding::order_after_previous_chunk`]); the
/// next chunk's will run after ours. The ANE runs the function of `rows` rows of `executor` and
/// the GPU continues when it is done.
pub fn submit<E: AneExecutor>(
    executor: &Arc<E>,
    order: usize,
    rows: u32,
    command_buffer: &mut MetalCommandBufferEncoding,
) -> Result<(), MetalError> {
    let lane = command_buffer.lane();
    let timeline = command_buffer.context().lane(lane)?;
    let ready = timeline.reserve_values(2);
    let done = ready + 1;
    command_buffer.order_after_previous_chunk(order);
    command_buffer.record_layer_ready(order, ready);
    let failures = command_buffer.failure_sink();
    trace::begin(lane, command_buffer.chunk(), order, rows, ready);
    enqueue_when_ready(executor, lane, rows, &timeline, ready, done, failures);
    command_buffer.handoff_to_ane(&timeline, ready, done)
}

/// Queues the ANE request of a `rows`-row function on `lane` once the GPU signals `ready` (which
/// it does after the work before the call). The request signals `done`.
fn enqueue_when_ready<E: AneExecutor>(
    executor: &Arc<E>,
    lane: usize,
    rows: u32,
    timeline: &Arc<MetalLane>,
    ready: u64,
    done: u64,
    failures: Arc<Handoffs>,
) {
    let target = executor.clone();
    let block_timeline = timeline.clone();
    let block = MTLSharedEventNotificationBlock::new(move |_event, _value| {
        trace::mark(lane, ready, Mark::Notified);
        enqueue(&target, lane, rows, &block_timeline, ready, done, &failures);
    });
    timeline.event.notify_listener_at_value(executor.listener(lane), ready, &block);
    if let Some(listener) = &executor.health().trace_listener {
        let block = MTLSharedEventNotificationBlock::new(move |_event, _value| {
            trace::mark(lane, ready, Mark::Signalled);
        });
        timeline.event.notify_listener_at_value(listener, done, &block);
    }
}

/// Tells the lane's queue the request that became ready at `ready` is done.
fn relay_done(
    timeline: &MetalLane,
    lane: usize,
    ready: u64,
    done: u64,
) {
    trace::mark(lane, ready, Mark::RelayStarted);
    timeline.relay_done(done);
    trace::mark(lane, ready, Mark::RelayReturned);
}

fn enqueue<E: AneExecutor>(
    executor: &Arc<E>,
    lane: usize,
    rows: u32,
    timeline: &Arc<MetalLane>,
    ready: u64,
    done: u64,
    failures: &Arc<Handoffs>,
) {
    let health = executor.health();
    if !health.is_healthy() {
        // Retired while this call was already encoded: its result is lost either way, so do not
        // keep sending work to a failing ANE. The GPU waits for `done` and must not hang.
        failures.fail(FAILED);
        relay_done(timeline, lane, ready, done);
        return;
    }
    let fault = health.begin_request();
    let completion_executor = executor.clone();
    let completion_timeline = timeline.clone();
    let completion_failures = failures.clone();
    let completion = Box::new(move |success: bool| {
        trace::mark(lane, ready, Mark::Completed);
        if !success || fault == Some(FaultKind::Completion) {
            // The result is garbage: the engine retires and the prefill fails.
            completion_executor.health().retire();
            completion_failures.fail(FAILED);
        }
        // On failure too: the GPU waits for `done` and must not hang.
        relay_done(&completion_timeline, lane, ready, done);
    });
    trace::mark(lane, ready, Mark::EnqueueStarted);
    let enqueued = if fault == Some(FaultKind::Enqueue) {
        Err(AneError::Unavailable("injected enqueue fault".to_string()))
    } else {
        executor.program().enqueue(executor.binding(lane, rows), &timeline.event, ready, done, completion)
    };
    trace::mark(lane, ready, Mark::EnqueueReturned);
    if let Err(error) = enqueued {
        health.retire();
        failures.fail(FAILED);
        relay_done(timeline, lane, ready, done);
        eprintln!("uzu: ANE enqueue failed, everything runs on the GPU from now on: {error}");
    }
}
