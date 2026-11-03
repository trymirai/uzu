//! The Neural Engine FFN executor: one W8A8 program shared by every dense MLP of a shape.
//!
//! Per MLP call (prefill batches only) the GPU
//! 1. stages the layer's W4 weights into int8 rows the ANE reads (`AneStageRows`), folding the
//!    projections' input rotations and the ANE-side rotations into the rows,
//! 2. packs the activations: `a = H_hidden(D_a * x)`, int8 per token, channel-major (`AnePack`,
//!    `AneTransposeCodes`),
//! 3. hands the timeline to the ANE (`handoff`; the request is queued to the ANE once the GPU
//!    work before it completes), which runs gate/up, the up output rotation,
//!    SiLU * up, the intermediate rotation `D_m H_512`, per-token requantization and down,
//! 4. unpacks the result with the down projection's output rotation (`AneUnpack`).
//!
//! Algebra (rows `r` of the dequantized mirai weights, which hold `R_out W R_in^T`):
//! up rows `r' = H_hidden(D_a D_in H32(r))`, down rows `r' = D_m H_512(D_din H32(r))`.
//! See scripts/ane_prefill/numerics/pipeline_v1.py.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use half::f16;
use metal::{MTLSharedEventExt, MTLSharedEventListener, MTLSharedEventNotificationBlock};
use objc2::rc::Retained;
use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
            gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
            kernel::matmul::{MatmulB, QuantizedCorrection},
        },
        metal::{
            Metal,
            ane::{AneBinding, AneElement, AneError, AneProgram, AneRuntime, AneSurface, mil::FfnShape},
            command_buffer::{Handoffs, MetalCommandBufferEncoding},
            context::{MetalContext, MetalLane},
            error::MetalError,
            kernel::{AnePackMetalKernel, AneStageRowsMetalKernel, AneTransposeCodesMetalKernel, AneUnpackMetalKernel},
        },
    },
    encodable_block::{linear::LinearWeights, mlp::DenseMlpWeights, weight_matrix::WeightMatrix},
};

type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;

const ROWS: [u32; 4] = [256, 512, 768, 1024];
/// Smaller batches stay on the GPU: the ANE's fixed cost per call dominates them.
pub const MIN_TOKENS: u32 = 128;
const SEGMENT: u32 = 2048;
const CHUNK: u32 = 4096;
const INTERMEDIATE_ROTATION_BLOCK: u32 = 512;
const UP_ROTATION_BLOCK: u32 = 32;
const SIGN_SEED: u64 = 0x5eed_a11e_2026_1103;

/// The error a prefill fails with when an ANE request of it failed: its output would be wrong.
/// The executor has retired by then, so running the prefill again keeps every MLP on the GPU.
const FAILED: &str = "ANE request failed during prefill (the Neural Engine is retired; retry runs on the GPU)";

/// Test-only failure injection, `UZU_ANE_FAULT=enqueue:K` or `completion:K` (K counts from 1): the
/// K-th ANE request of the executor fails to enqueue, or enqueues and reports failure.
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

/// Deterministic ±1 signs.
fn random_signs(
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
fn signed_hadamard_blocks(
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

fn read_signs(buffer: &GpuBuffer) -> Vec<f32> {
    buffer.as_slice::<i32>().iter().map(|sign| *sign as f32).collect()
}

fn upload(
    context: &MetalContext,
    values: &[f32],
) -> Result<GpuBuffer, MetalError> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(values))?;
    buffer.as_slice_mut::<f32>().copy_from_slice(values);
    Ok(buffer.handoff())
}

/// Whether a projection is mirai W4 (asymmetric, group 32) in the plain row-major layout this
/// executor stages from, with input and output RHT signs and no biases.
fn stageable(weights: &LinearWeights<'_, Metal>) -> bool {
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
fn scale_stride(matrix: &WeightMatrix<Metal>) -> u32 {
    let MatmulB::Quantized(quantized) = matrix.matmul_b() else {
        unreachable!("staged weights are quantized")
    };
    quantized.params.scale_shape()[1]
}

/// Per-layer constants of an ANE-run MLP.
pub struct AneLayer {
    /// Position among the model's ANE-run layers: orders the layer across pipelined prefill chunks.
    index: usize,
    up_pre_signs: GpuBuffer,
    down_pre_signs: GpuBuffer,
    down_output_signs: GpuBuffer,
}

impl AneLayer {
    /// Keeps the layer's mixer (in the segment being encoded) after the previous prefill chunk's.
    /// For batches the ANE does not run; [`AneFfn::encode`] does it for the rest.
    pub fn order(
        &self,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) {
        command_buffer.order_after_previous_chunk(self.index);
    }
}

struct Surfaces {
    x: AneSurface,
    tx: AneSurface,
    wg: AneSurface,
    wu: AneSurface,
    sg: AneSurface,
    su: AneSurface,
    wd: AneSurface,
    sd: AneSurface,
    y: AneSurface,
}

struct Kernels {
    pack: AnePackMetalKernel,
    transpose: AneTransposeCodesMetalKernel,
    unpack: AneUnpackMetalKernel,
    stage_up: AneStageRowsMetalKernel,
    stage_down: AneStageRowsMetalKernel,
}

/// What one lane (see [`MetalLane`]) of the pipelined prefill needs to run its chunk's MLPs: chunks
/// on different lanes overlap, so they cannot share surfaces.
struct LaneResources {
    surfaces: Mutex<Surfaces>,
    bindings: Vec<(u32, AneBinding)>,
    /// Queues the ANE request of an MLP once the GPU work before it is done, so requests reach
    /// the ANE (which serves them in order) in the order they become ready, not the order they
    /// were encoded in.
    listener: Retained<MTLSharedEventListener>,
}

impl LaneResources {
    fn new(
        context: &MetalContext,
        shape: &FfnShape,
        program: &AneProgram,
    ) -> Result<Self, AneError> {
        let surface = |name: &str| -> Result<AneSurface, AneError> {
            let spec = shape
                .inputs()
                .into_iter()
                .chain(std::iter::once(shape.output()))
                .find(|spec| spec.name == name)
                .ok_or_else(|| AneError::MissingInput(name.to_string()))?;
            AneSurface::new(context, spec.rows, spec.width, spec.element)
        };
        let surfaces = Surfaces {
            x: surface("x")?,
            tx: surface("tx")?,
            wg: surface("wg")?,
            wu: surface("wu")?,
            sg: surface("sg")?,
            su: surface("su")?,
            wd: surface("wd")?,
            sd: surface("sd")?,
            y: surface("y")?,
        };
        debug_assert_eq!(surfaces.y.element(), AneElement::Float16);

        let mut bindings = Vec::new();
        for rows in &shape.rows {
            let procedure = program
                .procedure(&FfnShape::function_name(*rows))
                .ok_or_else(|| AneError::UnknownProcedure(FfnShape::function_name(*rows)))?;
            let inputs = [
                ("x", &surfaces.x),
                ("tx", &surfaces.tx),
                ("wg", &surfaces.wg),
                ("wu", &surfaces.wu),
                ("sg", &surfaces.sg),
                ("su", &surfaces.su),
                ("wd", &surfaces.wd),
                ("sd", &surfaces.sd),
            ];
            bindings.push((*rows, program.bind(procedure, &inputs, &surfaces.y)?));
        }
        Ok(Self {
            surfaces: Mutex::new(surfaces),
            bindings,
            listener: MTLSharedEventListener::new(),
        })
    }
}

pub struct AneFfn {
    hidden: u32,
    intermediate: u32,
    shape: FfnShape,
    program: AneProgram,
    lanes: [LaneResources; 2],
    kernels: Kernels,
    pack_signs: Vec<f32>,
    pack_signs_buffer: GpuBuffer,
    intermediate_signs_buffer: GpuBuffer,
    up_output_signs: Vec<f32>,
    healthy: AtomicBool,
    fault: Option<Fault>,
    requests: AtomicUsize,
}

impl AneFfn {
    /// Builds the executor for the shape of `weights` (compiling the program, ~4 s the first time
    /// per Mac, cached afterwards). `None` if the weights are not in a form it runs.
    pub fn new(
        context: &MetalContext,
        weights: &DenseMlpWeights<'_, Metal>,
    ) -> Result<Option<Self>, AneError> {
        if !stageable(&weights.up) || !stageable(&weights.down) {
            return Ok(None);
        }
        let hidden = read_signs(weights.up.input_rht_signs.unwrap()).len() as u32;
        let intermediate = read_signs(weights.down.input_rht_signs.unwrap()).len() as u32;
        // The GPU kernels hold a token (and a staged row) as 4096-value chunks in registers.
        if hidden != CHUNK || !intermediate.is_multiple_of(CHUNK) || intermediate > 3 * CHUNK {
            return Ok(None);
        }
        let up_output_signs = read_signs(weights.up.output_rht_signs.unwrap());
        let intermediate_signs = random_signs(intermediate as usize, 2);
        let shape = FfnShape {
            hidden,
            intermediate,
            segment: SEGMENT,
            rows: ROWS.to_vec(),
            max_rows: *ROWS.last().unwrap(),
            rotation_block: INTERMEDIATE_ROTATION_BLOCK,
            rotation: signed_hadamard_blocks(&intermediate_signs, INTERMEDIATE_ROTATION_BLOCK),
            up_rotation_block: UP_ROTATION_BLOCK,
            value_rotation: signed_hadamard_blocks(&up_output_signs[..intermediate as usize], UP_ROTATION_BLOCK),
            gate_rotation: signed_hadamard_blocks(&up_output_signs[intermediate as usize..], UP_ROTATION_BLOCK),
            single_weight_surfaces: true,
            single_input_surface: true,
        };

        let runtime = AneRuntime::new()?;
        let (mil, blob) = super::mil::ffn_program(&shape);
        let program = AneProgram::new(runtime, &mil, &blob)?;

        let lanes = [LaneResources::new(context, &shape, &program)?, LaneResources::new(context, &shape, &program)?];

        let kernel_error = |error: MetalError| AneError::Unavailable(error.to_string());
        let kernels = Kernels {
            pack: AnePackMetalKernel::new(context).map_err(kernel_error)?,
            transpose: AneTransposeCodesMetalKernel::new(context).map_err(kernel_error)?,
            unpack: AneUnpackMetalKernel::new(context).map_err(kernel_error)?,
            stage_up: AneStageRowsMetalKernel::new(context, hidden, hidden / CHUNK, false).map_err(kernel_error)?,
            stage_down: AneStageRowsMetalKernel::new(context, INTERMEDIATE_ROTATION_BLOCK, intermediate / CHUNK, true)
                .map_err(kernel_error)?,
        };
        let pack_signs = random_signs(hidden as usize, 1);

        Ok(Some(Self {
            hidden,
            intermediate,
            shape,
            program,
            lanes,
            kernels,
            pack_signs_buffer: upload(context, &pack_signs).map_err(kernel_error)?,
            pack_signs,
            intermediate_signs_buffer: upload(context, &intermediate_signs).map_err(kernel_error)?,
            up_output_signs,
            healthy: AtomicBool::new(true),
            fault: Fault::from_env(),
            requests: AtomicUsize::new(0),
        }))
    }

    /// The per-layer constants for an MLP this executor can run, or `None` if it cannot (other
    /// shape, other format, or signs the shared program was not built for).
    pub fn layer(
        &self,
        context: &MetalContext,
        weights: &DenseMlpWeights<'_, Metal>,
    ) -> Result<Option<AneLayer>, MetalError> {
        if !stageable(&weights.up) || !stageable(&weights.down) {
            return Ok(None);
        }
        let up_input_signs = read_signs(weights.up.input_rht_signs.unwrap());
        let down_input_signs = read_signs(weights.down.input_rht_signs.unwrap());
        if up_input_signs.len() != self.hidden as usize
            || down_input_signs.len() != self.intermediate as usize
            || read_signs(weights.up.output_rht_signs.unwrap()) != self.up_output_signs
        {
            return Ok(None);
        }
        let up_pre_signs: Vec<f32> =
            up_input_signs.iter().zip(&self.pack_signs).map(|(input, pack)| input * pack).collect();
        Ok(Some(AneLayer {
            index: context.next_ane_layer_index(),
            up_pre_signs: upload(context, &up_pre_signs)?,
            down_pre_signs: upload(context, &down_input_signs)?,
            down_output_signs: upload(context, &read_signs(weights.down.output_rht_signs.unwrap()))?,
        }))
    }

    /// Whether this executor was built for MLPs of `weights`' shape.
    pub fn matches(
        &self,
        weights: &DenseMlpWeights<'_, Metal>,
    ) -> bool {
        let (Some(up), Some(down)) = (weights.up.input_rht_signs, weights.down.input_rht_signs) else {
            return false;
        };
        up.as_slice::<i32>().len() == self.hidden as usize && down.as_slice::<i32>().len() == self.intermediate as usize
    }

    /// Whether the executor still runs MLPs (it retires after the first ANE failure).
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// Whether a batch of `tokens` runs on the ANE.
    pub fn accepts(
        &self,
        tokens: u32,
    ) -> bool {
        tokens >= MIN_TOKENS && self.shape.rows_for(tokens).is_some() && self.is_healthy()
    }

    /// Encodes the MLP of `tokens` rows of `input` (token-major bf16 `[tokens, hidden]`), returning
    /// its output in the same layout.
    pub fn encode(
        self: &Arc<Self>,
        layer: &AneLayer,
        weights: &DenseMlpWeights<'_, Metal>,
        input: &<Metal as Backend>::ScratchBuffer,
        tokens: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (ane)");
        let (hidden, intermediate) = (self.hidden, self.intermediate);
        let rows = self.shape.rows_for(tokens).expect("batch accepted by the ANE executor");
        let lane = command_buffer.lane();
        let mut surfaces = self.lanes[lane].surfaces.lock();
        let Surfaces {
            x,
            tx,
            wg,
            wu,
            sg,
            su,
            wd,
            sd,
            y,
        } = &mut *surfaces;

        let up = weights.up.matrix;
        let down = weights.down.matrix;
        let up_scale_stride = scale_stride(up);
        let down_scale_stride = scale_stride(down);
        for (staged, scales, row_offset) in [(&mut *wu, &mut *su, 0), (&mut *wg, &mut *sg, intermediate)] {
            let staged_row_stride = staged.row_stride_bytes() as u32;
            let scale_stride = scales.row_stride_elements() as u32;
            self.kernels.stage_up.encode(
                up.values(),
                up.scales().unwrap(),
                up.zero_points().unwrap(),
                &layer.up_pre_signs,
                None::<&GpuBuffer>,
                staged.buffer_mut(),
                scales.buffer_mut(),
                row_offset,
                intermediate,
                up_scale_stride,
                staged_row_stride,
                scale_stride,
                command_buffer,
            );
        }
        let (wd_row_stride, sd_stride) = (wd.row_stride_bytes() as u32, sd.row_stride_elements() as u32);
        self.kernels.stage_down.encode(
            down.values(),
            down.scales().unwrap(),
            down.zero_points().unwrap(),
            &layer.down_pre_signs,
            Some(&self.intermediate_signs_buffer),
            wd.buffer_mut(),
            sd.buffer_mut(),
            0,
            hidden,
            down_scale_stride,
            wd_row_stride,
            sd_stride,
            command_buffer,
        );

        let mut codes = command_buffer.allocate_scratch((tokens * hidden) as usize)?;
        self.kernels.pack.encode(
            input,
            &self.pack_signs_buffer,
            &mut codes,
            tx.buffer_mut(),
            hidden,
            tokens,
            command_buffer,
        );
        let x_row_stride = x.row_stride_bytes() as u32;
        self.kernels.transpose.encode(&codes, x.buffer_mut(), hidden, tokens, x_row_stride, command_buffer);
        drop(codes);

        // Everything encoded since the last handoff, this layer's mixer included, runs after the
        // previous chunk's mixer of the layer; the next chunk's will run after ours.
        let timeline = command_buffer.context().lane(lane)?;
        let ready = timeline.reserve_values(2);
        let done = ready + 1;
        command_buffer.order_after_previous_chunk(layer.index);
        command_buffer.record_layer_ready(layer.index, ready);
        let failures = command_buffer.failure_sink();
        self.enqueue_when_ready(lane, rows, &timeline, ready, done, failures);
        command_buffer.handoff_to_ane(&timeline, ready, done)?;

        let mut output = command_buffer.allocate_scratch((tokens * hidden * 2) as usize)?;
        self.kernels.unpack.encode(
            y.buffer(),
            tx.buffer(),
            &layer.down_output_signs,
            &mut output,
            hidden,
            tokens,
            y.row_stride_elements() as u32,
            command_buffer,
        );
        Ok(output)
    }

    /// Queues the ANE request for the MLP of a `rows`-row procedure on `lane` once the GPU signals
    /// `ready` (which it does after the work before the MLP). The request signals `done`.
    fn enqueue_when_ready(
        self: &Arc<Self>,
        lane: usize,
        rows: u32,
        timeline: &Arc<MetalLane>,
        ready: u64,
        done: u64,
        failures: Arc<Handoffs>,
    ) {
        let ffn = self.clone();
        let block_timeline = timeline.clone();
        let block = MTLSharedEventNotificationBlock::new(move |_event, _value| {
            ffn.enqueue(lane, rows, &block_timeline, ready, done, &failures);
        });
        timeline.event.notify_listener_at_value(&self.lanes[lane].listener, ready, &block);
    }

    fn enqueue(
        self: &Arc<Self>,
        lane: usize,
        rows: u32,
        timeline: &Arc<MetalLane>,
        ready: u64,
        done: u64,
        failures: &Arc<Handoffs>,
    ) {
        if !self.is_healthy() {
            // Retired while this MLP was already encoded: its result is lost either way, so do not
            // keep sending work to a failing ANE. The GPU waits for `done` and must not hang.
            failures.fail(FAILED);
            timeline.relay_done(done);
            return;
        }
        let binding = &self.lanes[lane].bindings.iter().find(|(function_rows, _)| *function_rows == rows).unwrap().1;
        let request = self.requests.fetch_add(1, Ordering::AcqRel) + 1;
        let fault = self.fault.as_ref().filter(|fault| fault.at == request).map(|fault| fault.kind);
        let ffn = self.clone();
        let completion_timeline = timeline.clone();
        let completion_failures = failures.clone();
        let completion = Box::new(move |success: bool| {
            if !success || fault == Some(FaultKind::Completion) {
                // The result is garbage: the executor retires and the prefill fails.
                ffn.retire();
                completion_failures.fail(FAILED);
            }
            // On failure too: the GPU waits for `done` and must not hang.
            completion_timeline.relay_done(done);
        });
        let enqueued = if fault == Some(FaultKind::Enqueue) {
            Err(AneError::Unavailable("injected enqueue fault".to_string()))
        } else {
            self.program.enqueue(binding, &timeline.event, ready, done, completion)
        };
        if let Err(error) = enqueued {
            self.retire();
            failures.fail(FAILED);
            timeline.relay_done(done);
            eprintln!("uzu: ANE enqueue failed, MLPs run on the GPU from now on: {error}");
        }
    }

    /// Stops running MLPs after the first ANE failure.
    fn retire(&self) {
        self.healthy.store(false, Ordering::Release);
    }
}
