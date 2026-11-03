//! The Neural Engine FFN executor: one W8A8 program shared by every dense MLP of a shape.
//!
//! Per MLP call (prefill batches only) the GPU
//! 1. stages the layer's W4 weights into int8 rows the ANE reads (`AneStageUpRows`, `AneStageDownRows`), folding the
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

use std::sync::Arc;

use metal::MTLSharedEventListener;
use objc2::rc::Retained;
use parking_lot::Mutex;

use crate::{
    backends::{
        common::{Backend, CommandBufferEncoding, CpuBuffer},
        metal::{
            Metal,
            ane::{
                AneBinding, AneElement, AneError, AneProgram, AneRuntime, AneSurface,
                mil::FfnShape,
                shared::{
                    AneExecutor, AneHealth, AneKernels, CHUNK, GpuBuffer, MIN_TOKENS, ROWS, SEGMENT, random_signs,
                    read_signs, scale_stride, signed_hadamard_blocks, stageable, submit, upload, upload_sign_bits,
                },
            },
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::AneStageDownRowsMetalKernel,
        },
    },
    encodable_block::mlp::DenseMlpWeights,
};

const INTERMEDIATE_ROTATION_BLOCK: u32 = 512;
const UP_ROTATION_BLOCK: u32 = 32;

/// Per-layer constants of an ANE-run MLP.
pub struct AneLayer {
    /// Position among the model's ANE call sites: orders the layer's segment across pipelined
    /// prefill chunks.
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

/// What one lane (see `MetalLane`) of the pipelined prefill needs to run its chunk's MLPs: chunks
/// on different lanes overlap, so they cannot share surfaces.
struct LaneResources {
    surfaces: Mutex<Surfaces>,
    bindings: Vec<(u32, AneBinding)>,
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
    kernels: AneKernels,
    stage_down: AneStageDownRowsMetalKernel,
    pack_signs: Vec<f32>,
    pack_signs_buffer: GpuBuffer,
    intermediate_signs_buffer: GpuBuffer,
    up_output_signs: Vec<f32>,
    health: Arc<AneHealth>,
}

impl AneExecutor for AneFfn {
    fn health(&self) -> &AneHealth {
        &self.health
    }

    fn program(&self) -> &AneProgram {
        &self.program
    }

    fn binding(
        &self,
        lane: usize,
        rows: u32,
    ) -> &AneBinding {
        &self.lanes[lane].bindings.iter().find(|(function_rows, _)| *function_rows == rows).unwrap().1
    }

    fn listener(
        &self,
        lane: usize,
    ) -> &MTLSharedEventListener {
        &self.lanes[lane].listener
    }
}

impl AneFfn {
    /// Builds the executor for the shape of `weights` (compiling the program, ~4 s the first time
    /// per Mac, cached afterwards). `None` if the weights are not in a form it runs.
    pub fn new(
        context: &MetalContext,
        health: Arc<AneHealth>,
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
        let pack_signs = random_signs(hidden as usize, 1);

        Ok(Some(Self {
            hidden,
            intermediate,
            shape,
            program,
            lanes,
            kernels: AneKernels::new(context, true)?,
            stage_down: AneStageDownRowsMetalKernel::new(context, intermediate / CHUNK).map_err(kernel_error)?,
            pack_signs_buffer: upload(context, &pack_signs).map_err(kernel_error)?,
            pack_signs,
            intermediate_signs_buffer: upload_sign_bits(context, &intermediate_signs).map_err(kernel_error)?,
            up_output_signs,
            health,
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
            down_pre_signs: upload_sign_bits(context, &down_input_signs)?,
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

    /// Whether a batch of `tokens` runs on the ANE.
    pub fn accepts(
        &self,
        tokens: u32,
    ) -> bool {
        tokens >= MIN_TOKENS && self.shape.rows_for(tokens).is_some() && self.health.is_healthy()
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
        for (staged, scales, row_offset) in [(&mut *wu, &mut *su, 0), (&mut *wg, &mut *sg, intermediate)] {
            self.kernels.stage_rows(up, &layer.up_pre_signs, staged, scales, row_offset, intermediate, command_buffer);
        }
        let (wd_row_stride, sd_stride) = (wd.row_stride_bytes() as u32, sd.row_stride_elements() as u32);
        self.stage_down.encode(
            down.values(),
            down.scales().unwrap(),
            down.zero_points().unwrap(),
            &layer.down_pre_signs,
            &self.intermediate_signs_buffer,
            wd.buffer_mut(),
            sd.buffer_mut(),
            0,
            hidden,
            scale_stride(down),
            wd_row_stride,
            sd_stride,
            command_buffer,
        );

        self.kernels.pack(input, &self.pack_signs_buffer, tokens, x, tx, command_buffer)?;
        submit(self, layer.index, rows, command_buffer)?;
        self.kernels.unpack(y, tx, &layer.down_output_signs, hidden, tokens, command_buffer)
    }
}
