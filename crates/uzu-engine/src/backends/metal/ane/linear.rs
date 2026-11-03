//! The Neural Engine executor of single linear layers (the DeltaNet and attention projections):
//! one matmul program per output width and weight type (fp16 activations, fp16 weights by default:
//! int8 weights, one scale per row, add 0.8-1.1% relative error per projection, which the model's
//! logits do not tolerate on top of the int8 MLPs; `UZU_ANE_FP16_WEIGHTS`).
//!
//! A call is the FFN's up projection alone: the GPU stages the layer's W4 weights into int8 rows
//! `r' = H_in(D_a D_in H32(r))` (`AneStageUpRows`), packs the activations `a = H_in(D_a x)` (`AnePack`),
//! the ANE computes `y = s_row (W' a)`, and the GPU unpacks `out = D_out H32(y * tx)`, the layer's
//! output rotation (`AneUnpack` without a scale row). See [`super::ffn`] and
//! scripts/ane_prefill/numerics/pipeline_v1.py.

use std::sync::Arc;

use metal::MTLSharedEventListener;
use objc2::rc::Retained;
use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferEncoding, CpuBuffer,
            kernel::matmul::{ActivationFormat, MatmulA},
        },
        metal::{
            Metal,
            ane::{
                AneBinding, AneElement, AneError, AneProgram, AneRuntime, AneSurface,
                mil::{FfnSurfaceSpec, LinearShape, linear_program},
                shared::{
                    AneExecutor, AneHealth, AneKernels, CHUNK, GpuBuffer, MIN_TOKENS, ROWS, SEGMENT, WEIGHT_CLIP,
                    random_signs, read_signs, stageable, submit, upload, weight_clip,
                },
            },
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
        },
    },
    encodable_block::linear::{Linear, LinearEncodeError, LinearInput, LinearRole, LinearWeights},
};

/// Rows of the weight, scale and output surfaces, the widest linear layer one lane can run.
pub const MAX_OUTPUT: u32 = 16384;
/// Whether the weights of the roles `UZU_ANE_FP16_WEIGHTS` does not name are fp16 (all roles by
/// default: with int8 weights the mean KL of the prefill logits against the GPU is 1.3e-3 instead of 9.8e-4).
const FP16_WEIGHTS_DEFAULT: bool = true;
/// The output rotation is applied per 32-channel block.
const OUTPUT_BLOCK: u32 = 32;

/// The weight, scale and output surfaces of programs.
struct Operands {
    w: AneSurface,
    s: AneSurface,
    y: AneSurface,
}

impl Operands {
    fn new(
        context: &MetalContext,
        shape: &LinearShape,
    ) -> Result<Self, AneError> {
        let surface = |spec: FfnSurfaceSpec| AneSurface::new(context, spec.rows, spec.width, spec.element);
        let [_, w, s]: [_; 3] = shape.inputs().try_into().expect("linear programs take three inputs");
        Ok(Self {
            w: surface(w)?,
            s: surface(s)?,
            y: surface(shape.output_spec())?,
        })
    }
}

struct Surfaces {
    x: AneSurface,
    tx: AneSurface,
    /// What the programs with int8 weights share, created with the first of them.
    int8: Option<Operands>,
}

/// What every linear program shares: per lane (see `MetalLane`) one set of surfaces sized for the
/// widest layer (a lane's ANE calls run one after the other, and chunks on different lanes
/// overlap, so lanes cannot share them), and the GPU kernels around the programs.
pub struct AneLinears {
    surfaces: [Mutex<Surfaces>; 2],
    kernels: AneKernels,
    pack_signs: Vec<f32>,
    pack_signs_buffer: GpuBuffer,
    health: Arc<AneHealth>,
}

/// The shape of the program of `output`-wide layers whose weights are `weights`. Int8 weights use
/// the surfaces every program shares (sized for the widest layer); fp16 weights are twice as big and
/// only some layers use them, so such a program has surfaces of its own, `output` rows tall.
fn template_shape(
    output: u32,
    weights: AneElement,
) -> LinearShape {
    LinearShape {
        input: CHUNK,
        output,
        max_output: if weights == AneElement::Int8 {
            MAX_OUTPUT
        } else {
            output
        },
        segment: SEGMENT,
        rows: ROWS.to_vec(),
        max_rows: *ROWS.last().unwrap(),
        activations: activations(),
        weights,
    }
}

/// The element type of the linear programs' activations: fp16 rounds them only to fp16, int8 (the
/// ANE's faster W8A8 path) to 127 levels per token. `UZU_ANE_ACTIVATIONS=int8` selects int8.
fn activations() -> AneElement {
    if std::env::var("UZU_ANE_ACTIVATIONS").is_ok_and(|value| value == "int8") {
        AneElement::Int8
    } else {
        AneElement::Float16
    }
}

impl AneLinears {
    pub fn new(
        context: &MetalContext,
        health: Arc<AneHealth>,
    ) -> Result<Self, AneError> {
        let shape = template_shape(OUTPUT_BLOCK, AneElement::Int8);
        let surface = |spec: FfnSurfaceSpec| AneSurface::new(context, spec.rows, spec.width, spec.element);
        let lane_surfaces = || -> Result<Mutex<Surfaces>, AneError> {
            let [x, _, _]: [_; 3] = shape.inputs().try_into().expect("linear programs take three inputs");
            Ok(Mutex::new(Surfaces {
                x: surface(x)?,
                tx: surface(shape.token_scales())?,
                int8: None,
            }))
        };
        let kernel_error = |error: MetalError| AneError::Unavailable(error.to_string());
        let pack_signs = random_signs(CHUNK as usize, 1);
        Ok(Self {
            surfaces: [lane_surfaces()?, lane_surfaces()?],
            kernels: AneKernels::new(
                context,
                false,
                activations(),
                weight_clip(&["UZU_ANE_WEIGHT_CLIP"], WEIGHT_CLIP),
            )?,
            pack_signs_buffer: upload(context, &pack_signs).map_err(kernel_error)?,
            pack_signs,
            health,
        })
    }
}

/// The program of the linear layers with one output width.
pub struct AneLinearProgram {
    shape: LinearShape,
    program: AneProgram,
    bindings: [Vec<(u32, AneBinding)>; 2],
    listeners: [Retained<MTLSharedEventListener>; 2],
    /// Per lane, the surfaces of a program with fp16 weights (see [`template_shape`]).
    own: Option<[Mutex<Operands>; 2]>,
    shared: Arc<AneLinears>,
}

impl AneExecutor for AneLinearProgram {
    fn health(&self) -> &AneHealth {
        &self.shared.health
    }

    fn program(&self) -> &AneProgram {
        &self.program
    }

    fn binding(
        &self,
        lane: usize,
        rows: u32,
    ) -> &AneBinding {
        &self.bindings[lane].iter().find(|(function_rows, _)| *function_rows == rows).unwrap().1
    }

    fn listener(
        &self,
        lane: usize,
    ) -> &MTLSharedEventListener {
        &self.listeners[lane]
    }
}

/// Whether linear layers of `role` run on the ANE. `UZU_ANE_LINEARS` picks them (comma separated:
/// `deltanet`, `attention`, or single roles `deltanet_in`, `deltanet_out`, `attention_in`,
/// `attention_out`; `none` for no linears; default all), to compare configurations.
pub fn runs_on_ane(role: LinearRole) -> bool {
    selected("UZU_ANE_LINEARS", role).unwrap_or(true)
}

/// The element type of the weights of the ANE program of `role`. `UZU_ANE_FP16_WEIGHTS` picks the
/// roles whose weights are fp16 (same syntax as `UZU_ANE_LINEARS`): more accurate, slower on the ANE.
pub fn weights_element(role: LinearRole) -> AneElement {
    if selected("UZU_ANE_FP16_WEIGHTS", role).unwrap_or(FP16_WEIGHTS_DEFAULT) {
        AneElement::Float16
    } else {
        AneElement::Int8
    }
}

/// Whether `role` is in the comma separated list in the environment variable `variable`, or `None`
/// if it is not set.
fn selected(
    variable: &str,
    role: LinearRole,
) -> Option<bool> {
    let selection = std::env::var(variable).ok()?;
    let (group, name) = match role {
        LinearRole::DeltaNetInput => ("deltanet", "deltanet_in"),
        LinearRole::DeltaNetOutput => ("deltanet", "deltanet_out"),
        LinearRole::AttentionInput => ("attention", "attention_in"),
        LinearRole::AttentionOutput => ("attention", "attention_out"),
    };
    Some(selection.split(',').any(|item| item == group || item == name || item == "all"))
}

/// The output width of a layer these programs can run, or `None` if they cannot (other format,
/// a width the kernels do not take, or a layer whose input the ANE would not see unrotated).
pub fn supported_output(weights: &LinearWeights<'_, Metal>) -> Option<u32> {
    if !stageable(weights) {
        return None;
    }
    let input = weights.input_rht_signs?.as_slice::<i32>().len() as u32;
    let output = weights.output_rht_signs?.as_slice::<i32>().len() as u32;
    // The GPU kernels hold a token (and a staged row) as 4096-value chunks in registers.
    if input != CHUNK || output == 0 || output > MAX_OUTPUT || !output.is_multiple_of(OUTPUT_BLOCK) {
        return None;
    }
    Some(output)
}

impl AneLinearProgram {
    /// Builds the program of `output`-wide layers (compiling it the first time per Mac).
    pub fn new(
        context: &MetalContext,
        shared: Arc<AneLinears>,
        output: u32,
        weights: AneElement,
    ) -> Result<Self, AneError> {
        let shape = template_shape(output, weights);
        let runtime = AneRuntime::new()?;
        let (mil, blob) = linear_program(&shape);
        let program = AneProgram::new(runtime, &mil, &blob)?;

        let own = if weights == AneElement::Int8 {
            None
        } else {
            Some([Mutex::new(Operands::new(context, &shape)?), Mutex::new(Operands::new(context, &shape)?)])
        };
        let bind = |lane: usize| -> Result<Vec<(u32, AneBinding)>, AneError> {
            let mut surfaces = shared.surfaces[lane].lock();
            let own = own.as_ref().map(|own| own[lane].lock());
            if own.is_none() && surfaces.int8.is_none() {
                surfaces.int8 = Some(Operands::new(context, &shape)?);
            }
            let Operands {
                w,
                s,
                y,
            } = own.as_deref().or(surfaces.int8.as_ref()).expect("operands exist");
            let inputs = [("x", &surfaces.x), ("w", w), ("s", s)];
            shape
                .rows
                .iter()
                .map(|rows| {
                    let name = LinearShape::function_name(*rows);
                    let procedure = program.procedure(&name).ok_or(AneError::UnknownProcedure(name))?;
                    Ok((*rows, program.bind(procedure, &inputs, y)?))
                })
                .collect()
        };
        let bindings = [bind(0)?, bind(1)?];
        Ok(Self {
            shape,
            program,
            bindings,
            listeners: [MTLSharedEventListener::new(), MTLSharedEventListener::new()],
            own,
            shared,
        })
    }

    pub fn output(&self) -> u32 {
        self.shape.output
    }

    /// The element type of the weights the program reads.
    pub fn weights(&self) -> AneElement {
        self.shape.weights
    }

    /// The per-layer constants for a layer these programs can run, `supported_output` wide.
    pub fn layer(
        &self,
        context: &MetalContext,
        weights: &LinearWeights<'_, Metal>,
    ) -> Result<AneLinearLayer, MetalError> {
        let input_signs = read_signs(weights.input_rht_signs.expect("supported layers have input signs"));
        let pre_signs: Vec<f32> =
            input_signs.iter().zip(&self.shared.pack_signs).map(|(input, pack)| input * pack).collect();
        Ok(AneLinearLayer {
            index: context.next_ane_layer_index(),
            pre_signs: upload(context, &pre_signs)?,
            output_signs: upload(
                context,
                &read_signs(weights.output_rht_signs.expect("supported layers have output signs")),
            )?,
        })
    }

    /// Whether a batch of `tokens` runs on the ANE.
    pub fn accepts(
        &self,
        tokens: u32,
    ) -> bool {
        tokens >= MIN_TOKENS && self.shape.rows_for(tokens).is_some() && self.shared.health.is_healthy()
    }

    /// Encodes the layer's matmul of `tokens` rows of `input` (token-major bf16 `[tokens, 4096]`),
    /// returning its output (`[tokens, output]`).
    pub fn encode(
        self: &Arc<Self>,
        layer: &AneLinearLayer,
        weights: &LinearWeights<'_, Metal>,
        input: &<Metal as Backend>::ScratchBuffer,
        tokens: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let command_buffer = &mut *command_buffer.debug_group("linear (ane)");
        let rows = self.shape.rows_for(tokens).expect("batch accepted by the ANE executor");
        let kernels = &self.shared.kernels;
        let lane = command_buffer.lane();
        let mut surfaces = self.shared.surfaces[lane].lock();
        let mut own = self.own.as_ref().map(|own| own[lane].lock());
        let Surfaces {
            x,
            tx,
            int8,
        } = &mut *surfaces;
        let Operands {
            w,
            s,
            y,
        } = own.as_deref_mut().or(int8.as_mut()).expect("operands exist");

        kernels.stage_rows(weights.matrix, &layer.pre_signs, w, s, 0, self.shape.output, command_buffer);
        kernels.pack(input, &self.shared.pack_signs_buffer, tokens, x, tx, command_buffer)?;
        submit(self, layer.index, rows, command_buffer)?;
        kernels.unpack(y, tx, &layer.output_signs, self.shape.output, tokens, command_buffer)
    }
}

/// Per-layer constants of an ANE-run linear layer.
pub struct AneLinearLayer {
    /// Position among the model's ANE call sites: orders the layer's segment across pipelined
    /// prefill chunks.
    index: usize,
    /// The layer's input signs times the pack signs `D_a`.
    pre_signs: GpuBuffer,
    output_signs: GpuBuffer,
}

/// A linear layer whose prefill batches run on the ANE; everything else (decode, small batches,
/// inputs the GPU linear prepares itself, and every batch once the ANE has failed) runs on the
/// wrapped GPU linear.
pub struct AneLinear {
    inner: Box<dyn Linear<Metal>>,
    program: Arc<AneLinearProgram>,
    layer: AneLinearLayer,
    /// Recorded by the test-only ANE/GPU comparison.
    #[cfg(test)]
    role: LinearRole,
}

impl AneLinear {
    /// Wraps `inner` if the ANE can run it; otherwise gives it back.
    pub fn wrap(
        context: &MetalContext,
        inner: Box<dyn Linear<Metal>>,
        role: LinearRole,
    ) -> Result<Self, Box<dyn Linear<Metal>>> {
        let Some(weights) = inner.weights() else {
            return Err(inner);
        };
        let Some(program) = context.ane_linear(&weights, role) else {
            return Err(inner);
        };
        let layer = match program.layer(context, &weights) {
            Ok(layer) => layer,
            Err(error) => {
                eprintln!("uzu: ANE layer setup failed, linear stays on the GPU: {error}");
                return Err(inner);
            },
        };
        Ok(Self {
            inner,
            program,
            layer,
            #[cfg(test)]
            role,
        })
    }
}

impl Linear<Metal> for AneLinear {
    fn encode_input(
        &self,
        input: LinearInput<Metal>,
        batch_dim: u32,
        command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, LinearEncodeError<Metal>> {
        if let MatmulA::FullPrecision {
            values,
            offset: 0,
        } = &input
            && self.program.accepts(batch_dim)
        {
            #[cfg(test)]
            if comparison::enabled() {
                return self.encode_compared(input, batch_dim, command_buffer);
            }
            let weights = self.inner.weights().expect("weights were exposed when the linear was wrapped");
            return self
                .program
                .encode(&self.layer, &weights, values, batch_dim, command_buffer)
                .map_err(LinearEncodeError::Backend);
        }
        // Still part of the pipelined prefill: keep this layer after the previous chunk's.
        command_buffer.order_after_previous_chunk(self.layer.index);
        self.inner.encode_input(input, batch_dim, command_buffer)
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &MetalContext,
    ) -> ActivationFormat {
        self.inner.select_activation_format(batch_dim, context)
    }

    fn weights(&self) -> Option<LinearWeights<'_, Metal>> {
        self.inner.weights()
    }
}

/// Test-only: the ANE and the GPU output of the same input, to measure the ANE path's error on
/// real activations (see `unit/engine/language_model/ane_linear_numerics_test.rs`).
#[cfg(test)]
pub mod comparison {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use parking_lot::Mutex;

    use crate::{
        backends::{common::Backend, metal::Metal},
        encodable_block::linear::LinearRole,
    };

    pub struct Comparison {
        pub role: LinearRole,
        pub output: u32,
        pub ane: <Metal as Backend>::GlobalBufferCpuGpu,
        pub gpu: <Metal as Backend>::GlobalBufferCpuGpu,
    }

    pub static RECORDS: Mutex<Vec<Comparison>> = Mutex::new(Vec::new());
    static ENABLED: AtomicBool = AtomicBool::new(false);
    /// Record every `STRIDE`-th call site only (the buffers are large).
    static STRIDE: AtomicUsize = AtomicUsize::new(1);

    pub fn enable(stride: usize) {
        STRIDE.store(stride, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
    }

    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    pub fn records_index(index: usize) -> bool {
        index % STRIDE.load(Ordering::Relaxed) == 0
    }
}

#[cfg(test)]
impl AneLinear {
    /// Runs both paths on `input`, records their outputs, and returns the GPU's, so later layers
    /// see clean inputs and each projection's error is measured on its own.
    fn encode_compared(
        &self,
        input: LinearInput<Metal>,
        batch_dim: u32,
        command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, LinearEncodeError<Metal>> {
        use crate::backends::common::Context;

        let MatmulA::FullPrecision {
            values,
            ..
        } = &input
        else {
            unreachable!("checked by the caller")
        };
        let weights = self.inner.weights().expect("weights were exposed when the linear was wrapped");
        let ane = self
            .program
            .encode(&self.layer, &weights, values, batch_dim, command_buffer)
            .map_err(LinearEncodeError::Backend)?;
        drop(weights);
        let gpu = self.inner.encode_input(input, batch_dim, command_buffer)?;
        if comparison::records_index(self.layer.index) {
            let size = (batch_dim * self.program.output() * 2) as usize;
            let context = command_buffer.context();
            let mut ane_copy = context.create_buffer_cpu_gpu(size).map_err(LinearEncodeError::Backend)?;
            let mut gpu_copy = context.create_buffer_cpu_gpu(size).map_err(LinearEncodeError::Backend)?;
            command_buffer.encode_copy(&ane, &mut ane_copy);
            command_buffer.encode_copy(&gpu, &mut gpu_copy);
            comparison::RECORDS.lock().push(comparison::Comparison {
                role: self.role,
                output: self.program.output(),
                ane: ane_copy,
                gpu: gpu_copy,
            });
        }
        Ok(gpu)
    }
}
