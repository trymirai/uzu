//! Mixture-of-experts MLP block: a float32 sigmoid router over stacked 4-bit routed experts.
//!
//! Small batches (decode and speculative verify, up to [`MOE_GEMV_MAX_TOKENS`] tokens) take three dispatches: the
//! router (logits, top-k selection and grouping of the routes by expert), the gate/up projections with the activation
//! and the route weight, and the down projections summed per token. Each expert kernel reads every active expert's
//! rows once for all of the expert's tokens. Larger batches run the grouped GEMM path on the matrix units; for affine
//! experts it dequantizes each weight tile once for all of an expert's routes, which it pads to multiples of 16.

pub mod weights;

use derive_more::Debug;
use thiserror::Error;

pub use self::weights::{ExpertWeights, MoeWeightsError};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::moe::{
            MOE_GEMV_MAX_TOKENS, MOE_MAX_EXPERTS, MOE_MAX_TOP_K, MOE_ROUTE_EXPERTS_PER_GROUP, MoeExpertFormat,
        },
        kernel::{
            MoeCombineRoutesKernel, MoeExpertDownBatchKernel, MoeExpertDownKernel, MoeExpertUpBatchKernel,
            MoeExpertUpKernel, MoeGroupedAffineDownKernel, MoeGroupedAffineUpKernel, MoeGroupedDownKernel,
            MoeGroupedUpKernel, MoeRouteFinalizeKernel, MoeRouteKernel, MoeRouterLogitsKernel, MoeSumRoutesKernel,
        },
    },
    config::{
        activation::AnyActivation,
        mlp::{
            mixture_of_experts::MixtureOfExpertsConfig,
            routing_function::{
                AnyRoutingFunction,
                sigmoid_routing::{RoutingBias, SigmoidRouting},
            },
        },
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::DataType,
    encodable_block::mlp::{Mlp, MlpEncodeError},
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum MixtureOfExpertsNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Expert weights error: {0}")]
    Weights(#[from] MoeWeightsError<B>),
    #[error("Unsupported mixture-of-experts configuration: {0}")]
    Unsupported(String),
}

/// How the router turns sigmoid scores into routes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRouting {
    pub top_k: u32,
    pub routed_scaling_factor: f32,
    /// The expert bias shifts the scores that select experts.
    pub bias_selects: bool,
    /// The expert bias shifts the weights of the selected experts.
    pub bias_weights: bool,
}

/// The gated activation of an expert in float32: `cap(up, up_soft_cap) * cap(gate, gate_soft_cap) *
/// sigmoid(alpha * gate)`, where a missing cap leaves its operand unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeActivation {
    pub up_soft_cap: Option<f32>,
    pub gate_soft_cap: Option<f32>,
    pub alpha: f32,
}

/// Router parameters, all float32.
pub struct RouterWeights<B: Backend> {
    /// `[experts, model_dim]`.
    pub weights: B::GlobalGpuBuffer,
    /// `[experts]`, added to the logits.
    pub biases: Option<B::GlobalGpuBuffer>,
    /// `[experts]`, the routing bias.
    pub expert_biases: Option<B::GlobalGpuBuffer>,
}

/// Everything a mixture-of-experts block is made of, already on the device.
pub struct MixtureOfExpertsParts<B: Backend> {
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub experts: u32,
    pub data_type: DataType,
    pub routing: MoeRouting,
    pub activation: MoeActivation,
    pub router: RouterWeights<B>,
    /// `[experts, 2 * hidden_dim, model_dim]`: the up rows, then the gate rows.
    pub up: ExpertWeights<B>,
    /// `[experts, model_dim, hidden_dim]`.
    pub down: ExpertWeights<B>,
}

/// Weight rows one grouped GEMM tile covers: the down kernel's output columns, twice the up kernel's hidden units.
const GROUPED_TILE_ROWS: u32 = 128;
/// Inputs one grouped GEMM step consumes.
const GROUPED_TILE_DEPTH: u32 = 32;
/// Experts one router logits threadgroup covers.
const ROUTER_LOGITS_BLOCK: u32 = 32;
/// Weight rows one affine grouped GEMM tile covers: the down kernel's output columns, twice the up kernel's hidden
/// units.
const GROUPED_AFFINE_TILE_ROWS: u32 = 128;

/// Kernels of the large-batch path: router logits on simdgroup matrices, one-threadgroup routing, the grouped expert
/// GEMMs on the matrix units and the per-token sum of the route outputs.
struct GroupedKernels<B: Backend> {
    router_logits: <B::Kernels as Kernels>::MoeRouterLogitsKernel,
    route_finalize: <B::Kernels as Kernels>::MoeRouteFinalizeKernel,
    experts: GroupedExpertKernels<B>,
    combine: <B::Kernels as Kernels>::MoeCombineRoutesKernel,
}

/// The grouped expert GEMMs: affine experts that tile by [`GROUPED_AFFINE_TILE_ROWS`] dequantize each weight tile
/// once for all of an expert's routes; the others run the format-generic kernels.
enum GroupedExpertKernels<B: Backend> {
    Generic {
        up: <B::Kernels as Kernels>::MoeGroupedUpKernel,
        down: <B::Kernels as Kernels>::MoeGroupedDownKernel,
    },
    Affine {
        up: <B::Kernels as Kernels>::MoeGroupedAffineUpKernel,
        down: <B::Kernels as Kernels>::MoeGroupedAffineDownKernel,
    },
}

/// Kernels of the affine batch path for 2 to [`MOE_GEMV_MAX_TOKENS`] tokens: gate/up and down stream every active
/// expert's rows once for all of its routes, [`BatchKernels::pass`] routes at a time, and a last dispatch sums each
/// token's route outputs.
struct BatchKernels<B: Backend> {
    /// Up and down kernels by the most routes one pass of an expert's rows serves.
    up: [(u32, <B::Kernels as Kernels>::MoeExpertUpBatchKernel); 2],
    down: [(u32, <B::Kernels as Kernels>::MoeExpertDownBatchKernel); 2],
    sum: <B::Kernels as Kernels>::MoeSumRoutesKernel,
}

impl<B: Backend> BatchKernels<B> {
    /// The most routes one pass serves for a batch of `tokens`: two tokens route at most two routes to an expert.
    fn pass(tokens: u32) -> u32 {
        if tokens <= 2 {
            2
        } else {
            4
        }
    }
}

pub struct MixtureOfExpertsMlp<B: Backend> {
    parts: MixtureOfExpertsParts<B>,
    /// Router kernels for at most 1, 2, 4, 8, 16 and [`MOE_GEMV_MAX_TOKENS`] tokens.
    route_kernels: [(u32, <B::Kernels as Kernels>::MoeRouteKernel); 6],
    /// Up kernels for at most 1 and [`MOE_GEMV_MAX_TOKENS`] tokens.
    up_kernels: [(u32, <B::Kernels as Kernels>::MoeExpertUpKernel); 2],
    /// Down kernels for at most 1, 8 and [`MOE_GEMV_MAX_TOKENS`] tokens.
    down_kernels: [(u32, <B::Kernels as Kernels>::MoeExpertDownKernel); 3],
    /// Affine experts only: the small-batch path for more than one token.
    batch: Option<BatchKernels<B>>,
    /// Absent when the shapes do not tile or the device has no matrix units; large batches then run the small-batch
    /// path in slices of [`MOE_GEMV_MAX_TOKENS`] tokens.
    grouped: Option<GroupedKernels<B>>,
}

impl<B: Backend> MixtureOfExpertsMlp<B> {
    pub async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &MixtureOfExpertsConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, MixtureOfExpertsNewError<B>> {
        let unsupported = |message: String| Err(MixtureOfExpertsNewError::Unsupported(message));
        if config.num_shared_experts != 0 || config.gate_config.is_some() {
            return unsupported("shared experts are not implemented".into());
        }
        if !config.has_float32_router {
            return unsupported("the router must be float32".into());
        }
        let expert_config = &config.expert_config;
        if expert_config.has_up_biases || expert_config.has_down_biases {
            return unsupported("expert biases are not implemented".into());
        }
        if expert_config.gate_clipping.into_pair().is_some() || expert_config.up_clipping.into_pair().is_some() {
            return unsupported("expert clipping is not implemented".into());
        }
        let activation = match &expert_config.activation {
            AnyActivation::SoftCappedSiLU(activation) => MoeActivation {
                up_soft_cap: expert_config.up_soft_cap,
                gate_soft_cap: Some(activation.soft_cap),
                alpha: 1.0,
            },
            AnyActivation::SiLU(activation) => MoeActivation {
                up_soft_cap: expert_config.up_soft_cap,
                gate_soft_cap: None,
                alpha: activation.alpha,
            },
            activation => return unsupported(format!("expert activation {activation:?} is not implemented")),
        };
        let AnyRoutingFunction::SigmoidRouting(SigmoidRouting {
            bias_mode,
            routed_scaling_factor,
            ..
        }) = &config.routing_function
        else {
            return unsupported(format!("routing function {:?} is not implemented", config.routing_function));
        };
        let (bias_selects, bias_weights) = match bias_mode {
            None => (false, false),
            Some(RoutingBias::Gating) => (true, false),
            Some(RoutingBias::Weights) => (false, true),
            Some(RoutingBias::Both) => (true, true),
        };
        let experts = config.num_routed_experts;
        let hidden_dim = config.expert_hidden_dim;

        let router_tree = parameter_tree.subtree("router");
        let router_weights_tree = router_tree.subtree("weights");
        match router_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")? {
            AnyWeightMatrixSpec::FullPrecisionSpec(spec) if spec.layout == Layout::OutputInput => {},
            spec => {
                return unsupported(format!("the router must be a full precision output-input matrix, got {spec:?}"));
            },
        }
        let router = RouterWeights {
            weights: router_weights_tree.read_buffer("weights", &[experts, model_dim], DataType::F32).await?,
            biases: if config.router_has_biases {
                Some(router_tree.read_buffer("biases", &[experts], DataType::F32).await?)
            } else {
                None
            },
            expert_biases: if bias_mode.is_some() {
                Some(parameter_tree.read_buffer("expert_bias", &[experts], DataType::F32).await?)
            } else {
                None
            },
        };

        let experts_tree = parameter_tree.subtree("routed_experts");
        let up = ExpertWeights::load(
            experts,
            2 * hidden_dim,
            model_dim,
            &experts_tree.subtree("up_projection").subtree("weights"),
            context,
        )
        .await?;
        let down = ExpertWeights::load(
            experts,
            model_dim,
            hidden_dim,
            &experts_tree.subtree("down_projection").subtree("weights"),
            context,
        )
        .await?;

        Self::from_parts(
            MixtureOfExpertsParts {
                model_dim,
                hidden_dim,
                experts,
                data_type,
                routing: MoeRouting {
                    top_k: config.num_active_routed_experts,
                    routed_scaling_factor: *routed_scaling_factor,
                    bias_selects,
                    bias_weights,
                },
                activation,
                router,
                up,
                down,
            },
            context,
        )
    }

    pub fn from_parts(
        parts: MixtureOfExpertsParts<B>,
        context: &B::Context,
    ) -> Result<Self, MixtureOfExpertsNewError<B>> {
        let unsupported = |message: String| Err(MixtureOfExpertsNewError::Unsupported(message));
        let MixtureOfExpertsParts {
            model_dim,
            hidden_dim,
            experts,
            data_type,
            routing,
            ..
        } = parts;
        if data_type != DataType::BF16 {
            return unsupported(format!("activations must be bfloat16, got {data_type:?}"));
        }
        if experts > MOE_MAX_EXPERTS || !experts.is_multiple_of(MOE_ROUTE_EXPERTS_PER_GROUP) {
            return unsupported(format!(
                "{experts} routed experts: at most {MOE_MAX_EXPERTS}, in multiples of {MOE_ROUTE_EXPERTS_PER_GROUP}"
            ));
        }
        if routing.top_k == 0 || routing.top_k > MOE_MAX_TOP_K.min(experts) {
            return unsupported(format!("{} active experts of {experts}", routing.top_k));
        }
        if !model_dim.is_multiple_of(256) || !hidden_dim.is_multiple_of(128) {
            return unsupported(format!(
                "model dimension {model_dim} must be a multiple of 256 and expert hidden dimension {hidden_dim} of 128"
            ));
        }
        if parts.up.format != parts.down.format {
            return unsupported("the up and down projections must share a format".into());
        }
        if (parts.up.rows, parts.up.cols, parts.down.rows, parts.down.cols)
            != (2 * hidden_dim, model_dim, model_dim, hidden_dim)
        {
            return unsupported("expert matrix shapes do not match the block dimensions".into());
        }
        let format = parts.up.format;
        let route_kernel = |max_tokens: u32| {
            <B::Kernels as Kernels>::MoeRouteKernel::new(
                context,
                data_type,
                max_tokens,
                parts.router.biases.is_some(),
                parts.router.expert_biases.is_some(),
                routing.bias_selects,
                routing.bias_weights,
            )
            .map(|kernel| (max_tokens, kernel))
            .map_err(MixtureOfExpertsNewError::Backend)
        };
        let route_kernels = [
            route_kernel(1)?,
            route_kernel(2)?,
            route_kernel(4)?,
            route_kernel(8)?,
            route_kernel(16)?,
            route_kernel(MOE_GEMV_MAX_TOKENS)?,
        ];
        let up_kernel = |max_tokens: u32| {
            <B::Kernels as Kernels>::MoeExpertUpKernel::new(
                context,
                data_type,
                max_tokens,
                format,
                parts.activation.up_soft_cap.is_some(),
                parts.activation.gate_soft_cap.is_some(),
            )
            .map(|kernel| (max_tokens, kernel))
            .map_err(MixtureOfExpertsNewError::Backend)
        };
        let up_kernels = [up_kernel(1)?, up_kernel(MOE_GEMV_MAX_TOKENS)?];
        let down_kernel = |max_tokens: u32| {
            <B::Kernels as Kernels>::MoeExpertDownKernel::new(context, data_type, max_tokens, format)
                .map(|kernel| (max_tokens, kernel))
                .map_err(MixtureOfExpertsNewError::Backend)
        };
        let down_kernels = [down_kernel(1)?, down_kernel(8)?, down_kernel(MOE_GEMV_MAX_TOKENS)?];
        let batch = if format == MoeExpertFormat::AffineU4 {
            let up = |pass: u32| {
                <B::Kernels as Kernels>::MoeExpertUpBatchKernel::new(
                    context,
                    data_type,
                    pass,
                    parts.activation.up_soft_cap.is_some(),
                    parts.activation.gate_soft_cap.is_some(),
                )
                .map(|kernel| (pass, kernel))
            };
            let down = |pass: u32| {
                <B::Kernels as Kernels>::MoeExpertDownBatchKernel::new(context, pass).map(|kernel| (pass, kernel))
            };
            Some(BatchKernels {
                up: [
                    up(2).map_err(MixtureOfExpertsNewError::Backend)?,
                    up(4).map_err(MixtureOfExpertsNewError::Backend)?,
                ],
                down: [
                    down(2).map_err(MixtureOfExpertsNewError::Backend)?,
                    down(4).map_err(MixtureOfExpertsNewError::Backend)?,
                ],
                sum: <B::Kernels as Kernels>::MoeSumRoutesKernel::new(context, data_type)
                    .map_err(MixtureOfExpertsNewError::Backend)?,
            })
        } else {
            None
        };
        let tiles = model_dim.is_multiple_of(GROUPED_TILE_ROWS)
            && hidden_dim.is_multiple_of(GROUPED_TILE_ROWS / 2)
            && hidden_dim.is_multiple_of(GROUPED_TILE_DEPTH)
            && experts.is_multiple_of(ROUTER_LOGITS_BLOCK);
        let grouped = tiles.then(|| Self::grouped_kernels(&parts, context).ok()).flatten();
        Ok(Self {
            parts,
            route_kernels,
            up_kernels,
            down_kernels,
            batch,
            grouped,
        })
    }

    fn grouped_kernels(
        parts: &MixtureOfExpertsParts<B>,
        context: &B::Context,
    ) -> Result<GroupedKernels<B>, B::Error> {
        let (data_type, format) = (parts.data_type, parts.up.format);
        Ok(GroupedKernels {
            router_logits: <B::Kernels as Kernels>::MoeRouterLogitsKernel::new(
                context,
                data_type,
                parts.router.biases.is_some(),
            )?,
            route_finalize: <B::Kernels as Kernels>::MoeRouteFinalizeKernel::new(
                context,
                parts.router.expert_biases.is_some(),
                parts.routing.bias_selects,
                parts.routing.bias_weights,
            )?,
            experts: if format == MoeExpertFormat::AffineU4
                && parts.model_dim.is_multiple_of(GROUPED_AFFINE_TILE_ROWS)
                && parts.hidden_dim.is_multiple_of(GROUPED_AFFINE_TILE_ROWS / 2)
            {
                GroupedExpertKernels::Affine {
                    up: <B::Kernels as Kernels>::MoeGroupedAffineUpKernel::new(
                        context,
                        data_type,
                        parts.activation.up_soft_cap.is_some(),
                        parts.activation.gate_soft_cap.is_some(),
                    )?,
                    down: <B::Kernels as Kernels>::MoeGroupedAffineDownKernel::new(context, data_type)?,
                }
            } else {
                GroupedExpertKernels::Generic {
                    up: <B::Kernels as Kernels>::MoeGroupedUpKernel::new(
                        context,
                        data_type,
                        format,
                        parts.activation.up_soft_cap.is_some(),
                        parts.activation.gate_soft_cap.is_some(),
                    )?,
                    down: <B::Kernels as Kernels>::MoeGroupedDownKernel::new(context, data_type, format)?,
                }
            },
            combine: <B::Kernels as Kernels>::MoeCombineRoutesKernel::new(context, data_type)?,
        })
    }

    /// Whether `tokens` rows take the grouped GEMM path.
    pub fn uses_grouped_path(
        &self,
        tokens: u32,
    ) -> bool {
        tokens > MOE_GEMV_MAX_TOKENS && self.grouped.is_some()
    }

    /// Runs the block on `tokens` rows of `input`, writing the rows of `output`.
    pub fn encode_rows(
        &self,
        input: impl BufferRef<Backend = B>,
        mut output: impl BufferMut<Backend = B>,
        tokens: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        if self.uses_grouped_path(tokens) {
            return self.encode_large_batch(input, output, tokens, command_buffer);
        }
        let row_bytes = self.parts.model_dim as usize * self.parts.data_type.size_in_bytes();
        let mut first_token = 0;
        while first_token < tokens {
            let batch = (tokens - first_token).min(MOE_GEMV_MAX_TOKENS);
            let range = first_token as usize * row_bytes..(first_token + batch) as usize * row_bytes;
            self.encode_small_batch(
                input.subrange(range.clone()),
                output.reborrow().subrange_mut(range),
                batch,
                command_buffer,
            )?;
            first_token += batch;
        }
        Ok(())
    }

    /// Runs the small-batch path for `tokens <= MOE_GEMV_MAX_TOKENS` rows of `input` into `output`.
    fn encode_small_batch(
        &self,
        input: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        tokens: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let parts = &self.parts;
        let (experts, top_k) = (parts.experts, parts.routing.top_k);
        let routes = tokens * top_k;
        let mut logits = command_buffer.allocate_scratch_for_shape(&[tokens, experts], DataType::F32)?;
        let mut route_experts = command_buffer.allocate_scratch_for_shape(&[routes], DataType::U32)?;
        let mut route_weights = command_buffer.allocate_scratch_for_shape(&[routes], DataType::F32)?;
        let mut expert_counts = command_buffer.allocate_scratch_for_shape(&[experts], DataType::U32)?;
        let mut expert_routes = command_buffer.allocate_scratch_for_shape(&[experts, tokens], DataType::U32)?;
        let mut active_experts = command_buffer.allocate_scratch_for_shape(&[experts + 1], DataType::U32)?;
        let mut hidden = command_buffer.allocate_scratch_for_shape(&[routes, parts.hidden_dim], DataType::F16)?;
        let activation = parts.activation;
        let mut finish_counter = command_buffer.allocate_constant_from_slice(&[0u32])?;
        let (_, route_kernel) = self
            .route_kernels
            .iter()
            .find(|(max_tokens, _)| tokens <= *max_tokens)
            .expect("the small batch path takes at most MOE_GEMV_MAX_TOKENS tokens");
        route_kernel.encode(
            input,
            &parts.router.weights,
            parts.router.biases.as_ref(),
            parts.router.expert_biases.as_ref(),
            &mut logits,
            &mut route_experts,
            &mut route_weights,
            &mut expert_counts,
            &mut expert_routes,
            &mut active_experts,
            &mut finish_counter,
            tokens,
            parts.model_dim,
            experts,
            top_k,
            parts.routing.routed_scaling_factor,
            command_buffer,
        );

        if let Some(batch) = self.batch.as_ref().filter(|_| tokens > 1) {
            let pass = BatchKernels::<B>::pass(tokens);
            let (_, up_kernel) = batch.up.iter().find(|(kernel_pass, _)| *kernel_pass == pass).expect("pass sizes");
            up_kernel.encode(
                input,
                &parts.up.codes,
                &parts.up.scales,
                &expert_routes,
                &active_experts,
                &route_weights,
                &mut hidden,
                tokens,
                parts.model_dim,
                parts.hidden_dim,
                experts,
                top_k,
                activation.up_soft_cap.unwrap_or(1.0),
                activation.gate_soft_cap.unwrap_or(1.0),
                activation.alpha,
                command_buffer,
            );
            let mut route_outputs =
                command_buffer.allocate_scratch_for_shape(&[routes, parts.model_dim], DataType::F32)?;
            let (_, down_kernel) = batch.down.iter().find(|(kernel_pass, _)| *kernel_pass == pass).expect("pass sizes");
            down_kernel.encode(
                &hidden,
                &parts.down.codes,
                &parts.down.scales,
                &expert_routes,
                &active_experts,
                &mut route_outputs,
                tokens,
                parts.model_dim,
                parts.hidden_dim,
                experts,
                top_k,
                command_buffer,
            );
            batch.sum.encode(&route_outputs, output, tokens, parts.model_dim, top_k, command_buffer);
            return Ok(());
        }

        let (_, up_kernel) = self
            .up_kernels
            .iter()
            .find(|(max_tokens, _)| tokens <= *max_tokens)
            .expect("the small batch path takes at most MOE_GEMV_MAX_TOKENS tokens");
        up_kernel.encode(
            input,
            &parts.up.codes,
            &parts.up.scales,
            parts.up.global_scales.as_ref(),
            &expert_routes,
            &active_experts,
            &route_weights,
            &mut hidden,
            tokens,
            parts.model_dim,
            parts.hidden_dim,
            top_k,
            activation.up_soft_cap.unwrap_or(1.0),
            activation.gate_soft_cap.unwrap_or(1.0),
            activation.alpha,
            command_buffer,
        );

        let (_, down_kernel) = self
            .down_kernels
            .iter()
            .find(|(max_tokens, _)| tokens <= *max_tokens)
            .expect("the small batch path takes at most MOE_GEMV_MAX_TOKENS tokens");
        down_kernel.encode(
            &hidden,
            &parts.down.codes,
            &parts.down.scales,
            parts.down.global_scales.as_ref(),
            &expert_routes,
            &active_experts,
            output,
            tokens,
            parts.model_dim,
            parts.hidden_dim,
            top_k,
            command_buffer,
        );
        Ok(())
    }
}

impl<B: Backend> MixtureOfExpertsMlp<B> {
    /// Runs the grouped GEMM path on `tokens` rows.
    fn encode_large_batch(
        &self,
        input: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        tokens: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let grouped = self.grouped.as_ref().expect("the grouped path needs its kernels");
        let parts = &self.parts;
        let (experts, top_k) = (parts.experts, parts.routing.top_k);
        let routes = tokens * top_k;
        let mut logits = command_buffer.allocate_scratch_for_shape(&[tokens, experts], DataType::F32)?;
        grouped.router_logits.encode(
            input,
            &parts.router.weights,
            parts.router.biases.as_ref(),
            &mut logits,
            tokens,
            parts.model_dim,
            experts,
            command_buffer,
        );

        let mut route_experts = command_buffer.allocate_scratch_for_shape(&[routes], DataType::U32)?;
        let mut route_weights = command_buffer.allocate_scratch_for_shape(&[routes], DataType::F32)?;
        let mut expert_counts = command_buffer.allocate_scratch_for_shape(&[experts], DataType::U32)?;
        let mut expert_routes = command_buffer.allocate_scratch_for_shape(&[experts, tokens], DataType::U32)?;
        let mut active_experts = command_buffer.allocate_scratch_for_shape(&[experts + 1], DataType::U32)?;
        grouped.route_finalize.encode(
            &logits,
            parts.router.expert_biases.as_ref(),
            &mut route_experts,
            &mut route_weights,
            &mut expert_counts,
            &mut expert_routes,
            &mut active_experts,
            tokens,
            experts,
            top_k,
            parts.routing.routed_scaling_factor,
            command_buffer,
        );

        let mut hidden = command_buffer.allocate_scratch_for_shape(&[routes, parts.hidden_dim], DataType::F16)?;
        let activation = parts.activation;
        let mut expert_outputs =
            command_buffer.allocate_scratch_for_shape(&[routes, parts.model_dim], parts.data_type)?;
        match &grouped.experts {
            GroupedExpertKernels::Generic {
                up,
                down,
            } => {
                up.encode(
                    input,
                    &parts.up.codes,
                    &parts.up.scales,
                    parts.up.global_scales.as_ref(),
                    &expert_counts,
                    &expert_routes,
                    &route_weights,
                    &mut hidden,
                    tokens,
                    experts,
                    parts.model_dim,
                    parts.hidden_dim,
                    top_k,
                    activation.up_soft_cap.unwrap_or(1.0),
                    activation.gate_soft_cap.unwrap_or(1.0),
                    activation.alpha,
                    command_buffer,
                );
                down.encode(
                    &hidden,
                    &parts.down.codes,
                    &parts.down.scales,
                    parts.down.global_scales.as_ref(),
                    &expert_counts,
                    &expert_routes,
                    &mut expert_outputs,
                    tokens,
                    experts,
                    parts.model_dim,
                    parts.hidden_dim,
                    command_buffer,
                );
            },
            GroupedExpertKernels::Affine {
                up,
                down,
            } => {
                up.encode(
                    input,
                    &parts.up.codes,
                    &parts.up.scales,
                    &expert_counts,
                    &expert_routes,
                    &route_weights,
                    &mut hidden,
                    tokens,
                    experts,
                    parts.model_dim,
                    parts.hidden_dim,
                    top_k,
                    activation.up_soft_cap.unwrap_or(1.0),
                    activation.gate_soft_cap.unwrap_or(1.0),
                    activation.alpha,
                    command_buffer,
                );
                down.encode(
                    &hidden,
                    &parts.down.codes,
                    &parts.down.scales,
                    &expert_counts,
                    &expert_routes,
                    &mut expert_outputs,
                    tokens,
                    experts,
                    parts.model_dim,
                    parts.hidden_dim,
                    command_buffer,
                );
            },
        }
        grouped.combine.encode(&expert_outputs, output, tokens, parts.model_dim, top_k, command_buffer);
        Ok(())
    }
}

impl<B: Backend> Mlp<B> for MixtureOfExpertsMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (mixture of experts)");
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.parts.model_dim], self.parts.data_type)
            .map_err(MlpEncodeError::Backend)?;
        self.encode_rows(&input, &mut output, batch_dim, command_buffer).map_err(MlpEncodeError::Backend)?;
        Ok(output)
    }
}
