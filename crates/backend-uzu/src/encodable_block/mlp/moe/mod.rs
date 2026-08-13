//! MoE (Mixture of Experts) block encodable.

mod experts_mxfp4_decode;
mod experts_mxfp4_prefill;
mod experts_two_pass_decode;
mod experts_two_pass_prefill;
mod gather;

use experts_mxfp4_decode::{MoeExpertsMxfp4Arguments, MoeExpertsMxfp4DecodeBlock};
use experts_mxfp4_prefill::MoeExpertsMxfp4PrefillBlock;
use experts_two_pass_decode::MoeExpertsTwoPassDecodeBlock;
use experts_two_pass_prefill::{MoeExpertsTwoPassArguments, MoeExpertsTwoPassPrefillBlock};
use gather::MoeGather;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{
            MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeFinalizeKernel, MoeRouterTopKKernel,
            MoeScatterBucketsMapKernel,
        },
    },
    config::{
        mlp::{mixture_of_experts::MixtureOfExpertsConfig, routing_function::AnyRoutingFunction},
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            full_precision_spec::FullPrecisionSpec,
            microfloat_spec::{MicrofloatScaleMode, MicrofloatSpec},
        },
    },
    data_type::DataType,
    encodable_block::mlp::Mlp,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub struct MoeBlock<B: Backend> {
    router_topk_kernel: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    counts_offsets_kernel: <B::Kernels as Kernels>::MoeCountsOffsetsFusedKernel,
    scatter_bases_kernel: <B::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel,
    scatter_map_kernel: <B::Kernels as Kernels>::MoeScatterBucketsMapKernel,
    gather: MoeGather<B>,
    experts: MoeExperts<B>,
    finalize_kernel: <B::Kernels as Kernels>::MoeFinalizeKernel,
    router_weights: Allocation<B>,
    router_biases: Allocation<B>,
    router_renorm: bool,
    model_dim: usize,
    hidden_dim: usize,
    num_routed_experts: usize,
    num_active_experts: usize,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
    silu_alpha: f32,
    data_type: DataType,
}

/// Expert storage and execution stay coupled so packed weights cannot enter a dense kernel.
enum MoeExperts<B: Backend> {
    FullPrecision {
        decode: MoeExpertsTwoPassDecodeBlock<B>,
        prefill: MoeExpertsTwoPassPrefillBlock<B>,
        w13: Allocation<B>,
        w2: Allocation<B>,
        up_biases: Allocation<B>,
        down_biases: Allocation<B>,
    },
    Mxfp4 {
        decode: MoeExpertsMxfp4DecodeBlock<B>,
        prefill: MoeExpertsMxfp4PrefillBlock<B>,
        w13_blocks: Allocation<B>,
        w13_scales: Allocation<B>,
        w13_global_scale: Allocation<B>,
        w2_blocks: Allocation<B>,
        w2_scales: Allocation<B>,
        w2_global_scale: Allocation<B>,
        up_biases: Allocation<B>,
        down_biases: Allocation<B>,
    },
}

struct MoeExpertArguments<'a, B: Backend> {
    x_perm: &'a Allocation<B>,
    expert_offsets: &'a Allocation<B>,
    total_rows: usize,
    d_model: usize,
    d_ff: usize,
    num_routed_experts: usize,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
    silu_alpha: f32,
}

impl<B: Backend> MoeExperts<B> {
    fn encode(
        &self,
        args: MoeExpertArguments<'_, B>,
        prefill: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match self {
            Self::FullPrecision {
                decode,
                prefill: prefill_block,
                w13,
                w2,
                up_biases,
                down_biases,
            } => {
                let args = MoeExpertsTwoPassArguments {
                    x_perm: args.x_perm,
                    expert_offsets: args.expert_offsets,
                    w13_all: w13,
                    w2_all: w2,
                    up_biases,
                    down_biases,
                    total_rows: args.total_rows,
                    d_model: args.d_model,
                    d_ff: args.d_ff,
                    num_routed_experts: args.num_routed_experts,
                    gate_clip_min: args.gate_clip_min,
                    gate_clip_max: args.gate_clip_max,
                    up_clip_min: args.up_clip_min,
                    up_clip_max: args.up_clip_max,
                    silu_alpha: args.silu_alpha,
                };
                if prefill {
                    return prefill_block.encode(args, encoder);
                }
                decode.encode(args, encoder)
            },
            Self::Mxfp4 {
                decode,
                prefill: prefill_block,
                w13_blocks,
                w13_scales,
                w13_global_scale,
                w2_blocks,
                w2_scales,
                w2_global_scale,
                up_biases,
                down_biases,
            } => {
                let args = MoeExpertsMxfp4Arguments {
                    x_perm: args.x_perm,
                    expert_offsets: args.expert_offsets,
                    w13_blocks,
                    w13_scales,
                    w13_global_scale,
                    w2_blocks,
                    w2_scales,
                    w2_global_scale,
                    up_biases,
                    down_biases,
                    total_rows: args.total_rows,
                    d_model: args.d_model,
                    d_ff: args.d_ff,
                    num_routed_experts: args.num_routed_experts,
                    gate_clip_min: args.gate_clip_min,
                    gate_clip_max: args.gate_clip_max,
                    up_clip_min: args.up_clip_min,
                    up_clip_max: args.up_clip_max,
                    silu_alpha: args.silu_alpha,
                };
                if prefill {
                    return prefill_block.encode(args, encoder);
                }
                decode.encode(args, encoder)
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum MoeBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("MoE requires model_dim % 4 == 0")]
    InvalidModelDim,
    #[error("MoE num_routed_experts must be > 0 and <= 512")]
    InvalidRoutedExpertCount,
    #[error("MoE num_active_routed_experts must be > 0, <= 128, and <= num_routed_experts")]
    InvalidActiveExpertCount,
    #[error("MoE shared experts are not supported")]
    UnsupportedSharedExperts,
    #[error("MoE expert gate is not supported")]
    UnsupportedExpertGate,
    #[error("MoE without router, up, and down biases is not supported")]
    UnsupportedNoBiases,
    #[error("Unsupported MoE router configuration: {0}")]
    UnsupportedRouterConfiguration(String),
    #[error("Unsupported MoE expert activation: {0:?}")]
    UnsupportedExpertActivation(ActivationType),
    #[error("Unsupported MoE expert configuration: {0}")]
    UnsupportedExpertConfiguration(String),
}

impl<B: Backend> MoeBlock<B> {
    pub fn new(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: usize,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, MoeBlockError<B>> {
        if !model_dim.is_multiple_of(4) {
            return Err(MoeBlockError::InvalidModelDim);
        }
        if moe_config.num_routed_experts == 0 || moe_config.num_routed_experts > 512 {
            return Err(MoeBlockError::InvalidRoutedExpertCount);
        }
        if moe_config.num_active_routed_experts == 0
            || moe_config.num_active_routed_experts > 128
            || moe_config.num_active_routed_experts > moe_config.num_routed_experts
        {
            return Err(MoeBlockError::InvalidActiveExpertCount);
        }
        if moe_config.num_shared_experts != 0 {
            return Err(MoeBlockError::UnsupportedSharedExperts);
        }
        if moe_config.gate_config.is_some() {
            return Err(MoeBlockError::UnsupportedExpertGate);
        }
        if !moe_config.router_has_biases
            || !moe_config.expert_config.has_up_biases
            || !moe_config.expert_config.has_down_biases
        {
            return Err(MoeBlockError::UnsupportedNoBiases);
        }

        let gating_code = match moe_config.expert_config.activation.act_type() {
            ActivationType::GELUApprox => 3,
            ActivationType::SILU => 2,
            activation_type => return Err(MoeBlockError::UnsupportedExpertActivation(activation_type)),
        };

        let router_renorm = matches!(moe_config.routing_function, AnyRoutingFunction::SoftmaxRouting(_));

        let router_tree = parameter_tree.subtree("router");
        let router_weights_tree = router_tree.subtree("weights");
        let router_spec = router_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
            layout: Layout::OutputInput,
            ..
        }) = &router_spec
        else {
            return Err(MoeBlockError::UnsupportedRouterConfiguration(format!("{router_spec:?}")));
        };
        let router_weights = router_weights_tree
            .leaf("weights")?
            .validate(&[moe_config.num_routed_experts, model_dim], data_type)?
            .read_allocation()?;
        let router_biases =
            router_tree.leaf("biases")?.validate(&[moe_config.num_routed_experts], data_type)?.read_allocation()?;

        let experts_tree = parameter_tree.subtree("experts");
        let up_tree = experts_tree.subtree("up_projection");
        let down_tree = experts_tree.subtree("down_projection");
        let up_weights_tree = up_tree.subtree("weights");
        let down_weights_tree = down_tree.subtree("weights");
        let up_biases = up_tree
            .leaf("biases")?
            .validate(&[moe_config.num_routed_experts, moe_config.expert_hidden_dim * 2], data_type)?
            .read_allocation()?;
        let down_biases = down_tree
            .leaf("biases")?
            .validate(&[moe_config.num_routed_experts, model_dim], data_type)?
            .read_allocation()?;

        let router_topk_kernel =
            <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, data_type, true, false, false, false, false)
                .map_err(MoeBlockError::BackendError)?;
        let counts_offsets_kernel = MoeCountsOffsetsFusedKernel::new(context).map_err(MoeBlockError::BackendError)?;

        let scatter_bases_kernel = <B::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(context)
            .map_err(MoeBlockError::BackendError)?;
        let scatter_map_kernel = <B::Kernels as Kernels>::MoeScatterBucketsMapKernel::new(context, data_type)
            .map_err(MoeBlockError::BackendError)?;

        let gather = MoeGather::new(context, data_type).map_err(MoeBlockError::BackendError)?;
        let up_spec = up_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let down_spec = down_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let experts = match (up_spec, down_spec) {
            (
                AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                    layout: Layout::OutputInput,
                    ..
                }),
                AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                    layout: Layout::OutputInput,
                    ..
                }),
            ) => {
                let w13 = up_weights_tree
                    .leaf("weights")?
                    .validate(&[moe_config.num_routed_experts, moe_config.expert_hidden_dim * 2, model_dim], data_type)?
                    .read_allocation()?;
                let w2 = down_weights_tree
                    .leaf("weights")?
                    .validate(&[moe_config.num_routed_experts, model_dim, moe_config.expert_hidden_dim], data_type)?
                    .read_allocation()?;
                let decode = MoeExpertsTwoPassDecodeBlock::new(context, data_type, gating_code)
                    .map_err(MoeBlockError::BackendError)?;
                let prefill = MoeExpertsTwoPassPrefillBlock::new(context, data_type, gating_code)
                    .map_err(MoeBlockError::BackendError)?;
                MoeExperts::FullPrecision {
                    decode,
                    prefill,
                    w13,
                    w2,
                    up_biases,
                    down_biases,
                }
            },
            (
                AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
                    bits: 4,
                    group_size: 16,
                    scale_mode: MicrofloatScaleMode::Mxfp4,
                    layout: Layout::OutputInput,
                    ..
                }),
                AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
                    bits: 4,
                    group_size: 32,
                    scale_mode: MicrofloatScaleMode::Mxfp4,
                    layout: Layout::OutputInput,
                    ..
                }),
            ) => {
                if !model_dim.is_multiple_of(16) || !moe_config.expert_hidden_dim.is_multiple_of(32) {
                    return Err(MoeBlockError::UnsupportedExpertConfiguration(format!(
                        "MXFP4 requires model_dim divisible by 16 and hidden_dim divisible by 32, got model_dim={model_dim}, hidden_dim={}",
                        moe_config.expert_hidden_dim
                    )));
                }
                let w13_blocks = up_weights_tree
                    .leaf("weights")?
                    .validate(
                        &[moe_config.num_routed_experts, moe_config.expert_hidden_dim * 2, model_dim / 2],
                        DataType::U8,
                    )?
                    .read_allocation()?;
                let w13_scales = up_weights_tree
                    .leaf("scales")?
                    .validate(
                        &[moe_config.num_routed_experts, moe_config.expert_hidden_dim * 2, model_dim / 16],
                        DataType::U8,
                    )?
                    .read_allocation()?;
                let w13_global_scale = up_weights_tree
                    .leaf("global_scale")?
                    .validate(&[moe_config.num_routed_experts], data_type)?
                    .read_allocation()?;
                let w2_blocks = down_weights_tree
                    .leaf("weights")?
                    .validate(
                        &[moe_config.num_routed_experts, model_dim, moe_config.expert_hidden_dim / 2],
                        DataType::U8,
                    )?
                    .read_allocation()?;
                let w2_scales = down_weights_tree
                    .leaf("scales")?
                    .validate(
                        &[moe_config.num_routed_experts, model_dim, moe_config.expert_hidden_dim / 32],
                        DataType::U8,
                    )?
                    .read_allocation()?;
                let w2_global_scale = down_weights_tree
                    .leaf("global_scale")?
                    .validate(&[moe_config.num_routed_experts], data_type)?
                    .read_allocation()?;
                let decode = MoeExpertsMxfp4DecodeBlock::new(context, data_type, gating_code)
                    .map_err(MoeBlockError::BackendError)?;
                let prefill = MoeExpertsMxfp4PrefillBlock::new(context, data_type, gating_code)
                    .map_err(MoeBlockError::BackendError)?;
                MoeExperts::Mxfp4 {
                    decode,
                    prefill,
                    w13_blocks,
                    w13_scales,
                    w13_global_scale,
                    w2_blocks,
                    w2_scales,
                    w2_global_scale,
                    up_biases,
                    down_biases,
                }
            },
            (up_spec, down_spec) => {
                return Err(MoeBlockError::UnsupportedExpertConfiguration(format!(
                    "up={up_spec:?}, down={down_spec:?}"
                )));
            },
        };
        let finalize_kernel =
            <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type).map_err(MoeBlockError::BackendError)?;

        let (gate_lo, gate_hi) = moe_config.expert_config.gate_clipping.unwrap_or_default();
        let (up_lo, up_hi) = moe_config.expert_config.up_clipping.unwrap_or_default();

        Ok(Self {
            router_topk_kernel,
            counts_offsets_kernel,
            scatter_bases_kernel,
            scatter_map_kernel,
            gather,
            experts,
            finalize_kernel,
            router_weights,
            router_biases,
            router_renorm,
            model_dim,
            hidden_dim: moe_config.expert_hidden_dim,
            num_routed_experts: moe_config.num_routed_experts,
            num_active_experts: moe_config.num_active_routed_experts,
            gate_clip_min: gate_lo.unwrap_or(f32::NEG_INFINITY),
            gate_clip_max: gate_hi.unwrap_or(f32::INFINITY),
            up_clip_min: up_lo.unwrap_or(f32::NEG_INFINITY),
            up_clip_max: up_hi.unwrap_or(f32::INFINITY),
            silu_alpha: moe_config.expert_config.activation.alpha(),
            data_type,
        })
    }
}

impl<B: Backend> Mlp<B> for MoeBlock<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("mlp (moe)");

        let total_rows = batch_dim * self.num_active_experts;
        let num_blocks = batch_dim.div_ceil(256);
        let num_tiles = self.num_routed_experts.div_ceil(512);

        let mut topk_ids =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.num_active_experts], DataType::I32))?;
        let mut topk_probs =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.num_active_experts], self.data_type))?;

        // The constructor guarantees k <= E, so the router writes every top-k slot.
        self.router_topk_kernel.encode(
            &input,
            &self.router_weights,
            Some(&self.router_biases),
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            &mut topk_ids,
            &mut topk_probs,
            batch_dim as u32,
            self.model_dim as u32,
            self.num_routed_experts as u32,
            self.num_active_experts as u32,
            self.router_renorm,
            None::<f32>,
            None::<f32>,
            encoder,
        );

        let mut offsets = encoder.allocate_scratch(size_for_shape(&[self.num_routed_experts + 1], DataType::U32))?;
        let mut sumk = encoder.allocate_scratch(size_for_shape(&[1], DataType::U32))?;
        let scatter_entries = num_blocks * num_tiles * 512;
        let mut partials = encoder.allocate_scratch(size_for_shape(&[scatter_entries], DataType::U32))?;
        self.counts_offsets_kernel.encode(
            &topk_ids,
            &mut offsets,
            &mut sumk,
            &mut partials,
            batch_dim as u32,
            self.num_routed_experts as u32,
            self.num_active_experts as u32,
            encoder,
        );

        let mut block_bases = encoder.allocate_scratch(size_for_shape(&[scatter_entries], DataType::U32))?;
        let mut block_alloc = encoder.allocate_scratch(size_for_shape(&[scatter_entries], DataType::U32))?;
        let mut bucketed_ids = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::I32))?;
        let mut bucketed_probs = encoder.allocate_scratch(size_for_shape(&[total_rows], self.data_type))?;
        let mut tok2row = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::I32))?;

        // Every routed slot is valid here, so scatter initializes the complete reverse map.
        self.scatter_bases_kernel.encode(
            &partials,
            &mut block_bases,
            &mut block_alloc,
            self.num_routed_experts as u32,
            num_blocks as u32,
            num_tiles as u32,
            0u32,
            encoder,
        );
        self.scatter_map_kernel.encode(
            &topk_ids,
            &topk_probs,
            &offsets,
            &block_bases,
            &block_alloc,
            &mut bucketed_ids,
            &mut bucketed_probs,
            batch_dim as u32,
            self.num_routed_experts as u32,
            self.num_active_experts as u32,
            num_blocks as u32,
            num_tiles as u32,
            &mut tok2row,
            encoder,
        );

        let x_perm = self.gather.encode(
            &input,
            &bucketed_ids,
            &sumk,
            batch_dim,
            self.num_active_experts,
            self.model_dim,
            encoder,
        )?;

        let expert_args = MoeExpertArguments {
            x_perm: &x_perm,
            expert_offsets: &offsets,
            total_rows,
            d_model: self.model_dim,
            d_ff: self.hidden_dim,
            num_routed_experts: self.num_routed_experts,
            gate_clip_min: self.gate_clip_min,
            gate_clip_max: self.gate_clip_max,
            up_clip_min: self.up_clip_min,
            up_clip_max: self.up_clip_max,
            silu_alpha: self.silu_alpha,
        };
        let y_partial = self.experts.encode(expert_args, batch_dim > 1, encoder)?;

        let mut output = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        self.finalize_kernel.encode(
            &tok2row,
            &topk_probs,
            &y_partial,
            &mut output,
            batch_dim as u32,
            self.model_dim as u32,
            self.num_active_experts as u32,
            encoder,
        );

        encoder.pop_debug_group();

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/encodable_block/moe/mod.rs"]
mod tests;
