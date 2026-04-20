//! MoE (Mixture of Experts) block encodable.

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{
            MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeFinalizeKernel, MoeRouterTopKKernel,
            MoeScatterBucketsMapKernel,
            moe::{
                MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeBlock, MoeExpertsTwoPassPrefillBlock,
                MoeGatherArguments, MoeGatherKernels,
            },
        },
    },
    config::{LinearConfig, MixtureOfExpertsConfig, RoutingFunctionConfig},
    encodable_block::mlp::Mlp,
    parameters::{ParameterLoaderError, ParameterTree},
};

struct RouterBlock<B: Backend> {
    weights: Allocation<B>,
    biases: Allocation<B>,
}

struct SharedMoeWeights<B: Backend> {
    pub w13: Allocation<B>,
    pub w2: Allocation<B>,
    pub up_biases: Allocation<B>,
    pub down_biases: Allocation<B>,
}

pub struct MoeBlock<B: Backend> {
    router: RouterBlock<B>,
    router_renorm: bool,
    router_topk_kernel: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    counts_offsets_kernel: <B::Kernels as Kernels>::MoeCountsOffsetsFusedKernel,
    scatter_bases_kernel: <B::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel,
    scatter_map_kernel: <B::Kernels as Kernels>::MoeScatterBucketsMapKernel,
    gather_kernels: MoeGatherKernels<B>,
    experts_two_pass_decode_kernel: MoeExpertsTwoPassDecodeBlock<B>,
    experts_two_pass_prefill_kernel: MoeExpertsTwoPassPrefillBlock<B>,
    finalize_kernel: <B::Kernels as Kernels>::MoeFinalizeKernel,
    moe_config: MixtureOfExpertsConfig,
    model_dim: usize,
    hidden_dim: usize,
    data_type: DataType,
    shared_weights: SharedMoeWeights<B>,
}

#[derive(Debug, Error)]
pub enum MoeBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

impl<B: Backend> MoeBlock<B> {
    pub fn new(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: usize,
        hidden_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, MoeBlockError<B>> {
        let data_type: DataType = moe_config.expert_config.linear_config.activation_precision().into();

        let router_data_type: DataType = moe_config.router_config.activation_precision().into();

        let router_renorm = matches!(moe_config.routing_function, RoutingFunctionConfig::SoftmaxRouting);

        let router_tree = parameter_tree.subtree("router").map_err(MoeBlockError::ParameterLoaderError)?;

        let router = match &moe_config.router_config {
            LinearConfig::Quantized(_) => {
                unimplemented!("Quantized router with fused router+topk not yet supported");
            },
            LinearConfig::MLXQuantized(_) => {
                unimplemented!("MLX quantized router with fused router+topk not yet supported");
            },
            LinearConfig::FullPrecision {
                ..
            } => {
                let weights =
                    router_tree.leaf("weights").map_err(MoeBlockError::ParameterLoaderError)?.read_allocation()?;
                let biases =
                    router_tree.leaf("biases").map_err(MoeBlockError::ParameterLoaderError)?.read_allocation()?;
                RouterBlock {
                    weights,
                    biases,
                }
            },
            LinearConfig::QLoRA {
                ..
            } => {
                unimplemented!("QLoRA router not yet supported for MoE");
            },
            LinearConfig::RHTLinearWrapper {
                ..
            } => {
                unimplemented!("RHTLinearWrapper router not yet supported for MoE");
            },
        };

        let router_topk_kernel = <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, router_data_type)
            .map_err(MoeBlockError::BackendError)?;
        let counts_offsets_kernel = MoeCountsOffsetsFusedKernel::new(context).map_err(MoeBlockError::BackendError)?;

        let scatter_bases_kernel = <B::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(context)
            .map_err(MoeBlockError::BackendError)?;
        let scatter_map_kernel = <B::Kernels as Kernels>::MoeScatterBucketsMapKernel::new(context, data_type)
            .map_err(MoeBlockError::BackendError)?;

        let gather_kernels = MoeGatherKernels::new(context).map_err(MoeBlockError::BackendError)?;
        let experts_two_pass_decode_kernel =
            MoeExpertsTwoPassDecodeBlock::new(context).map_err(MoeBlockError::BackendError)?;
        let experts_two_pass_prefill_kernel =
            MoeExpertsTwoPassPrefillBlock::new(context).map_err(MoeBlockError::BackendError)?;
        let finalize_kernel =
            <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type).map_err(MoeBlockError::BackendError)?;

        let experts_tree = parameter_tree.subtree("experts").map_err(MoeBlockError::ParameterLoaderError)?;

        let w13 = experts_tree
            .subtree("up_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .read_allocation()?;

        let w2 = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .read_allocation()?;

        let up_biases = experts_tree
            .subtree("up_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .read_allocation()?;

        let down_biases = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .read_allocation()?;

        let shared_weights = SharedMoeWeights {
            w13,
            w2,
            up_biases,
            down_biases,
        };

        Ok(Self {
            router,
            router_renorm,
            router_topk_kernel,
            counts_offsets_kernel,
            scatter_bases_kernel,
            scatter_map_kernel,
            gather_kernels,
            experts_two_pass_decode_kernel,
            experts_two_pass_prefill_kernel,
            finalize_kernel,
            moe_config: moe_config.clone(),
            model_dim,
            hidden_dim,
            data_type,
            shared_weights,
        })
    }

    fn gating_code_from_activation(activation: &ActivationType) -> u32 {
        match activation {
            ActivationType::GELU => 3,
            ActivationType::SILU {
                ..
            } => 2,
            _ => {
                panic!("{:?} is not supported for MoE kernels", activation)
            },
        }
    }
}

impl<B: Backend> Mlp<B> for MoeBlock<B> {
    fn encode(
        &self,
        context: &B::Context,
        input: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let _ = context;
        let suffix_length = batch_dim;

        let e = self.moe_config.num_routed_experts;
        let k = self.moe_config.num_active_routed_experts;
        let k_tile = 128;
        let total_rows = suffix_length * k;
        let num_blocks = suffix_length.div_ceil(256).max(1);
        let num_tiles = e.div_ceil(512).max(1);

        let mut topk_ids = encoder.allocate_scratch(size_for_shape(&[suffix_length, k], DataType::I32))?;
        let mut topk_probs = encoder.allocate_scratch(size_for_shape(&[suffix_length, k], self.data_type))?;
        let mut offsets = encoder.allocate_scratch(size_for_shape(&[e + 1], DataType::U32))?;
        let mut sumk = encoder.allocate_scratch(size_for_shape(&[1], DataType::U32))?;
        let mut bucketed_ids = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::I32))?;
        let mut bucketed_probs = encoder.allocate_scratch(size_for_shape(&[total_rows], self.data_type))?;
        let mut x_perm = encoder.allocate_scratch(size_for_shape(&[total_rows, self.model_dim], self.data_type))?;
        let mut tok2row = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::I32))?;
        let mut y_partial = encoder.allocate_scratch(size_for_shape(&[total_rows, self.model_dim], self.data_type))?;
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[total_rows, self.hidden_dim], DataType::F32))?;
        let mut row_expert_map = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::U32))?;
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[e], DataType::U32))?;
        let mut tile_offsets = encoder.allocate_scratch(size_for_shape(&[e + 1], DataType::U32))?;
        let h_blocks = self.hidden_dim.div_ceil(4);
        let mut tile_map =
            encoder.allocate_scratch(size_for_shape(&[total_rows * h_blocks.max(1) * 3], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[8], DataType::U32))?;
        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        let scatter_entries = num_blocks * num_tiles * 512;
        let mut partials = encoder.allocate_scratch(size_for_shape(&[scatter_entries], DataType::U32))?;
        let mut block_bases = encoder.allocate_scratch(size_for_shape(&[scatter_entries], DataType::U32))?;
        let mut block_alloc = encoder.allocate_scratch(size_for_shape(&[num_blocks * num_tiles], DataType::U32))?;
        let mut output = encoder.allocate_scratch(size_for_shape(&[suffix_length, self.model_dim], self.data_type))?;

        // Clear internal MoE buffers
        if suffix_length > 0 && k > 0 {
            // Clear topk_ids and tok2row buffers
            let (buffer, range) = topk_ids.as_buffer_range();
            if !range.is_empty() {
                encoder.encode_fill(buffer, range, 0xFF);
            }
            let (buffer, range) = tok2row.as_buffer_range();
            if !range.is_empty() {
                encoder.encode_fill(buffer, range, 0xFF);
            }

            // Clear hidden buffer
            let (buffer, range) = hidden.as_buffer_range();
            if !range.is_empty() {
                encoder.encode_fill(buffer, range, 0);
            }

            // Clear y_partial buffer
            let (buffer, range) = y_partial.as_buffer_range();
            if !range.is_empty() {
                encoder.encode_fill(buffer, range, 0);
            }

            // Clear x_perm buffer
            let (buffer, range) = x_perm.as_buffer_range();
            if !range.is_empty() {
                encoder.encode_fill(buffer, range, 0);
            }
        }

        // Use fused Router+TopK kernel for non-quantized routers
        if self.model_dim % 4 != 0 {
            panic!("MoE fused router+topk failed: {} % 4 != 0", self.model_dim);
        }
        if e > 512 {
            panic!("MoE fused router+topk failed: {e} > 512");
        }
        if k > 128 {
            panic!("MoE fused router+bottomk failed: {k} > 128");
        }

        if suffix_length > 0 && e > 0 && k > 0 {
            // Use the fused router+topk kernel
            self.router_topk_kernel.encode(
                input,
                &self.router.weights,
                &self.router.biases,
                &mut topk_ids,
                &mut topk_probs,
                suffix_length as u32,
                self.model_dim as u32,
                e as u32,
                k as u32,
                self.router_renorm,
                encoder,
            );
        }

        self.counts_offsets_kernel.encode(
            &topk_ids,
            &mut offsets,
            &mut sumk,
            &mut partials,
            suffix_length as u32,
            e as u32,
            k as u32,
            encoder,
        );

        self.scatter_bases_kernel.encode(
            &partials,
            &mut block_bases,
            &mut block_alloc,
            e as u32,
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
            suffix_length as u32,
            e as u32,
            k as u32,
            num_blocks as u32,
            num_tiles as u32,
            &mut tok2row,
            encoder,
        );

        self.gather_kernels.encode(
            encoder,
            self.data_type,
            MoeGatherArguments {
                x: input,
                bucketed_ids: &bucketed_ids,
                x_perm: &mut x_perm,
                sumk: &sumk,
                t: suffix_length,
                k,
                d_model: self.model_dim,
            },
        );

        let gating_code = Self::gating_code_from_activation(&self.moe_config.expert_config.activation.act_type());

        // Compute clipping values and alpha for expert kernels
        let gate_clip_min = self.moe_config.expert_config.gate_clipping[0].unwrap_or(f32::NEG_INFINITY);
        let gate_clip_max = self.moe_config.expert_config.gate_clipping[1].unwrap_or(f32::INFINITY);
        let up_clip_min = self.moe_config.expert_config.up_clipping[0];
        let up_clip_max = self.moe_config.expert_config.up_clipping[1];
        let silu_alpha = self.moe_config.expert_config.activation.alpha();

        if suffix_length == 1 {
            let num_tiles_k = ((self.hidden_dim + k_tile - 1) / k_tile) as u32;

            self.experts_two_pass_decode_kernel.encode(
                encoder,
                MoeExpertsTwoPassArguments {
                    x_perm: &x_perm,
                    expert_offsets: &offsets,
                    row_expert_map: &mut row_expert_map,
                    hidden: &mut hidden,
                    output: &mut y_partial,
                    w13_all: &self.shared_weights.w13,
                    w2_all: &self.shared_weights.w2,
                    up_biases: &self.shared_weights.up_biases,
                    down_biases: &self.shared_weights.down_biases,
                    tile_counts: &mut tile_counts,
                    tile_offsets: &mut tile_offsets,
                    tile_map: &mut tile_map,
                    total_tiles: &mut total_tiles,
                    dispatch_args: &mut dispatch_args,
                    total_rows,
                    d_model: self.model_dim,
                    d_ff: self.hidden_dim,
                    e,
                    num_tiles_k,
                    gating_code,
                    gate_clip_min,
                    gate_clip_max,
                    up_clip_min,
                    up_clip_max,
                    silu_alpha,
                    data_type: self.data_type,
                },
            );
        } else {
            let num_tiles_k = ((self.hidden_dim + k_tile - 1) / k_tile) as u32;
            let args = MoeExpertsTwoPassArguments {
                x_perm: &x_perm,
                expert_offsets: &offsets,
                row_expert_map: &mut row_expert_map,
                hidden: &mut hidden,
                output: &mut y_partial,
                w13_all: &self.shared_weights.w13,
                w2_all: &self.shared_weights.w2,
                up_biases: &self.shared_weights.up_biases,
                down_biases: &self.shared_weights.down_biases,
                tile_counts: &mut tile_counts,
                tile_offsets: &mut tile_offsets,
                tile_map: &mut tile_map,
                total_tiles: &mut total_tiles,
                dispatch_args: &mut dispatch_args,
                total_rows,
                d_model: self.model_dim,
                d_ff: self.hidden_dim,
                e,
                num_tiles_k,
                gating_code,
                gate_clip_min,
                gate_clip_max,
                up_clip_min,
                up_clip_max,
                silu_alpha,
                data_type: self.data_type,
            };
            self.experts_two_pass_prefill_kernel.encode(encoder, args);
        }

        self.finalize_kernel.encode(
            &tok2row,
            &topk_probs,
            &y_partial,
            &mut output,
            suffix_length as u32,
            self.model_dim as u32,
            k as u32,
            encoder,
        );

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/moe/mod.rs"]
mod tests;
