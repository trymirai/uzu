//! MoE (Mixture of Experts) block encodable.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
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
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

struct RouterBlock<B: Backend> {
    weights_buf: Rc<RefCell<B::Buffer>>,
    biases_buf: Rc<RefCell<B::Buffer>>,
}

#[derive(Clone)]
struct SharedMoeWeights<B: Backend> {
    pub w13_buf: Rc<RefCell<B::Buffer>>,
    pub w2_buf: Rc<RefCell<B::Buffer>>,
    pub up_biases_buf: Rc<RefCell<B::Buffer>>,
    pub down_biases_buf: Rc<RefCell<B::Buffer>>,
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
    ParameterLoaderError(#[source] ParameterLoaderError<B>),
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
                let weights_arr = router_tree.leaf_array("weights").map_err(MoeBlockError::ParameterLoaderError)?;
                let biases_arr = router_tree.leaf_array("biases").map_err(MoeBlockError::ParameterLoaderError)?;
                RouterBlock {
                    weights_buf: weights_arr.buffer(),
                    biases_buf: biases_arr.buffer(),
                }
            },
            LinearConfig::QLoRA {
                ..
            } => {
                unimplemented!("QLoRA router not yet supported for MoE");
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

        let w13_arr = experts_tree
            .subtree("up_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf_array("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?;

        let w2_arr = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf_array("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?;

        let up_biases_arr = experts_tree
            .subtree("up_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf_array("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?;

        let down_biases_arr = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf_array("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?;

        let shared_weights = SharedMoeWeights {
            w13_buf: w13_arr.buffer(),
            w2_buf: w2_arr.buffer(),
            up_biases_buf: up_biases_arr.buffer(),
            down_biases_buf: down_biases_arr.buffer(),
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
            ActivationType::TANH => {
                panic!("Tanh activation is not supported for MoE kernels")
            },
            ActivationType::IDENTITY => {
                panic!("Identity activation is not supported for MoE kernels")
            },
        }
    }
}

impl<B: Backend> Mlp<B> for MoeBlock<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let suffix_length = state.active_row_count();

        let main_buf = state.array(ArrayId::Main).buffer();
        let topk_ids_buf = state.array(ArrayId::MoeTopkIds).buffer();
        let topk_probs_buf = state.array(ArrayId::MoeTopkProbs).buffer();
        let offsets_buf = state.array(ArrayId::MoeOffsets).buffer();
        let sumk_buf = state.array(ArrayId::MoeSumK).buffer();
        let bucketed_ids_buf = state.array(ArrayId::MoeBucketedTokenIds).buffer();
        let bucketed_probs_buf = state.array(ArrayId::MoeBucketedProbs).buffer();
        let x_perm_buf = state.array(ArrayId::MoeXPerm).buffer();
        let tok2row_buf = state.array(ArrayId::MoeTok2Row).buffer();
        let y_partial_buf = state.array(ArrayId::MoeYPartial).buffer();
        let tile_counts_buf = state.array(ArrayId::MoeTileCounts).buffer();
        let tile_offsets_buf = state.array(ArrayId::MoeTileOffsets).buffer();
        let tile_map_buf = state.array(ArrayId::MoeTileMap).buffer();
        let total_tiles_buf = state.array(ArrayId::MoeTotalTiles).buffer();
        let dispatch_args_buf = state.array(ArrayId::MoeDispatchArgs).buffer();
        let partials_buf = state.array(ArrayId::MoeScatterPartials).buffer();
        let block_bases_buf = state.array(ArrayId::MoeScatterBlockBases).buffer();
        let block_alloc_buf = state.array(ArrayId::MoeBlockAlloc).buffer();
        let hidden_buf = state.array(ArrayId::MoeHidden).buffer();
        let row_expert_map_buf = state.array(ArrayId::MoeTwoPassRowExpertMap).buffer();

        let e = self.moe_config.num_routed_experts;
        let k = self.moe_config.num_active_routed_experts;
        let k_tile = 128;

        // Clear internal MoE buffers
        if suffix_length > 0 && k > 0 {
            let entries = suffix_length * k;
            let topk_bytes = entries * std::mem::size_of::<u32>();
            let tok2row_bytes = entries * std::mem::size_of::<i32>();

            // Clear topk_ids and tok2row buffers
            if topk_bytes > 0 {
                encoder.encode_fill(topk_ids_buf.borrow_mut().deref_mut(), 0..topk_bytes, 0xFF);
            }
            if tok2row_bytes > 0 {
                encoder.encode_fill(tok2row_buf.borrow_mut().deref_mut(), 0..tok2row_bytes, 0xFF);
            }

            // Clear hidden buffer
            let hidden_bytes = suffix_length * k * self.hidden_dim * self.data_type.size_in_bytes();
            if hidden_bytes > 0 {
                encoder.encode_fill(hidden_buf.borrow_mut().deref_mut(), 0..hidden_bytes, 0);
            }

            // Clear y_partial buffer
            let y_partial_bytes = suffix_length * k * self.model_dim * self.data_type.size_in_bytes();
            if y_partial_bytes > 0 {
                encoder.encode_fill(y_partial_buf.borrow_mut().deref_mut(), 0..y_partial_bytes, 0);
            }

            // Clear x_perm buffer
            let x_perm_bytes = suffix_length * k * self.model_dim * self.data_type.size_in_bytes();
            if x_perm_bytes > 0 {
                encoder.encode_fill(x_perm_buf.borrow_mut().deref_mut(), 0..x_perm_bytes, 0);
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
                main_buf.borrow().deref(),
                self.router.weights_buf.borrow().deref(),
                self.router.biases_buf.borrow().deref(),
                topk_ids_buf.borrow_mut().deref_mut(),
                topk_probs_buf.borrow_mut().deref_mut(),
                suffix_length as u32,
                self.model_dim as u32,
                e as u32,
                k as u32,
                self.router_renorm,
                encoder,
            );
        }

        self.counts_offsets_kernel.encode(
            topk_ids_buf.borrow().deref(),
            offsets_buf.borrow_mut().deref_mut(),
            sumk_buf.borrow_mut().deref_mut(),
            partials_buf.borrow_mut().deref_mut(),
            suffix_length as u32,
            e as u32,
            k as u32,
            encoder,
        );

        let num_blocks = ((suffix_length + 255) / 256).max(1);
        let num_tiles = ((e + 512 - 1) / 512).max(1);
        self.scatter_bases_kernel.encode(
            partials_buf.borrow().deref(),
            block_bases_buf.borrow_mut().deref_mut(),
            block_alloc_buf.borrow_mut().deref_mut(),
            e as u32,
            num_blocks as u32,
            num_tiles as u32,
            0u32,
            encoder,
        );
        self.scatter_map_kernel.encode(
            topk_ids_buf.borrow().deref(),
            topk_probs_buf.borrow().deref(),
            offsets_buf.borrow().deref(),
            block_bases_buf.borrow().deref(),
            block_alloc_buf.borrow().deref(),
            bucketed_ids_buf.borrow_mut().deref_mut(),
            bucketed_probs_buf.borrow_mut().deref_mut(),
            suffix_length as u32,
            e as u32,
            k as u32,
            num_blocks as u32,
            num_tiles as u32,
            tok2row_buf.borrow_mut().deref_mut(),
            encoder,
        );

        self.gather_kernels.encode(
            encoder,
            self.data_type,
            MoeGatherArguments {
                x_buffer: main_buf.borrow().deref(),
                bucketed_ids_buffer: bucketed_ids_buf.borrow().deref(),
                x_perm_buffer: x_perm_buf.borrow_mut().deref_mut(),
                sumk_buffer: sumk_buf.borrow().deref(),
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
            let total_rows = suffix_length * k;
            let num_tiles_k = ((self.hidden_dim + k_tile - 1) / k_tile) as u32;

            self.experts_two_pass_decode_kernel.encode(
                encoder,
                MoeExpertsTwoPassArguments {
                    x_perm_buffer: x_perm_buf.borrow().deref(),
                    expert_offsets: offsets_buf.borrow().deref(),
                    row_expert_map: row_expert_map_buf.borrow_mut().deref_mut(),
                    hidden_buffer: hidden_buf.borrow_mut().deref_mut(),
                    output_buffer: y_partial_buf.borrow_mut().deref_mut(),
                    w13_all: self.shared_weights.w13_buf.borrow().deref(),
                    w2_all: self.shared_weights.w2_buf.borrow().deref(),
                    up_biases: self.shared_weights.up_biases_buf.borrow().deref(),
                    down_biases: self.shared_weights.down_biases_buf.borrow().deref(),
                    tile_counts: tile_counts_buf.borrow_mut().deref_mut(),
                    tile_offsets: tile_offsets_buf.borrow_mut().deref_mut(),
                    tile_map: tile_map_buf.borrow_mut().deref_mut(),
                    total_tiles: total_tiles_buf.borrow_mut().deref_mut(),
                    dispatch_args: dispatch_args_buf.borrow_mut().deref_mut(),
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
            let total_rows = suffix_length * k;
            let num_tiles_k = ((self.hidden_dim + k_tile - 1) / k_tile) as u32;

            let x_perm_borrow = x_perm_buf.borrow();
            let offsets_borrow = offsets_buf.borrow();
            let mut row_expert_map_borrow = row_expert_map_buf.borrow_mut();
            let mut hidden_borrow = hidden_buf.borrow_mut();
            let mut y_partial_borrow = y_partial_buf.borrow_mut();
            let w13_borrow = self.shared_weights.w13_buf.borrow();
            let w2_borrow = self.shared_weights.w2_buf.borrow();
            let up_biases_borrow = self.shared_weights.up_biases_buf.borrow();
            let down_biases_borrow = self.shared_weights.down_biases_buf.borrow();
            let mut tile_counts_borrow = tile_counts_buf.borrow_mut();
            let mut tile_offsets_borrow = tile_offsets_buf.borrow_mut();
            let mut tile_map_borrow = tile_map_buf.borrow_mut();
            let mut total_tiles_borrow = total_tiles_buf.borrow_mut();
            let mut dispatch_args_borrow = dispatch_args_buf.borrow_mut();

            let args = MoeExpertsTwoPassArguments {
                x_perm_buffer: x_perm_borrow.deref(),
                expert_offsets: offsets_borrow.deref(),
                row_expert_map: row_expert_map_borrow.deref_mut(),
                hidden_buffer: hidden_borrow.deref_mut(),
                output_buffer: y_partial_borrow.deref_mut(),
                w13_all: w13_borrow.deref(),
                w2_all: w2_borrow.deref(),
                up_biases: up_biases_borrow.deref(),
                down_biases: down_biases_borrow.deref(),
                tile_counts: tile_counts_borrow.deref_mut(),
                tile_offsets: tile_offsets_borrow.deref_mut(),
                tile_map: tile_map_borrow.deref_mut(),
                total_tiles: total_tiles_borrow.deref_mut(),
                dispatch_args: dispatch_args_borrow.deref_mut(),
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
            tok2row_buf.borrow().deref(),
            topk_probs_buf.borrow().deref(),
            y_partial_buf.borrow().deref(),
            main_buf.borrow_mut().deref_mut(),
            suffix_length as u32,
            self.model_dim as u32,
            k as u32,
            encoder,
        );

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/moe/mod.rs"]
mod tests;
