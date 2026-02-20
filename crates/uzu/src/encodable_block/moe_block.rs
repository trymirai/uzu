//! MoE (Mixture of Experts) block encodable.

use std::{cell::RefCell, rc::Rc};

use thiserror::Error;

use crate::{
    Activation, DataType, LinearConfig, MixtureOfExpertsConfig, RoutingFunctionConfig,
    array::Array,
    backends::common::{
        Backend, CommandBuffer, CopyEncoder, Kernels,
        kernel::{
            MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeFinalizeKernel, MoeRouterTopKKernel,
            MoeScatterBucketsMapKernel,
            moe::{
                MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeBlock, MoeExpertsTwoPassPrefillBlock,
                MoeGatherArguments, MoeGatherKernels,
            },
        },
    },
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

struct RouterBlock<B: Backend> {
    weights_buf: B::NativeBuffer,
    biases_buf: B::NativeBuffer,
}

#[derive(Clone)]
struct SharedMoeWeights<B: Backend> {
    pub w13_buf: Rc<B::NativeBuffer>,
    pub w2_buf: Rc<B::NativeBuffer>,
    pub up_biases_buf: Rc<B::NativeBuffer>,
    pub down_biases_buf: Rc<B::NativeBuffer>,
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
    ParameterLoaderError(#[source] ParameterLoaderError),
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
                let weights_arr = router_tree.leaf("weights").map_err(MoeBlockError::ParameterLoaderError)?;
                let weights_buf = weights_arr.buffer();

                let biases_arr = router_tree.leaf("biases").map_err(MoeBlockError::ParameterLoaderError)?;
                let biases_buf = biases_arr.buffer();

                RouterBlock {
                    weights_buf: weights_buf.clone(),
                    biases_buf: biases_buf.clone(),
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
            .leaf("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?;
        let w13_buf = w13_arr.buffer().clone();

        let w2_arr = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("weights")
            .map_err(MoeBlockError::ParameterLoaderError)?;
        let w2_buf = w2_arr.buffer().clone();

        let up_biases_arr = experts_tree
            .subtree("up_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?;
        let up_biases_buf = up_biases_arr.buffer().clone();

        let down_biases_arr = experts_tree
            .subtree("down_projection")
            .map_err(MoeBlockError::ParameterLoaderError)?
            .leaf("biases")
            .map_err(MoeBlockError::ParameterLoaderError)?;
        let down_biases_buf = down_biases_arr.buffer().clone();

        let shared_weights = SharedMoeWeights {
            w13_buf: Rc::new(w13_buf),
            w2_buf: Rc::new(w2_buf),
            up_biases_buf: Rc::new(up_biases_buf),
            down_biases_buf: Rc::new(down_biases_buf),
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

    fn gating_code_from_activation(activation: &Activation) -> u32 {
        match activation {
            Activation::Gelu => 3,
            Activation::SiLU {
                ..
            } => 2,
            Activation::Identity => {
                panic!("Identity activation is not supported for MoE kernels")
            },
        }
    }
}

impl<B: Backend> EncodableBlock<B> for MoeBlock<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        let suffix_length = state.active_suffix_length();
        let arrays = state.arrays(&[
            ArrayId::Main,
            ArrayId::MoeTopkIds,
            ArrayId::MoeTopkProbs,
            ArrayId::MoeOffsets,
            ArrayId::MoeSumK,
            ArrayId::MoeBucketedTokenIds,
            ArrayId::MoeBucketedProbs,
            ArrayId::MoeXPerm,
            ArrayId::MoeTok2Row,
            ArrayId::MoeYPartial,
            ArrayId::MoeTileCounts,
            ArrayId::MoeTileOffsets,
            ArrayId::MoeTileMap,
            ArrayId::MoeTotalTiles,
            ArrayId::MoeDispatchArgs,
            ArrayId::MoeScatterPartials,
            ArrayId::MoeScatterBlockBases,
            ArrayId::MoeBlockAlloc,
            ArrayId::MoeHidden,
            ArrayId::MoeTwoPassRowExpertMap,
        ]);

        let clone_buffer = |array: &RefCell<Array<B>>| -> B::NativeBuffer { array.borrow().buffer().to_owned().into() };

        let mut array_iter = arrays.iter();
        let main_buf = clone_buffer(array_iter.next().unwrap());
        let topk_ids_buf = clone_buffer(array_iter.next().unwrap());
        let topk_probs_buf = clone_buffer(array_iter.next().unwrap());
        let offsets_buf = clone_buffer(array_iter.next().unwrap());
        let sumk_buf = clone_buffer(array_iter.next().unwrap());
        let bucketed_ids_buf = clone_buffer(array_iter.next().unwrap());
        let bucketed_probs_buf = clone_buffer(array_iter.next().unwrap());
        let x_perm_buf = clone_buffer(array_iter.next().unwrap());
        let tok2row_buf = clone_buffer(array_iter.next().unwrap());
        let y_partial_buf = clone_buffer(array_iter.next().unwrap());
        let tile_counts_buf = clone_buffer(array_iter.next().unwrap());
        let tile_offsets_buf = clone_buffer(array_iter.next().unwrap());
        let tile_map_buf = clone_buffer(array_iter.next().unwrap());
        let total_tiles_buf = clone_buffer(array_iter.next().unwrap());
        let dispatch_args_buf = clone_buffer(array_iter.next().unwrap());
        let partials_buf = clone_buffer(array_iter.next().unwrap());
        let block_bases_buf = clone_buffer(array_iter.next().unwrap());
        let block_alloc_buf = clone_buffer(array_iter.next().unwrap());
        let hidden_buf = clone_buffer(array_iter.next().unwrap());
        let row_expert_map_buf = clone_buffer(array_iter.next().unwrap());
        debug_assert!(array_iter.next().is_none());

        let e = self.moe_config.num_routed_experts;
        let k = self.moe_config.num_active_routed_experts;

        let root = command_buffer;
        let k_tile = 128;

        // Clear internal MoE buffers
        root.with_copy_encoder(|encoder| {
            if suffix_length > 0 && k > 0 {
                let entries = suffix_length * k;
                let topk_bytes = entries * std::mem::size_of::<u32>();
                let tok2row_bytes = entries * std::mem::size_of::<i32>();

                // Clear topk_ids and tok2row buffers
                if topk_bytes > 0 {
                    encoder.encode_fill(&topk_ids_buf, 0..topk_bytes, 0xFF);
                }
                if tok2row_bytes > 0 {
                    encoder.encode_fill(&tok2row_buf, 0..tok2row_bytes, 0xFF);
                }

                // Clear hidden buffer
                let hidden_bytes = suffix_length * k * self.hidden_dim * self.data_type.size_in_bytes();
                if hidden_bytes > 0 {
                    encoder.encode_fill(&hidden_buf, 0..hidden_bytes, 0);
                }

                // Clear y_partial buffer
                let y_partial_bytes = suffix_length * k * self.model_dim * self.data_type.size_in_bytes();
                if y_partial_bytes > 0 {
                    encoder.encode_fill(&y_partial_buf, 0..y_partial_bytes, 0);
                }

                // Clear x_perm buffer
                let x_perm_bytes = suffix_length * k * self.model_dim * self.data_type.size_in_bytes();
                if x_perm_bytes > 0 {
                    encoder.encode_fill(&x_perm_buf, 0..x_perm_bytes, 0);
                }
            }
        });

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
            command_buffer.with_compute_encoder(|encoder| {
                // Use the fused router+topk kernel
                self.router_topk_kernel.encode(
                    &main_buf,
                    &self.router.weights_buf,
                    &self.router.biases_buf,
                    &topk_ids_buf,
                    &topk_probs_buf,
                    suffix_length as u32,
                    self.model_dim as u32,
                    e as u32,
                    k as u32,
                    self.router_renorm,
                    &encoder,
                );
            });
        }

        command_buffer.with_compute_encoder(|encoder| {
            self.counts_offsets_kernel.encode(
                &topk_ids_buf,
                &offsets_buf,
                &sumk_buf,
                &partials_buf,
                suffix_length as u32,
                e as u32,
                k as u32,
                &encoder,
            );
        });

        let num_blocks = ((suffix_length + 255) / 256).max(1);
        let num_tiles = ((e + 512 - 1) / 512).max(1);
        command_buffer.with_compute_encoder(|encoder| {
            self.scatter_bases_kernel.encode(
                &partials_buf,
                &block_bases_buf,
                &block_alloc_buf,
                e as u32,
                num_blocks as u32,
                num_tiles as u32,
                0u32,
                &encoder,
            );
        });
        command_buffer.with_compute_encoder(|encoder| {
            self.scatter_map_kernel.encode(
                &topk_ids_buf,
                &topk_probs_buf,
                &offsets_buf,
                &block_bases_buf,
                &block_alloc_buf,
                &bucketed_ids_buf,
                &bucketed_probs_buf,
                suffix_length as u32,
                e as u32,
                k as u32,
                num_blocks as u32,
                num_tiles as u32,
                &tok2row_buf,
                &encoder,
            );
        });

        self.gather_kernels.encode(
            root,
            self.data_type,
            &MoeGatherArguments {
                x_buffer: &main_buf,
                bucketed_ids_buffer: &bucketed_ids_buf,
                x_perm_buffer: &x_perm_buf,
                sumk_buffer: &sumk_buf,
                t: suffix_length,
                k,
                d_model: self.model_dim,
            },
        );

        let gating_code = Self::gating_code_from_activation(&self.moe_config.expert_config.activation);

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
                root,
                &MoeExpertsTwoPassArguments {
                    x_perm_buffer: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    row_expert_map: &row_expert_map_buf,
                    hidden_buffer: &hidden_buf,
                    output_buffer: &y_partial_buf,
                    w13_all: &self.shared_weights.w13_buf,
                    w2_all: &self.shared_weights.w2_buf,
                    up_biases: &self.shared_weights.up_biases_buf,
                    down_biases: &self.shared_weights.down_biases_buf,
                    tile_counts: &tile_counts_buf,
                    tile_offsets: &tile_offsets_buf,
                    tile_map: &tile_map_buf,
                    total_tiles: &total_tiles_buf,
                    dispatch_args: &dispatch_args_buf,
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

            let args = MoeExpertsTwoPassArguments {
                x_perm_buffer: &x_perm_buf,
                expert_offsets: &offsets_buf,
                row_expert_map: &row_expert_map_buf,
                hidden_buffer: &hidden_buf,
                output_buffer: &y_partial_buf,
                w13_all: self.shared_weights.w13_buf.as_ref(),
                w2_all: self.shared_weights.w2_buf.as_ref(),
                up_biases: self.shared_weights.up_biases_buf.as_ref(),
                down_biases: self.shared_weights.down_biases_buf.as_ref(),
                tile_counts: &tile_counts_buf,
                tile_offsets: &tile_offsets_buf,
                tile_map: &tile_map_buf,
                total_tiles: &total_tiles_buf,
                dispatch_args: &dispatch_args_buf,
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
            self.experts_two_pass_prefill_kernel.encode(root, &args);
        }

        command_buffer.with_compute_encoder(|encoder| {
            self.finalize_kernel.encode(
                &tok2row_buf,
                &topk_probs_buf,
                &y_partial_buf,
                &main_buf,
                suffix_length as u32,
                self.model_dim as u32,
                k as u32,
                &encoder,
            );
        });

        if parameters.wait_until_completed {
            command_buffer.submit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        false
    }

    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        _encoder: &B::ComputeEncoder,
    ) {
        unreachable!("MoeBlock does not support shared compute encoder");
    }
}
