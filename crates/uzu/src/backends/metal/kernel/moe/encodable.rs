use std::{cell::RefCell, rc::Rc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    MoeCountsOffsetsFusedArguments, MoeCountsOffsetsFusedKernel,
    MoeExpertsArguments, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    MoeFinalizeKernel, MoeGatherArguments, MoeGatherKernel,
    MoeRouterTopKArguments, MoeRouterTopKKernel, MoeScatterKernels,
    MoeScatterWithMapArguments,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassState, MOE_TWO_PASS_K_TILE,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::{
        mlp::RoutingFunctionConfig, Activation, LinearConfig,
        MixtureOfExpertsConfig,
    },
    parameters::ParameterTree,
};

enum RouterBlock {
    Metal {
        weights_buf: metal::Buffer,
        biases_buf: metal::Buffer,
    },
}

#[derive(Clone)]
pub struct SharedMoeWeights {
    pub w13_buf: Rc<metal::Buffer>,
    pub w2_buf: Rc<metal::Buffer>,
    pub up_biases_buf: Rc<metal::Buffer>,
    pub down_biases_buf: Rc<metal::Buffer>,
}

pub struct MoeBlockEncodable {
    router: RouterBlock,
    router_data_type: KernelDataType,
    router_renorm: bool,
    router_topk_kernel: MoeRouterTopKKernel,
    counts_offsets_kernel: MoeCountsOffsetsFusedKernel,
    scatter_kernels: MoeScatterKernels,
    gather_kernel: MoeGatherKernel,
    experts_two_pass_decode_kernel: MoeExpertsTwoPassDecodeKernel,
    experts_two_pass_prefill_kernel: MoeExpertsTwoPassPrefillKernel,
    finalize_kernel: MoeFinalizeKernel,
    moe_config: MixtureOfExpertsConfig,
    model_dim: usize,
    hidden_dim: usize,
    data_type: KernelDataType,
    shared_weights: SharedMoeWeights,
}

impl MoeBlockEncodable {
    pub fn new(
        context: &MTLContext,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: usize,
        hidden_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, crate::backends::metal::MTLError> {
        let activation_data_type: DataType = moe_config
            .expert_config
            .linear_config
            .activation_precision()
            .into();
        let data_type: KernelDataType = activation_data_type.into();

        let router_data_type: DataType =
            moe_config.router_config.activation_precision().into();
        let router_kernel_data_type: KernelDataType = router_data_type.into();

        let router_renorm = matches!(
            moe_config.routing_function,
            RoutingFunctionConfig::SoftmaxRouting
        );

        let router_tree = parameter_tree.subtree("router").map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Router subtree error: {:?}",
                e
            ))
        })?;

        let router = match &moe_config.router_config {
            LinearConfig::Quantized(_) => {
                unimplemented!(
                    "Quantized router with fused router+topk not yet supported"
                );
            },
            LinearConfig::FullPrecision {
                ..
            } => {
                // Load router weights directly (already transposed)
                let mut weights_arr =
                    router_tree.leaf("weights").map_err(|e| {
                        crate::backends::metal::MTLError::Generic(format!(
                            "Router weights error: {:?}",
                            e
                        ))
                    })?;
                let weights_buf = unsafe { weights_arr.mtl_buffer().clone() };

                let mut biases_arr =
                    router_tree.leaf("biases").map_err(|e| {
                        crate::backends::metal::MTLError::Generic(format!(
                            "Router biases error: {:?}",
                            e
                        ))
                    })?;
                let biases_buf = unsafe { biases_arr.mtl_buffer().clone() };

                RouterBlock::Metal {
                    weights_buf,
                    biases_buf,
                }
            },
            LinearConfig::QLoRA {
                ..
            } => {
                unimplemented!("QLoRA router not yet supported for MoE");
            },
        };

        let router_topk_kernel =
            MoeRouterTopKKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "RouterTopK fused kernel error: {:?}",
                    e
                ))
            })?;
        let counts_offsets_kernel = MoeCountsOffsetsFusedKernel::new(context)
            .map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Counts+offsets kernel error: {:?}",
                e
            ))
        })?;
        let scatter_kernels = MoeScatterKernels::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Scatter kernels error: {:?}",
                e
            ))
        })?;
        let gather_kernel = MoeGatherKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Gather kernel error: {:?}",
                e
            ))
        })?;
        let experts_two_pass_decode_kernel =
            MoeExpertsTwoPassDecodeKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "Experts two-pass decode kernel error: {:?}",
                    e
                ))
            })?;
        let experts_two_pass_prefill_kernel =
            MoeExpertsTwoPassPrefillKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "Experts two-pass prefill kernel error: {:?}",
                    e
                ))
            })?;
        let finalize_kernel = MoeFinalizeKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Finalize kernel error: {:?}",
                e
            ))
        })?;

        // Load expert weights directly from parameter tree (already transposed)
        let experts_tree = parameter_tree.subtree("experts").map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "experts subtree error: {:?}",
                e
            ))
        })?;

        let mut w13_arr = experts_tree
            .subtree("up_projection")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "up_projection error: {:?}",
                    e
                ))
            })?
            .leaf("weights")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "up weights error: {:?}",
                    e
                ))
            })?;
        let w13_buf = unsafe { w13_arr.mtl_buffer().clone() };

        let mut w2_arr = experts_tree
            .subtree("down_projection")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "down_projection error: {:?}",
                    e
                ))
            })?
            .leaf("weights")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "down weights error: {:?}",
                    e
                ))
            })?;
        let w2_buf = unsafe { w2_arr.mtl_buffer().clone() };

        let mut up_biases_arr = experts_tree
            .subtree("up_projection")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "up_projection biases error: {:?}",
                    e
                ))
            })?
            .leaf("biases")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "up biases error: {:?}",
                    e
                ))
            })?;
        let up_biases_buf = unsafe { up_biases_arr.mtl_buffer().clone() };

        let mut down_biases_arr = experts_tree
            .subtree("down_projection")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "down_projection biases error: {:?}",
                    e
                ))
            })?
            .leaf("biases")
            .map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "down biases error: {:?}",
                    e
                ))
            })?;
        let down_biases_buf = unsafe { down_biases_arr.mtl_buffer().clone() };

        let shared_weights = SharedMoeWeights {
            w13_buf: Rc::new(w13_buf),
            w2_buf: Rc::new(w2_buf),
            up_biases_buf: Rc::new(up_biases_buf),
            down_biases_buf: Rc::new(down_biases_buf),
        };

        Ok(Self {
            router,
            router_data_type: router_kernel_data_type,
            router_renorm,
            router_topk_kernel,
            counts_offsets_kernel,
            scatter_kernels,
            gather_kernel,
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
            Activation::GELU => 3,
            Activation::SILU {
                ..
            } => 2,
        }
    }
}

impl EncodableWithState for MoeBlockEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let suffix_length = state.aux_buffers_suffix_length();
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
            ArrayId::MoeTwoPassPartial,
            ArrayId::MoeTwoPassRowExpertMap,
        ]);

        let clone_buffer = |array: &RefCell<MetalArray>| -> metal::Buffer {
            let mut borrow = array.borrow_mut();
            let buffer = unsafe { borrow.mtl_buffer().clone() };
            buffer
        };

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
        let two_pass_partial_buf = clone_buffer(array_iter.next().unwrap());
        let row_expert_map_buf = clone_buffer(array_iter.next().unwrap());
        debug_assert!(array_iter.next().is_none());

        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let root = &*mtl_command_buffer;

        // Clear all internal MoE buffers
        let dtype_size = match self.data_type {
            KernelDataType::BFloat16 | KernelDataType::Float16 => 2,
            KernelDataType::Float32 => 4,
        };

        let blit_encoder = root.new_blit_command_encoder();

        if suffix_length > 0 && k > 0 {
            let entries = suffix_length * k;
            let topk_bytes = entries * std::mem::size_of::<u32>();
            let tok2row_bytes = entries * std::mem::size_of::<i32>();

            // Clear topk_ids and tok2row buffers
            if topk_bytes > 0 {
                blit_encoder.fill_buffer(
                    &topk_ids_buf,
                    metal::NSRange::new(0, topk_bytes as u64),
                    0xFF,
                );
            }
            if tok2row_bytes > 0 {
                blit_encoder.fill_buffer(
                    &tok2row_buf,
                    metal::NSRange::new(0, tok2row_bytes as u64),
                    0xFF,
                );
            }

            // Clear hidden buffer
            let hidden_bytes =
                (suffix_length * k * self.hidden_dim * dtype_size) as u64;
            if hidden_bytes > 0 {
                blit_encoder.fill_buffer(
                    &hidden_buf,
                    metal::NSRange::new(0, hidden_bytes),
                    0,
                );
            }

            // Clear two-pass partial buffer (f32 accumulator)
            let two_pass_partial_bytes =
                (suffix_length * k * self.model_dim * 4) as u64;
            if two_pass_partial_bytes > 0 {
                blit_encoder.fill_buffer(
                    &two_pass_partial_buf,
                    metal::NSRange::new(0, two_pass_partial_bytes),
                    0,
                );
            }

            // Clear y_partial buffer
            let y_partial_bytes =
                (suffix_length * k * self.model_dim * dtype_size) as u64;
            if y_partial_bytes > 0 {
                blit_encoder.fill_buffer(
                    &y_partial_buf,
                    metal::NSRange::new(0, y_partial_bytes),
                    0,
                );
            }

            // Clear x_perm buffer
            let x_perm_bytes =
                (suffix_length * k * self.model_dim * dtype_size) as u64;
            if x_perm_bytes > 0 {
                blit_encoder.fill_buffer(
                    &x_perm_buf,
                    metal::NSRange::new(0, x_perm_bytes),
                    0,
                );
            }
        }

        blit_encoder.end_encoding();

        // Use fused Router+TopK kernel for non-quantized routers
        match &self.router {
            RouterBlock::Metal {
                weights_buf,
                biases_buf,
            } => {
                // Use the fused router+topk kernel
                self.router_topk_kernel
                    .encode(
                        root,
                        self.router_data_type,
                        MoeRouterTopKArguments {
                            input_buffer: &main_buf,
                            weight_buffer: weights_buf,
                            bias_buffer: biases_buf,
                            topk_ids_buffer: &topk_ids_buf,
                            topk_probs_buffer: &topk_probs_buf,
                            t: suffix_length,
                            d_model: self.model_dim,
                            e,
                            k,
                            renorm: self.router_renorm,
                        },
                    )
                    .expect("MoE fused router+topk failed");
            },
        }

        self.counts_offsets_kernel
            .encode(
                root,
                MoeCountsOffsetsFusedArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    offsets_buffer: &offsets_buf,
                    sum_k_buffer: &sumk_buf,
                    partials_buffer: &partials_buf,
                    t: suffix_length,
                    e,
                    k,
                },
            )
            .expect("MoE counts+offsets failed");

        let num_blocks = ((suffix_length + 255) / 256).max(1);
        let num_tiles = ((e + 512 - 1) / 512).max(1);

        self.scatter_kernels
            .encode_block_bases(
                root,
                super::MoeBlockBasesArguments {
                    partials_buffer: &partials_buf,
                    block_bases_buffer: &block_bases_buf,
                    block_alloc_buffer: &block_alloc_buf,
                    e,
                    num_blocks,
                    num_tiles,
                },
            )
            .expect("MoE block bases failed");

        self.scatter_kernels
            .encode_scatter_with_map(
                root,
                MoeScatterWithMapArguments {
                    base: super::MoeScatterArguments {
                        topk_ids_buffer: &topk_ids_buf,
                        topk_probs_buffer: &topk_probs_buf,
                        offsets_buffer: &offsets_buf,
                        block_bases_buffer: &block_bases_buf,
                        block_alloc_buffer: &block_alloc_buf,
                        out_ids_buffer: &bucketed_ids_buf,
                        out_probs_buffer: &bucketed_probs_buf,
                        t: suffix_length,
                        e,
                        k,
                        num_blocks,
                        num_tiles,
                    },
                    tok2row_buffer: &tok2row_buf,
                },
                self.data_type,
            )
            .expect("MoE scatter failed");

        self.gather_kernel
            .encode(
                root,
                self.data_type,
                MoeGatherArguments {
                    x_buffer: &main_buf,
                    bucketed_ids_buffer: &bucketed_ids_buf,
                    x_perm_buffer: &x_perm_buf,
                    sumk_buffer: &sumk_buf,
                    t: suffix_length,
                    k,
                    d_model: self.model_dim,
                },
            )
            .expect("MoE gather failed");

        let gating_code = Self::gating_code_from_activation(
            &self.moe_config.expert_config.activation,
        );
        const BN: usize = 64;
        let num_tiles_n = (self.model_dim + BN - 1) / BN;

        let w2_buf = &self.shared_weights.w2_buf;

        let experts_args = MoeExpertsArguments {
            x_perm_buffer: &x_perm_buf,
            expert_offsets: &offsets_buf,
            w13_all: &self.shared_weights.w13_buf,
            w2_all: w2_buf,
            y_partial: &y_partial_buf,
            up_biases: &self.shared_weights.up_biases_buf,
            down_biases: &self.shared_weights.down_biases_buf,
            tile_counts: &tile_counts_buf,
            tile_row_offsets: &tile_offsets_buf,
            tile_map: &tile_map_buf,
            total_tiles: &total_tiles_buf,
            dispatch_args: &dispatch_args_buf,
            num_tiles_n,
            t: suffix_length,
            d_model: self.model_dim,
            d_ff: self.hidden_dim,
            e,
            k,
            gating_code,
            gate_clip_min: self.moe_config.expert_config.gate_clipping[0]
                .unwrap_or(f32::NEG_INFINITY),
            gate_clip_max: self.moe_config.expert_config.gate_clipping[1]
                .unwrap_or(f32::INFINITY),
            up_clip_min: self.moe_config.expert_config.up_clipping[0],
            up_clip_max: self.moe_config.expert_config.up_clipping[1],
            silu_alpha: self.moe_config.expert_config.activation.alpha(),
            data_type: self.data_type,
        };

        if suffix_length == 1 {
            let total_rows = suffix_length * k;
            let num_tiles_k = ((self.hidden_dim + MOE_TWO_PASS_K_TILE - 1)
                / MOE_TWO_PASS_K_TILE) as u32;

            self.experts_two_pass_decode_kernel
                .encode(
                    root,
                    MoeExpertsTwoPassArguments {
                        x_perm_buffer: &x_perm_buf,
                        expert_offsets: &offsets_buf,
                        row_expert_map: &row_expert_map_buf,
                        hidden_buffer: &hidden_buf,
                        partial_buffer: &two_pass_partial_buf,
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
                        gate_clip_min: experts_args.gate_clip_min,
                        gate_clip_max: experts_args.gate_clip_max,
                        up_clip_min: experts_args.up_clip_min,
                        up_clip_max: experts_args.up_clip_max,
                        silu_alpha: experts_args.silu_alpha,
                        data_type: self.data_type,
                    },
                )
                .expect("MoE experts two-pass failed");
        } else {
            let total_rows = suffix_length * k;
            let num_tiles_k = ((self.hidden_dim + MOE_TWO_PASS_K_TILE - 1)
                / MOE_TWO_PASS_K_TILE) as u32;

            self.experts_two_pass_prefill_kernel
                .encode(
                    root,
                    MoeExpertsTwoPassArguments {
                        x_perm_buffer: &x_perm_buf,
                        expert_offsets: &offsets_buf,
                        row_expert_map: &row_expert_map_buf,
                        hidden_buffer: &hidden_buf,
                        partial_buffer: &two_pass_partial_buf,
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
                        gate_clip_min: experts_args.gate_clip_min,
                        gate_clip_max: experts_args.gate_clip_max,
                        up_clip_min: experts_args.up_clip_min,
                        up_clip_max: experts_args.up_clip_max,
                        silu_alpha: experts_args.silu_alpha,
                        data_type: self.data_type,
                    },
                )
                .expect("MoE experts two-pass prefill failed");
        }

        self.finalize_kernel
            .encode(
                root,
                super::MoeFinalizeArguments {
                    tok2row_buffer: &tok2row_buf,
                    probs_buffer: &topk_probs_buf,
                    y_partial_buffer: &y_partial_buf,
                    y_out_buffer: &main_buf,
                    t: suffix_length,
                    d_model: self.model_dim,
                    k,
                },
                self.data_type,
            )
            .expect("MoE finalize failed");

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
