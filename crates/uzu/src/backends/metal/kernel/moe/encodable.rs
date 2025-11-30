//! MoE Block Encodable with path selection:
//! - suffix == 1: Single-token fused decode (no scatter/gather)
//! - suffix <= 32: Indirect decode path (GEMV-based)
//! - suffix > 32: Indirect prefill path (GEMM/MMA-based)

use std::{cell::RefCell, rc::Rc};

use metal::CommandBufferRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    MoeCountsOffsetsFusedArguments, MoeCountsOffsetsFusedKernel,
    MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernel,
    MoeExpertsTwoPassPrefillKernel, MoeFinalizeArguments, MoeFinalizeKernel,
    MoeGatherArguments, MoeGatherKernel, MoeRouterTopKArguments,
    MoeRouterTopKKernel, MoeScatterKernels, MoeScatterWithMapArguments,
    MoeSimpleDecodeFusedArguments, MoeSimpleDecodeFusedKernel,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::{
        Activation, LinearConfig, MixtureOfExpertsConfig,
        mlp::RoutingFunctionConfig,
    },
    parameters::ParameterTree,
};

/// Threshold for decode vs prefill path selection
const INDIRECT_DECODE_THRESHOLD: usize = 32;

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

/// Buffers used by MoE encoding paths
struct MoeBuffers {
    main: metal::Buffer,
    topk_ids: metal::Buffer,
    topk_probs: metal::Buffer,
    offsets: metal::Buffer,
    sumk: metal::Buffer,
    bucketed_ids: metal::Buffer,
    bucketed_probs: metal::Buffer,
    x_perm: metal::Buffer,
    tok2row: metal::Buffer,
    y_partial: metal::Buffer,
    tile_counts: metal::Buffer,
    tile_offsets: metal::Buffer,
    tile_map: metal::Buffer,
    total_tiles: metal::Buffer,
    dispatch_args: metal::Buffer,
    partials: metal::Buffer,
    block_bases: metal::Buffer,
    block_alloc: metal::Buffer,
    hidden: metal::Buffer,
    row_expert_map: metal::Buffer,
}

pub struct MoeBlockEncodable {
    // Router
    router: RouterBlock,
    router_data_type: KernelDataType,
    router_renorm: bool,
    router_topk_kernel: MoeRouterTopKKernel,

    // Indirect path shared components
    counts_offsets_kernel: MoeCountsOffsetsFusedKernel,
    scatter_kernels: MoeScatterKernels,
    gather_kernel: MoeGatherKernel,
    finalize_kernel: MoeFinalizeKernel,

    // Indirect path variants
    experts_two_pass_decode_kernel: MoeExpertsTwoPassDecodeKernel,
    experts_two_pass_prefill_kernel: MoeExpertsTwoPassPrefillKernel,

    // Single-token decode path
    simple_decode_fused_kernel: MoeSimpleDecodeFusedKernel,

    // Config and weights
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
            LinearConfig::MLXQuantized(_) => {
                unimplemented!(
                    "MLX quantized router with fused router+topk not yet supported"
                );
            },
            LinearConfig::FullPrecision {
                ..
            } => {
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

        // Initialize kernels
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
        let simple_decode_fused_kernel =
            MoeSimpleDecodeFusedKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "Simple decode fused kernel error: {:?}",
                    e
                ))
            })?;
        let finalize_kernel = MoeFinalizeKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Finalize kernel error: {:?}",
                e
            ))
        })?;

        // Load expert weights
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
            simple_decode_fused_kernel,
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
            Activation::Gelu => 3, // GEGLU
            Activation::SiLU {
                ..
            } => 2, // SwiGLU
            Activation::Identity => {
                panic!("Identity activation is not supported for MoE kernels")
            },
        }
    }

    fn dtype_size(&self) -> usize {
        match self.data_type {
            KernelDataType::BFloat16 | KernelDataType::Float16 => 2,
            KernelDataType::Float32 => 4,
        }
    }

    fn encode_router(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        match &self.router {
            RouterBlock::Metal {
                weights_buf,
                biases_buf,
            } => {
                self.router_topk_kernel
                    .encode(
                        root,
                        self.router_data_type,
                        MoeRouterTopKArguments {
                            input_buffer: &buffers.main,
                            weight_buffer: weights_buf,
                            bias_buffer: biases_buf,
                            topk_ids_buffer: &buffers.topk_ids,
                            topk_probs_buffer: &buffers.topk_probs,
                            t: suffix_length,
                            d_model: self.model_dim,
                            e,
                            k,
                            renorm: self.router_renorm,
                        },
                    )
                    .expect("MoE router+topk failed");
            },
        }
    }

    fn encode_indirect_setup(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        // Counts + offsets
        self.counts_offsets_kernel
            .encode(
                root,
                MoeCountsOffsetsFusedArguments {
                    topk_ids_buffer: &buffers.topk_ids,
                    offsets_buffer: &buffers.offsets,
                    sum_k_buffer: &buffers.sumk,
                    partials_buffer: &buffers.partials,
                    t: suffix_length,
                    e,
                    k,
                },
            )
            .expect("MoE counts+offsets failed");

        let num_blocks = ((suffix_length + 255) / 256).max(1);
        let num_tiles = ((e + 512 - 1) / 512).max(1);

        // Block bases
        self.scatter_kernels
            .encode_block_bases(
                root,
                super::MoeBlockBasesArguments {
                    partials_buffer: &buffers.partials,
                    block_bases_buffer: &buffers.block_bases,
                    block_alloc_buffer: &buffers.block_alloc,
                    e,
                    num_blocks,
                    num_tiles,
                },
            )
            .expect("MoE block bases failed");

        // Scatter with tok2row map
        self.scatter_kernels
            .encode_scatter_with_map(
                root,
                MoeScatterWithMapArguments {
                    base: super::MoeScatterArguments {
                        topk_ids_buffer: &buffers.topk_ids,
                        topk_probs_buffer: &buffers.topk_probs,
                        offsets_buffer: &buffers.offsets,
                        block_bases_buffer: &buffers.block_bases,
                        block_alloc_buffer: &buffers.block_alloc,
                        out_ids_buffer: &buffers.bucketed_ids,
                        out_probs_buffer: &buffers.bucketed_probs,
                        t: suffix_length,
                        e,
                        k,
                        num_blocks,
                        num_tiles,
                    },
                    tok2row_buffer: &buffers.tok2row,
                },
                self.data_type,
            )
            .expect("MoE scatter failed");

        // Gather
        self.gather_kernel
            .encode(
                root,
                self.data_type,
                MoeGatherArguments {
                    x_buffer: &buffers.main,
                    bucketed_ids_buffer: &buffers.bucketed_ids,
                    x_perm_buffer: &buffers.x_perm,
                    sumk_buffer: &buffers.sumk,
                    t: suffix_length,
                    k,
                    d_model: self.model_dim,
                },
            )
            .expect("MoE gather failed");
    }

    fn clear_indirect_buffers(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let k = self.moe_config.num_experts_per_token;
        if k == 0 {
            return;
        }

        let blit_encoder = root.new_blit_command_encoder();
        let entries = suffix_length * k;
        let topk_bytes = entries * std::mem::size_of::<u32>();
        let tok2row_bytes = entries * std::mem::size_of::<i32>();
        let dtype_size = self.dtype_size();

        if topk_bytes > 0 {
            blit_encoder.fill_buffer(
                &buffers.topk_ids,
                metal::NSRange::new(0, topk_bytes as u64),
                0xFF,
            );
        }
        if tok2row_bytes > 0 {
            blit_encoder.fill_buffer(
                &buffers.tok2row,
                metal::NSRange::new(0, tok2row_bytes as u64),
                0xFF,
            );
        }

        let hidden_bytes =
            (suffix_length * k * self.hidden_dim * dtype_size) as u64;
        if hidden_bytes > 0 {
            blit_encoder.fill_buffer(
                &buffers.hidden,
                metal::NSRange::new(0, hidden_bytes),
                0,
            );
        }

        let y_partial_bytes =
            (suffix_length * k * self.model_dim * dtype_size) as u64;
        if y_partial_bytes > 0 {
            blit_encoder.fill_buffer(
                &buffers.y_partial,
                metal::NSRange::new(0, y_partial_bytes),
                0,
            );
        }

        let x_perm_bytes =
            (suffix_length * k * self.model_dim * dtype_size) as u64;
        if x_perm_bytes > 0 {
            blit_encoder.fill_buffer(
                &buffers.x_perm,
                metal::NSRange::new(0, x_perm_bytes),
                0,
            );
        }

        blit_encoder.end_encoding();
    }

    fn encode_single_token_path(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
    ) {
        let k = self.moe_config.num_experts_per_token;
        let gating_code = Self::gating_code_from_activation(
            &self.moe_config.expert_config.activation,
        );

        self.simple_decode_fused_kernel
            .encode(
                root,
                MoeSimpleDecodeFusedArguments {
                    x: &buffers.main,
                    topk_ids: &buffers.topk_ids,
                    topk_probs: &buffers.topk_probs,
                    w13_all: &self.shared_weights.w13_buf,
                    w2_all: &self.shared_weights.w2_buf,
                    up_biases: &self.shared_weights.up_biases_buf,
                    down_biases: &self.shared_weights.down_biases_buf,
                    hidden: &buffers.hidden,
                    y: &buffers.main,
                    d_model: self.model_dim,
                    d_ff: self.hidden_dim,
                    k,
                    gating_code,
                    data_type: self.data_type,
                },
            )
            .expect("MoE single-token fused decode failed");
    }

    fn encode_indirect_decode_path(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;
        let gating_code = Self::gating_code_from_activation(
            &self.moe_config.expert_config.activation,
        );

        let gate_clip_min = self.moe_config.expert_config.gate_clipping[0]
            .unwrap_or(f32::NEG_INFINITY);
        let gate_clip_max = self.moe_config.expert_config.gate_clipping[1]
            .unwrap_or(f32::INFINITY);
        let up_clip_min = self.moe_config.expert_config.up_clipping[0];
        let up_clip_max = self.moe_config.expert_config.up_clipping[1];
        let silu_alpha = self.moe_config.expert_config.activation.alpha();

        let total_rows = suffix_length * k;
        const K_TILE: usize = 64;
        let num_tiles_k = ((self.hidden_dim + K_TILE - 1) / K_TILE) as u32;

        self.experts_two_pass_decode_kernel
            .encode(
                root,
                MoeExpertsTwoPassArguments {
                    x_perm_buffer: &buffers.x_perm,
                    expert_offsets: &buffers.offsets,
                    row_expert_map: &buffers.row_expert_map,
                    hidden_buffer: &buffers.hidden,
                    output_buffer: &buffers.y_partial,
                    w13_all: &self.shared_weights.w13_buf,
                    w2_all: &self.shared_weights.w2_buf,
                    up_biases: &self.shared_weights.up_biases_buf,
                    down_biases: &self.shared_weights.down_biases_buf,
                    tile_counts: &buffers.tile_counts,
                    tile_offsets: &buffers.tile_offsets,
                    tile_map: &buffers.tile_map,
                    total_tiles: &buffers.total_tiles,
                    dispatch_args: &buffers.dispatch_args,
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
            )
            .expect("MoE indirect decode failed");

        self.encode_finalize(root, buffers, suffix_length);
    }

    fn encode_indirect_prefill_path(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;
        let gating_code = Self::gating_code_from_activation(
            &self.moe_config.expert_config.activation,
        );

        let gate_clip_min = self.moe_config.expert_config.gate_clipping[0]
            .unwrap_or(f32::NEG_INFINITY);
        let gate_clip_max = self.moe_config.expert_config.gate_clipping[1]
            .unwrap_or(f32::INFINITY);
        let up_clip_min = self.moe_config.expert_config.up_clipping[0];
        let up_clip_max = self.moe_config.expert_config.up_clipping[1];
        let silu_alpha = self.moe_config.expert_config.activation.alpha();

        let total_rows = suffix_length * k;
        const K_TILE: usize = 128;
        let num_tiles_k = ((self.hidden_dim + K_TILE - 1) / K_TILE) as u32;

        self.experts_two_pass_prefill_kernel
            .encode(
                root,
                MoeExpertsTwoPassArguments {
                    x_perm_buffer: &buffers.x_perm,
                    expert_offsets: &buffers.offsets,
                    row_expert_map: &buffers.row_expert_map,
                    hidden_buffer: &buffers.hidden,
                    output_buffer: &buffers.y_partial,
                    w13_all: &self.shared_weights.w13_buf,
                    w2_all: &self.shared_weights.w2_buf,
                    up_biases: &self.shared_weights.up_biases_buf,
                    down_biases: &self.shared_weights.down_biases_buf,
                    tile_counts: &buffers.tile_counts,
                    tile_offsets: &buffers.tile_offsets,
                    tile_map: &buffers.tile_map,
                    total_tiles: &buffers.total_tiles,
                    dispatch_args: &buffers.dispatch_args,
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
            )
            .expect("MoE indirect prefill failed");

        self.encode_finalize(root, buffers, suffix_length);
    }

    fn encode_finalize(
        &self,
        root: &CommandBufferRef,
        buffers: &MoeBuffers,
        suffix_length: usize,
    ) {
        let k = self.moe_config.num_experts_per_token;

        self.finalize_kernel
            .encode(
                root,
                MoeFinalizeArguments {
                    tok2row_buffer: &buffers.tok2row,
                    probs_buffer: &buffers.topk_probs,
                    y_partial_buffer: &buffers.y_partial,
                    y_out_buffer: &buffers.main,
                    t: suffix_length,
                    d_model: self.model_dim,
                    k,
                },
                self.data_type,
            )
            .expect("MoE finalize failed");
    }
}

impl EncodableWithState for MoeBlockEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
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

        let clone_buffer = |array: &RefCell<MetalArray>| -> metal::Buffer {
            let mut borrow = array.borrow_mut();
            unsafe { borrow.mtl_buffer().clone() }
        };

        let mut array_iter = arrays.iter();
        let buffers = MoeBuffers {
            main: clone_buffer(array_iter.next().unwrap()),
            topk_ids: clone_buffer(array_iter.next().unwrap()),
            topk_probs: clone_buffer(array_iter.next().unwrap()),
            offsets: clone_buffer(array_iter.next().unwrap()),
            sumk: clone_buffer(array_iter.next().unwrap()),
            bucketed_ids: clone_buffer(array_iter.next().unwrap()),
            bucketed_probs: clone_buffer(array_iter.next().unwrap()),
            x_perm: clone_buffer(array_iter.next().unwrap()),
            tok2row: clone_buffer(array_iter.next().unwrap()),
            y_partial: clone_buffer(array_iter.next().unwrap()),
            tile_counts: clone_buffer(array_iter.next().unwrap()),
            tile_offsets: clone_buffer(array_iter.next().unwrap()),
            tile_map: clone_buffer(array_iter.next().unwrap()),
            total_tiles: clone_buffer(array_iter.next().unwrap()),
            dispatch_args: clone_buffer(array_iter.next().unwrap()),
            partials: clone_buffer(array_iter.next().unwrap()),
            block_bases: clone_buffer(array_iter.next().unwrap()),
            block_alloc: clone_buffer(array_iter.next().unwrap()),
            hidden: clone_buffer(array_iter.next().unwrap()),
            row_expert_map: clone_buffer(array_iter.next().unwrap()),
        };
        debug_assert!(array_iter.next().is_none());

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let root = &*mtl_command_buffer;

        // Path selection:
        // - suffix == 1: Single-token fused decode (no scatter/gather)
        // - suffix <= 32: Indirect decode (GEMV)
        // - suffix > 32: Indirect prefill (GEMM)
        if suffix_length == 1 {
            // Single-token path: router -> fused decode
            self.encode_router(root, &buffers, suffix_length);
            self.encode_single_token_path(root, &buffers);
        } else if suffix_length <= INDIRECT_DECODE_THRESHOLD {
            // Indirect decode path: router -> setup -> decode experts -> finalize
            self.clear_indirect_buffers(root, &buffers, suffix_length);
            self.encode_router(root, &buffers, suffix_length);
            self.encode_indirect_setup(root, &buffers, suffix_length);
            self.encode_indirect_decode_path(root, &buffers, suffix_length);
        } else {
            // Indirect prefill path: router -> setup -> prefill experts -> finalize
            self.clear_indirect_buffers(root, &buffers, suffix_length);
            self.encode_router(root, &buffers, suffix_length);
            self.encode_indirect_setup(root, &buffers, suffix_length);
            self.encode_indirect_prefill_path(root, &buffers, suffix_length);
        }

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        false // MoE uses blit encoders + complex multi-kernel pattern
    }
}
