use std::{cell::RefCell, mem::size_of, rc::Rc};

use half::{bf16, f16};
use metal::MTLResourceOptions;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    MoeBucketCountsArguments, MoeBucketCountsKernel, MoeExpertsArguments,
    MoeExpertsKernel, MoeFinalizeKernel, MoeGatherArguments, MoeGatherKernel,
    MoeOffsetsScanArguments, MoeOffsetsScanKernel, MoeScatterKernels,
    MoeScatterWithMapArguments, MoeTopKArguments, MoeTopKKernel,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
        kernel::linear::QuantizedLinearKernelBlock,
    },
    config::{Activation, LinearConfig, MixtureOfExpertsConfig},
    device::Array,
    parameters::ParameterTree,
};

enum RouterBlock {
    Quantized(QuantizedLinearKernelBlock),
    Metal {
        pipeline: metal::ComputePipelineState,
        weights_buf: metal::Buffer,
        biases_buf: metal::Buffer,
        mixture_size: usize,
        model_dim: usize,
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
    topk_kernel: MoeTopKKernel,
    counts_kernel: MoeBucketCountsKernel,
    offsets_kernel: MoeOffsetsScanKernel,
    scatter_kernels: MoeScatterKernels,
    gather_kernel: MoeGatherKernel,
    experts_kernel: MoeExpertsKernel,
    finalize_kernel: MoeFinalizeKernel,
    moe_config: MixtureOfExpertsConfig,
    model_dim: usize,
    hidden_dim: usize,
    data_type: KernelDataType,
    shared_weights: SharedMoeWeights,
    layer_index: usize,
}

impl MoeBlockEncodable {
    pub fn load_shared_expert_weights(
        _context: &MTLContext,
        _moe_config: &MixtureOfExpertsConfig,
        _model_dim: usize,
        hidden_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<SharedMoeWeights, crate::backends::metal::MTLError> {
        let experts_tree = parameter_tree.subtree("experts").map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "experts subtree error: {:?}",
                e
            ))
        })?;

        let mut up_arr = experts_tree
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

        let up_shape = up_arr.shape().to_vec();
        let fused_up_weights = up_shape[2] == hidden_dim * 2;

        if !fused_up_weights {
            return Err(crate::backends::metal::MTLError::Generic(
                "MoE experts require fused gate+up weights (shape [E, d_model, 2*d_ff])"
                    .to_string(),
            ));
        }

        let w13_src = unsafe { up_arr.mtl_buffer().clone() };

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
        let w2_src = unsafe { w2_arr.mtl_buffer().clone() };

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
        let up_biases_src = unsafe { up_biases_arr.mtl_buffer().clone() };

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
        let down_biases_src = unsafe { down_biases_arr.mtl_buffer().clone() };

        let w13_buf = w13_src;
        let w2_buf = w2_src;
        let up_biases_buf = up_biases_src;
        let down_biases_buf = down_biases_src;

        Ok(SharedMoeWeights {
            w13_buf: Rc::new(w13_buf),
            w2_buf: Rc::new(w2_buf),
            up_biases_buf: Rc::new(up_biases_buf),
            down_biases_buf: Rc::new(down_biases_buf),
        })
    }

    pub fn new(
        context: &MTLContext,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: usize,
        hidden_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        shared_weights: Option<SharedMoeWeights>,
        layer_index: usize,
    ) -> Result<Self, crate::backends::metal::MTLError> {
        let activation_data_type: DataType = moe_config
            .expert_config
            .linear_config
            .activation_precision()
            .into();
        let data_type: KernelDataType = activation_data_type.into();

        let router_tree = parameter_tree.subtree("router").map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Router subtree error: {:?}",
                e
            ))
        })?;

        let router = match &moe_config.router_config {
            LinearConfig::Quantized(quant_config) => {
                RouterBlock::Quantized(QuantizedLinearKernelBlock::new(
                    context,
                    quant_config,
                    model_dim,
                    moe_config.mixture_size,
                    &router_tree,
                    ArrayId::Main,
                    ArrayId::MoeRouterLogits,
                )?)
            },
            LinearConfig::FullPrecision {
                ..
            } => {
                let kernel_name = match activation_data_type {
                    DataType::BF16 => "moe_router_bf16",
                    DataType::F16 => "moe_router_f16",
                    _ => "moe_router_bf16",
                };
                let pipeline =
                    context.compute_pipeline_state(kernel_name, None)?;

                let mut weights_arr =
                    router_tree.leaf("weights").map_err(|e| {
                        crate::backends::metal::MTLError::Generic(format!(
                            "Router weights error: {:?}",
                            e
                        ))
                    })?;
                let weights_shape = weights_arr.shape().to_owned();
                if weights_shape.len() != 2 {
                    return Err(crate::backends::metal::MTLError::Generic(
                        format!(
                            "Router weights expected 2D tensor, got shape {:?}",
                            weights_shape
                        ),
                    ));
                }
                let rows = weights_shape[0];
                let cols = weights_shape[1];
                let expected_lalamo_layout =
                    (model_dim, moe_config.mixture_size);
                let expected_kernel_layout =
                    (moe_config.mixture_size, model_dim);

                let weights_buf = if (rows, cols) == expected_kernel_layout {
                    unsafe { weights_arr.mtl_buffer().clone() }
                } else if (rows, cols) == expected_lalamo_layout {
                    let transpose_err =
                        |err: crate::device::array::ArrayConversionError| {
                            crate::backends::metal::MTLError::Generic(format!(
                                "Router weights transpose failed: {}",
                                err
                            ))
                        };

                    match weights_arr.data_type() {
                        DataType::BF16 => {
                            let src = weights_arr
                                .as_slice::<bf16>()
                                .map_err(transpose_err)?;
                            let transposed =
                                transpose_router_weights(src, rows, cols);
                            context.device.new_buffer_with_data(
                                transposed.as_ptr() as *const _,
                                (transposed.len() * size_of::<bf16>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        },
                        DataType::F16 => {
                            let src = weights_arr
                                .as_slice::<f16>()
                                .map_err(transpose_err)?;
                            let transposed =
                                transpose_router_weights(src, rows, cols);
                            context.device.new_buffer_with_data(
                                transposed.as_ptr() as *const _,
                                (transposed.len() * size_of::<f16>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        },
                        DataType::F32 => {
                            let src = weights_arr
                                .as_slice::<f32>()
                                .map_err(transpose_err)?;
                            let transposed =
                                transpose_router_weights(src, rows, cols);
                            context.device.new_buffer_with_data(
                                transposed.as_ptr() as *const _,
                                (transposed.len() * size_of::<f32>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        },
                        other => {
                            return Err(
                                crate::backends::metal::MTLError::Generic(
                                    format!(
                                        "Unsupported router weight dtype {:?}",
                                        other
                                    ),
                                ),
                            );
                        },
                    }
                } else {
                    return Err(crate::backends::metal::MTLError::Generic(
                        format!(
                            "Unexpected router weight shape {:?}, expected {:?} or {:?}",
                            weights_shape,
                            expected_lalamo_layout,
                            expected_kernel_layout
                        ),
                    ));
                };

                let mut biases_arr =
                    router_tree.leaf("biases").map_err(|e| {
                        crate::backends::metal::MTLError::Generic(format!(
                            "Router biases error: {:?}",
                            e
                        ))
                    })?;
                let biases_buf = unsafe { biases_arr.mtl_buffer().clone() };

                RouterBlock::Metal {
                    pipeline,
                    weights_buf,
                    biases_buf,
                    mixture_size: moe_config.mixture_size,
                    model_dim,
                }
            },
            LinearConfig::QLoRA {
                ..
            } => {
                return Err(crate::backends::metal::MTLError::Generic(
                    "QLoRA router not yet supported for MoE".to_string(),
                ));
            },
        };

        let topk_kernel = MoeTopKKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "TopK kernel error: {:?}",
                e
            ))
        })?;
        let counts_kernel =
            MoeBucketCountsKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "Counts kernel error: {:?}",
                    e
                ))
            })?;
        let offsets_kernel =
            MoeOffsetsScanKernel::new(context).map_err(|e| {
                crate::backends::metal::MTLError::Generic(format!(
                    "Offsets kernel error: {:?}",
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
        let experts_kernel = MoeExpertsKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Experts kernel error: {:?}",
                e
            ))
        })?;
        let finalize_kernel = MoeFinalizeKernel::new(context).map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!(
                "Finalize kernel error: {:?}",
                e
            ))
        })?;

        let shared_weights = if let Some(weights) = shared_weights {
            weights
        } else {
            Self::load_shared_expert_weights(
                context,
                moe_config,
                model_dim,
                hidden_dim,
                parameter_tree,
            )?
        };

        Ok(Self {
            router,
            topk_kernel,
            counts_kernel,
            offsets_kernel,
            scatter_kernels,
            gather_kernel,
            experts_kernel,
            finalize_kernel,
            moe_config: moe_config.clone(),
            model_dim,
            hidden_dim,
            data_type,
            shared_weights,
            layer_index,
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

fn transpose_router_weights<T>(
    src: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T>
where
    T: Copy + Default,
{
    debug_assert_eq!(rows * cols, src.len());
    let mut dst = vec![T::default(); src.len()];
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            dst[c * rows + r] = src[row_offset + c];
        }
    }
    dst
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
            ArrayId::MoeRouterLogits,
            ArrayId::MoeTopkIds,
            ArrayId::MoeTopkProbs,
            ArrayId::MoeCounts,
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
        ]);

        let clone_buffer = |array: &RefCell<MetalArray>| -> metal::Buffer {
            let mut borrow = array.borrow_mut();
            let buffer = unsafe { borrow.mtl_buffer().clone() };
            buffer
        };

        let main_buf = clone_buffer(&arrays[0]);
        let router_logits_buf = clone_buffer(&arrays[1]);
        let topk_ids_buf = clone_buffer(&arrays[2]);
        let topk_probs_buf = clone_buffer(&arrays[3]);
        let counts_buf = clone_buffer(&arrays[4]);
        let offsets_buf = clone_buffer(&arrays[5]);
        let sumk_buf = clone_buffer(&arrays[6]);
        let bucketed_ids_buf = clone_buffer(&arrays[7]);
        let bucketed_probs_buf = clone_buffer(&arrays[8]);
        let x_perm_buf = clone_buffer(&arrays[9]);
        let tok2row_buf = clone_buffer(&arrays[10]);
        let y_partial_buf = clone_buffer(&arrays[11]);
        let tile_counts_buf = clone_buffer(&arrays[12]);
        let tile_offsets_buf = clone_buffer(&arrays[13]);
        let tile_map_buf = clone_buffer(&arrays[14]);
        let total_tiles_buf = clone_buffer(&arrays[15]);
        let dispatch_args_buf = clone_buffer(&arrays[16]);
        let partials_buf = clone_buffer(&arrays[17]);
        let block_bases_buf = clone_buffer(&arrays[18]);
        let block_alloc_buf = clone_buffer(&arrays[19]);

        let debug_moe = std::env::var_os("UZU_DEBUG_MOE_STATE").is_some();

        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        if suffix_length > 0 && k > 0 {
            let entries = suffix_length * k;
            let topk_bytes = entries * std::mem::size_of::<u32>();
            let tok2row_bytes = entries * std::mem::size_of::<i32>();

            unsafe {
                if topk_bytes > 0 {
                    std::ptr::write_bytes(
                        topk_ids_buf.contents() as *mut u8,
                        0xFF,
                        topk_bytes,
                    );
                }
                if tok2row_bytes > 0 {
                    std::ptr::write_bytes(
                        tok2row_buf.contents() as *mut u8,
                        0xFF,
                        tok2row_bytes,
                    );
                }
            }
        }

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let root = &*mtl_command_buffer;

        match &self.router {
            RouterBlock::Quantized(block) => {
                block.encode(state, command_buffer, parameters)
            },
            RouterBlock::Metal {
                pipeline,
                weights_buf,
                biases_buf,
                mixture_size,
                model_dim,
            } => {
                let compute_encoder = root.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(pipeline);
                compute_encoder.set_buffer(0, Some(&main_buf), 0);
                compute_encoder.set_buffer(1, Some(weights_buf), 0);
                compute_encoder.set_buffer(2, Some(biases_buf), 0);
                compute_encoder.set_buffer(3, Some(&router_logits_buf), 0);

                let t_u = suffix_length as u32;
                let dm_u = *model_dim as u32;
                let e_u = *mixture_size as u32;
                compute_encoder.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &t_u as *const u32 as *const _,
                );
                compute_encoder.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &dm_u as *const u32 as *const _,
                );
                compute_encoder.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &e_u as *const u32 as *const _,
                );

                // Optimized: 8 simdgroups per TG (256 threads) with TG input caching
                let num_simdgroups: u64 = 8;
                let tg_x = ((*mixture_size as u64) + num_simdgroups - 1)
                    / num_simdgroups;
                compute_encoder.dispatch_thread_groups(
                    metal::MTLSize::new(tg_x, suffix_length as u64, 1),
                    metal::MTLSize::new(32 * num_simdgroups, 1, 1),
                );
                compute_encoder.end_encoding();
            },
        }

        self.topk_kernel
            .encode(
                root,
                self.data_type,
                MoeTopKArguments {
                    logits_buffer: &router_logits_buf,
                    topk_ids_buffer: &topk_ids_buf,
                    topk_probs_buffer: &topk_probs_buf,
                    t: suffix_length,
                    e,
                    k,
                    renorm: true,
                },
            )
            .expect("MoE TopK failed");

        self.counts_kernel
            .encode(
                root,
                MoeBucketCountsArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    counts_buffer: &counts_buf,
                    partials_buffer: &partials_buf,
                    t: suffix_length,
                    e,
                    k,
                },
            )
            .expect("MoE counts failed");

        self.offsets_kernel
            .encode(
                root,
                MoeOffsetsScanArguments {
                    counts_buffer: &counts_buf,
                    offsets_buffer: &offsets_buf,
                    sumk_buffer: &sumk_buf,
                    e,
                },
            )
            .expect("MoE offsets scan failed");

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

        self.experts_kernel
            .encode(
                root,
                MoeExpertsArguments {
                    x_perm_buffer: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    w13_all: &self.shared_weights.w13_buf,
                    w2_all: &self.shared_weights.w2_buf,
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
                    gate_clip_min: self.moe_config.expert_config.gate_clipping
                        [0]
                    .unwrap_or(f32::NEG_INFINITY),
                    gate_clip_max: self.moe_config.expert_config.gate_clipping
                        [1]
                    .unwrap_or(f32::INFINITY),
                    up_clip_min: self.moe_config.expert_config.up_clipping[0],
                    up_clip_max: self.moe_config.expert_config.up_clipping[1],
                    silu_alpha: self
                        .moe_config
                        .expert_config
                        .activation
                        .alpha(),
                    data_type: self.data_type,
                },
            )
            .expect("MoE experts failed");

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

        // NOTE: When inspecting kernels, we temporarily comment out the buffer cleanup.

        let wait_for_gpu = parameters.wait_until_completed || debug_moe;
        if wait_for_gpu {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }

        if let Some(traces_rc) = state.traces.as_ref() {
            let layer_trace_rc = {
                let traces_borrow = traces_rc.borrow();
                traces_borrow.layer_results[self.layer_index].clone()
            };

            let mut layer_trace = layer_trace_rc.borrow_mut();
            let moe_trace = layer_trace.ensure_moe_trace(
                suffix_length,
                e,
                k,
                self.model_dim,
            );

            let routed_tokens = suffix_length * k;

            unsafe {
                let src_topk_ids = std::slice::from_raw_parts(
                    topk_ids_buf.contents() as *const i32,
                    routed_tokens,
                );
                moe_trace.topk_ids.copy_from_slice(src_topk_ids);

                let src_counts = std::slice::from_raw_parts(
                    counts_buf.contents() as *const u32,
                    e,
                );
                moe_trace.counts.copy_from_slice(src_counts);

                let src_offsets = std::slice::from_raw_parts(
                    offsets_buf.contents() as *const u32,
                    e + 1,
                );
                moe_trace.offsets.copy_from_slice(src_offsets);

                let src_sumk = std::slice::from_raw_parts(
                    sumk_buf.contents() as *const u32,
                    1,
                );
                moe_trace.sumk.copy_from_slice(src_sumk);

                let src_bucketed_ids = std::slice::from_raw_parts(
                    bucketed_ids_buf.contents() as *const u32,
                    routed_tokens,
                );
                moe_trace.bucketed_ids.copy_from_slice(src_bucketed_ids);

                let src_tok2row = std::slice::from_raw_parts(
                    tok2row_buf.contents() as *const i32,
                    routed_tokens,
                );
                moe_trace.tok2row.copy_from_slice(src_tok2row);
            }

            Self::copy_probs_buffer(
                self.data_type,
                &topk_probs_buf,
                &mut moe_trace.topk_probs,
                routed_tokens,
            );
            Self::copy_probs_buffer(
                self.data_type,
                &bucketed_probs_buf,
                &mut moe_trace.bucketed_probs,
                routed_tokens,
            );
            Self::copy_probs_buffer(
                self.data_type,
                &y_partial_buf,
                &mut moe_trace.y_partial,
                routed_tokens * self.model_dim,
            );
        }
    }
}

impl MoeBlockEncodable {
    fn copy_probs_buffer(
        data_type: KernelDataType,
        buffer: &metal::Buffer,
        destination: &mut [f32],
        len: usize,
    ) {
        if len == 0 {
            return;
        }

        unsafe {
            match data_type {
                KernelDataType::BFloat16 => {
                    let src = std::slice::from_raw_parts(
                        buffer.contents() as *const bf16,
                        len,
                    );
                    destination
                        .iter_mut()
                        .zip(src.iter())
                        .for_each(|(dst, value)| *dst = value.to_f32());
                },
                KernelDataType::Float16 => {
                    let src = std::slice::from_raw_parts(
                        buffer.contents() as *const f16,
                        len,
                    );
                    destination
                        .iter_mut()
                        .zip(src.iter())
                        .for_each(|(dst, value)| *dst = value.to_f32());
                },
                KernelDataType::Float32 => {
                    let src = std::slice::from_raw_parts(
                        buffer.contents() as *const f32,
                        len,
                    );
                    destination.copy_from_slice(src);
                },
            }
        }
    }
}
