use std::{cell::RefCell, mem::size_of, rc::Rc};

use half::{bf16, f16};
use metal::MTLResourceOptions;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    MoeBucketCountsArguments, MoeBucketCountsKernel, MoeExpertsArguments,
    MoeExpertsKernel, MoeExpertsTwoPassArguments, MoeFinalizeKernel,
    MoeGatherArguments, MoeGatherKernel, MoeOffsetsScanArguments,
    MoeOffsetsScanKernel, MoeScatterKernels, MoeScatterWithMapArguments,
    MoeTopKArguments, MoeTopKKernel,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassState, MOE_TWO_PASS_K_TILE,
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
    router_data_type: KernelDataType,
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
    decode_two_pass: bool,
}

impl MoeBlockEncodable {
    pub fn load_shared_expert_weights(
        context: &MTLContext,
        _moe_config: &MixtureOfExpertsConfig,
        model_dim: usize,
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
                "MoE experts require fused gate+up weights (input shape [E, d_model, 2*d_ff])"
                    .to_string(),
            ));
        }

        // Transpose W13 from source layout [E, d_model, 2*d_ff] to GPU layout [E, 2*d_ff, d_model]
        let e = up_shape[0];
        let d_model_size = up_shape[1];
        let two_d_ff = up_shape[2];

        let transpose_err =
            |err: crate::device::array::ArrayConversionError| {
                crate::backends::metal::MTLError::Generic(format!(
                    "W13 transpose failed: {}",
                    err
                ))
            };

        let total_size = e * d_model_size * two_d_ff;
        let w13_transposed = match up_arr.data_type() {
            DataType::BF16 => {
                let src = up_arr.as_slice::<bf16>().map_err(transpose_err)?;
                let mut dst = vec![bf16::from_f32(0.0); total_size];

                for expert in 0..e {
                    let expert_offset = expert * d_model_size * two_d_ff;
                    for dm in 0..d_model_size {
                        for ff in 0..two_d_ff {
                            // src: [E, d_model, 2*d_ff] -> index: expert_offset + dm * two_d_ff + ff
                            // dst: [E, 2*d_ff, d_model] -> index: expert_offset + ff * d_model + dm
                            dst[expert_offset + ff * d_model_size + dm] =
                                src[expert_offset + dm * two_d_ff + ff];
                        }
                    }
                }

                context.device.new_buffer_with_data(
                    dst.as_ptr() as *const _,
                    (total_size * size_of::<bf16>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            },
            DataType::F16 => {
                let src = up_arr.as_slice::<f16>().map_err(transpose_err)?;
                let mut dst = vec![f16::from_f32(0.0); total_size];

                for expert in 0..e {
                    let expert_offset = expert * d_model_size * two_d_ff;
                    for dm in 0..d_model_size {
                        for ff in 0..two_d_ff {
                            dst[expert_offset + ff * d_model_size + dm] =
                                src[expert_offset + dm * two_d_ff + ff];
                        }
                    }
                }

                context.device.new_buffer_with_data(
                    dst.as_ptr() as *const _,
                    (total_size * size_of::<f16>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            },
            DataType::F32 => {
                let src = up_arr.as_slice::<f32>().map_err(transpose_err)?;
                let mut dst = vec![0.0f32; total_size];

                for expert in 0..e {
                    let expert_offset = expert * d_model_size * two_d_ff;
                    for dm in 0..d_model_size {
                        for ff in 0..two_d_ff {
                            dst[expert_offset + ff * d_model_size + dm] =
                                src[expert_offset + dm * two_d_ff + ff];
                        }
                    }
                }

                context.device.new_buffer_with_data(
                    dst.as_ptr() as *const _,
                    (total_size * size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            },
            other => {
                return Err(crate::backends::metal::MTLError::Generic(
                    format!("Unsupported W13 weight dtype {:?}", other),
                ));
            },
        };

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

        // Use W2 in original layout [E, d_ff, d_model] - no transpose needed
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

        let w13_buf = w13_transposed;
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

        let router_data_type: DataType =
            moe_config.router_config.activation_precision().into();
        let router_kernel_data_type: KernelDataType = router_data_type.into();

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
                let kernel_name = match router_kernel_data_type {
                    KernelDataType::BFloat16 => "moe_router_bf16",
                    KernelDataType::Float16 => "moe_router_f16",
                    KernelDataType::Float32 => "moe_router_f32",
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

        let decode_two_pass = std::env::var_os("UZU_DEBUG_MOE_TWO_PASS")
            .map(|v| v != "0")
            .unwrap_or(false);

        Ok(Self {
            router,
            router_data_type: router_kernel_data_type,
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
            decode_two_pass,
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
        if std::env::var_os("UZU_DEBUG_ROUTER").is_some() {
            let dtype = array_iter.clone().next().unwrap().borrow().data_type();
            eprintln!(
                "[RouterDebug] layer={} router logits dtype={:?}",
                self.layer_index, dtype
            );
        }
        let router_logits_buf = clone_buffer(array_iter.next().unwrap());
        let topk_ids_buf = clone_buffer(array_iter.next().unwrap());
        let topk_probs_buf = clone_buffer(array_iter.next().unwrap());
        let counts_buf = clone_buffer(array_iter.next().unwrap());
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
                self.router_data_type,
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
            if self.decode_two_pass {
                let total_rows = suffix_length * k;
                let num_tiles_k = ((self.hidden_dim + MOE_TWO_PASS_K_TILE - 1)
                    / MOE_TWO_PASS_K_TILE)
                    as u32;

                self.experts_kernel
                    .encode_two_pass_decode(
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
                self.experts_kernel
                    .encode_gemv_decode(root, experts_args)
                    .expect("MoE experts decode failed");
            }
        } else {
            self.experts_kernel
                .encode(root, experts_args)
                .expect("MoE experts failed");
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
