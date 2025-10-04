use std::rc::Rc;

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
        KernelDataType, MTLContext,
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

        if std::env::var_os("UZU_DEBUG_MOE_WEIGHTS").is_some() {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_abs = 0f32;
            eprintln!(
                "[DebugMoeWeights] w13 dtype={:?} shape={:?}",
                up_arr.data_type(),
                up_shape
            );
            match up_arr.data_type() {
                DataType::BF16 => {
                    if let Ok(slice) = up_arr.as_slice::<half::bf16>() {
                        if let Some((idx, _)) = slice
                            .iter()
                            .enumerate()
                            .find(|(_, v)| f32::from(**v).is_nan())
                        {
                            eprintln!(
                                "[DebugMoeWeights] w13 first NaN at idx {}",
                                idx
                            );
                            eprintln!(
                                "[DebugMoeWeights] sample {:?}",
                                &slice[idx.saturating_sub(4)
                                    ..(idx + 1).min(slice.len())]
                            );
                        }
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F16 => {
                    if let Ok(slice) = up_arr.as_slice::<half::f16>() {
                        if let Some((idx, _)) = slice
                            .iter()
                            .enumerate()
                            .find(|(_, v)| f32::from(**v).is_nan())
                        {
                            eprintln!(
                                "[DebugMoeWeights] w13 first NaN at idx {}",
                                idx
                            );
                            eprintln!(
                                "[DebugMoeWeights] sample {:?}",
                                &slice[idx.saturating_sub(4)
                                    ..(idx + 1).min(slice.len())]
                            );
                        }
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                _ => {},
            }
            eprintln!(
                "[DebugMoeWeights] w13 stats min={:?} max={:?} max_abs={:?}",
                if min_val.is_finite() {
                    Some(min_val)
                } else {
                    None
                },
                if max_val.is_finite() {
                    Some(max_val)
                } else {
                    None
                },
                max_abs
            );
        }

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

        if std::env::var_os("UZU_DEBUG_MOE_WEIGHTS").is_some()
            || std::env::var_os("UZU_DEBUG_MOE").is_some()
        {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_abs = 0f32;
            eprintln!(
                "[DebugMoeWeights] w2 dtype={:?} shape={:?}",
                w2_arr.data_type(),
                w2_arr.shape()
            );
            match w2_arr.data_type() {
                DataType::BF16 => {
                    if let Ok(slice) = w2_arr.as_slice::<half::bf16>() {
                        if let Some((idx, _)) = slice
                            .iter()
                            .enumerate()
                            .find(|(_, v)| f32::from(**v).is_nan())
                        {
                            eprintln!(
                                "[DebugMoeWeights] w2 first NaN at idx {}",
                                idx
                            );
                        }
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F16 => {
                    if let Ok(slice) = w2_arr.as_slice::<half::f16>() {
                        if let Some((idx, _)) = slice
                            .iter()
                            .enumerate()
                            .find(|(_, v)| f32::from(**v).is_nan())
                        {
                            eprintln!(
                                "[DebugMoeWeights] w2 first NaN at idx {}",
                                idx
                            );
                        }
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                _ => {},
            }
            eprintln!(
                "[DebugMoeWeights] w2 stats min={:?} max={:?} max_abs={:?}",
                if min_val.is_finite() {
                    Some(min_val)
                } else {
                    None
                },
                if max_val.is_finite() {
                    Some(max_val)
                } else {
                    None
                },
                max_abs
            );
        }

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

        if std::env::var_os("UZU_DEBUG_MOE_WEIGHTS").is_some()
            || std::env::var_os("UZU_DEBUG_MOE").is_some()
        {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_abs = 0f32;
            match up_biases_arr.data_type() {
                DataType::BF16 => {
                    if let Ok(slice) = up_biases_arr.as_slice::<half::bf16>() {
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F16 => {
                    if let Ok(slice) = up_biases_arr.as_slice::<half::f16>() {
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F32 => {
                    if let Ok(slice) = up_biases_arr.as_slice::<f32>() {
                        for &val in slice {
                            if val.is_finite() {
                                min_val = min_val.min(val);
                                max_val = max_val.max(val);
                                max_abs = max_abs.max(val.abs());
                            }
                        }
                    }
                },
                _ => {},
            }
            eprintln!(
                "[DebugMoeWeights] up_biases stats min={:?} max={:?} max_abs={:?}",
                if min_val.is_finite() {
                    Some(min_val)
                } else {
                    None
                },
                if max_val.is_finite() {
                    Some(max_val)
                } else {
                    None
                },
                max_abs
            );
        }

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

        if std::env::var_os("UZU_DEBUG_MOE_WEIGHTS").is_some()
            || std::env::var_os("UZU_DEBUG_MOE").is_some()
        {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_abs = 0f32;
            match down_biases_arr.data_type() {
                DataType::BF16 => {
                    if let Ok(slice) = down_biases_arr.as_slice::<half::bf16>()
                    {
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F16 => {
                    if let Ok(slice) = down_biases_arr.as_slice::<half::f16>() {
                        for &val in slice {
                            let f = f32::from(val);
                            if f.is_finite() {
                                min_val = min_val.min(f);
                                max_val = max_val.max(f);
                                max_abs = max_abs.max(f.abs());
                            }
                        }
                    }
                },
                DataType::F32 => {
                    if let Ok(slice) = down_biases_arr.as_slice::<f32>() {
                        for &val in slice {
                            if val.is_finite() {
                                min_val = min_val.min(val);
                                max_val = max_val.max(val);
                                max_abs = max_abs.max(val.abs());
                            }
                        }
                    }
                },
                _ => {},
            }
            eprintln!(
                "[DebugMoeWeights] down_biases stats min={:?} max={:?} max_abs={:?}",
                if min_val.is_finite() {
                    Some(min_val)
                } else {
                    None
                },
                if max_val.is_finite() {
                    Some(max_val)
                } else {
                    None
                },
                max_abs
            );
        }

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

        let mut borrow0 = arrays[0].borrow_mut();
        let mut borrow1 = arrays[1].borrow_mut();
        let mut borrow2 = arrays[2].borrow_mut();
        let mut borrow3 = arrays[3].borrow_mut();
        let mut borrow4 = arrays[4].borrow_mut();
        let mut borrow5 = arrays[5].borrow_mut();
        let mut borrow6 = arrays[6].borrow_mut();
        let mut borrow7 = arrays[7].borrow_mut();
        let mut borrow8 = arrays[8].borrow_mut();
        let mut borrow9 = arrays[9].borrow_mut();
        let mut borrow10 = arrays[10].borrow_mut();
        let mut borrow11 = arrays[11].borrow_mut();
        let mut borrow12 = arrays[12].borrow_mut();
        let mut borrow13 = arrays[13].borrow_mut();
        let mut borrow14 = arrays[14].borrow_mut();
        let mut borrow15 = arrays[15].borrow_mut();
        let mut borrow16 = arrays[16].borrow_mut();
        let mut borrow17 = arrays[17].borrow_mut();
        let mut borrow18 = arrays[18].borrow_mut();
        let mut borrow19 = arrays[19].borrow_mut();

        let main_buf = unsafe { borrow0.mtl_buffer().clone() };
        let router_logits_buf = unsafe { borrow1.mtl_buffer().clone() };
        let topk_ids_buf = unsafe { borrow2.mtl_buffer().clone() };
        let topk_probs_buf = unsafe { borrow3.mtl_buffer().clone() };
        let counts_buf = unsafe { borrow4.mtl_buffer().clone() };
        let offsets_buf = unsafe { borrow5.mtl_buffer().clone() };
        let sumk_buf = unsafe { borrow6.mtl_buffer().clone() };
        let bucketed_ids_buf = unsafe { borrow7.mtl_buffer().clone() };
        let bucketed_probs_buf = unsafe { borrow8.mtl_buffer().clone() };
        let x_perm_buf = unsafe { borrow9.mtl_buffer().clone() };
        let tok2row_buf = unsafe { borrow10.mtl_buffer().clone() };
        let y_partial_buf = unsafe { borrow11.mtl_buffer().clone() };
        let tile_counts_buf = unsafe { borrow12.mtl_buffer().clone() };
        let tile_offsets_buf = unsafe { borrow13.mtl_buffer().clone() };
        let tile_map_buf = unsafe { borrow14.mtl_buffer().clone() };
        let total_tiles_buf = unsafe { borrow15.mtl_buffer().clone() };
        let dispatch_args_buf = unsafe { borrow16.mtl_buffer().clone() };
        let partials_buf = unsafe { borrow17.mtl_buffer().clone() };
        let block_bases_buf = unsafe { borrow18.mtl_buffer().clone() };
        let block_alloc_buf = unsafe { borrow19.mtl_buffer().clone() };

        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        let root = command_buffer.root_command_buffer();

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

                let num_simdgroups = 4u64;
                let threadgroups_x = (*mixture_size as u64 + num_simdgroups
                    - 1)
                    / num_simdgroups;
                let threadgroups_y = suffix_length as u64;
                compute_encoder.dispatch_thread_groups(
                    metal::MTLSize::new(threadgroups_x, threadgroups_y, 1),
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
    }
}
