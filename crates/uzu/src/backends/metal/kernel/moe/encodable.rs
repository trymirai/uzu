use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    MoeBucketCountsArguments, MoeBucketCountsKernel, MoeExpertsArguments,
    MoeExpertsKernel, MoeFinalizeArguments, MoeFinalizeKernel,
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

pub struct MoeBlockEncodable {
    router: RouterBlock,
    topk_kernel: MoeTopKKernel,
    counts_kernel: MoeBucketCountsKernel,
    offsets_kernel: MoeOffsetsScanKernel,
    scatter_kernels: MoeScatterKernels,
    experts_kernel: MoeExpertsKernel,
    finalize_kernel: MoeFinalizeKernel,
    moe_config: MixtureOfExpertsConfig,
    model_dim: usize,
    hidden_dim: usize,
    data_type: KernelDataType,
    fused_up_weights: bool,
    w1_buf: metal::Buffer,
    w2_buf: metal::Buffer,
    w3_buf: metal::Buffer,
    up_biases_buf: metal::Buffer,
    down_biases_buf: metal::Buffer,
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
                eprintln!(
                    "[MoE Init] Requesting router kernel: {}",
                    kernel_name
                );
                let pipeline = context
                    .compute_pipeline_state(kernel_name, None)
                    .map_err(|e| {
                        eprintln!(
                            "[MoE Init] FAILED to load '{}': {:?}",
                            kernel_name, e
                        );
                        e
                    })?;
                eprintln!("[MoE Init] Router kernel loaded OK");

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

        eprintln!("[MoE Init] Loading TopK kernel...");
        let topk_kernel = MoeTopKKernel::new(context).map_err(|e| {
            eprintln!("[MoE Init] FAILED TopK: {:?}", e);
            crate::backends::metal::MTLError::Generic(format!(
                "TopK kernel error: {:?}",
                e
            ))
        })?;
        eprintln!("[MoE Init] TopK OK");
        eprintln!("[MoE Init] Loading Counts...");
        let counts_kernel =
            MoeBucketCountsKernel::new(context).map_err(|e| {
                eprintln!("[MoE Init] FAILED Counts: {:?}", e);
                crate::backends::metal::MTLError::Generic(format!(
                    "Counts kernel error: {:?}",
                    e
                ))
            })?;
        eprintln!("[MoE Init] Counts OK");
        eprintln!("[MoE Init] Loading Offsets...");
        let offsets_kernel =
            MoeOffsetsScanKernel::new(context).map_err(|e| {
                eprintln!("[MoE Init] FAILED Offsets: {:?}", e);
                crate::backends::metal::MTLError::Generic(format!(
                    "Offsets kernel error: {:?}",
                    e
                ))
            })?;
        eprintln!("[MoE Init] Offsets OK");
        eprintln!("[MoE Init] Loading Scatter...");
        let scatter_kernels = MoeScatterKernels::new(context).map_err(|e| {
            eprintln!("[MoE Init] FAILED Scatter: {:?}", e);
            crate::backends::metal::MTLError::Generic(format!(
                "Scatter kernels error: {:?}",
                e
            ))
        })?;
        eprintln!("[MoE Init] Scatter OK");
        eprintln!("[MoE Init] Loading Experts...");
        let experts_kernel = MoeExpertsKernel::new(context).map_err(|e| {
            eprintln!("[MoE Init] FAILED Experts: {:?}", e);
            crate::backends::metal::MTLError::Generic(format!(
                "Experts kernel error: {:?}",
                e
            ))
        })?;
        eprintln!("[MoE Init] Experts OK");
        eprintln!("[MoE Init] Loading Finalize...");
        let finalize_kernel = MoeFinalizeKernel::new(context).map_err(|e| {
            eprintln!("[MoE Init] FAILED Finalize: {:?}", e);
            crate::backends::metal::MTLError::Generic(format!(
                "Finalize kernel error: {:?}",
                e
            ))
        })?;
        eprintln!("[MoE Init] Finalize OK");

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

        let up_shape = up_arr.shape();
        let fused_up_weights = up_shape[2] == hidden_dim * 2;

        eprintln!(
            "[MoE Init] Up weights shape: {:?}, fused={}",
            up_shape, fused_up_weights
        );
        let up_buf = unsafe { up_arr.mtl_buffer().clone() };
        let (w1_buf, w3_buf) = if fused_up_weights {
            eprintln!("[MoE Init] Starting weight transformation...");
            let elem_size: u64 = match activation_data_type {
                DataType::F16 | DataType::BF16 => 2,
                DataType::F32 => 4,
                _ => 2,
            };

            let single_expert_bytes =
                (model_dim * hidden_dim) as u64 * elem_size;
            let total_bytes =
                single_expert_bytes * moe_config.mixture_size as u64;

            let w1_buf = context.device.new_buffer(
                total_bytes,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let w3_buf = context.device.new_buffer(
                total_bytes,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let kernel_name = match activation_data_type {
                DataType::F16 => "transpose_split_fused_expert_weights_f16",
                DataType::BF16 => "transpose_split_fused_expert_weights_bf16",
                _ => "transpose_split_fused_expert_weights_f16",
            };
            eprintln!(
                "[MoE Init] Loading weight transform kernel: {}",
                kernel_name
            );
            let pipeline = context
                .compute_pipeline_state(kernel_name, None)
                .map_err(|e| {
                    eprintln!("[MoE Init] FAILED weight transform: {:?}", e);
                    e
                })?;
            eprintln!("[MoE Init] Weight transform kernel loaded");

            let command_buffer = context.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&up_buf), 0);
            encoder.set_buffer(1, Some(&w1_buf), 0);
            encoder.set_buffer(2, Some(&w3_buf), 0);

            let e_u = moe_config.mixture_size as u32;
            let dm_u = model_dim as u32;
            let dff_u = hidden_dim as u32;
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &e_u as *const u32 as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &dm_u as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &dff_u as *const u32 as *const _,
            );

            let threads_per_tg = metal::MTLSize::new(16, 16, 1);
            let num_tg_x = ((model_dim + 15) / 16) as u64;
            let num_tg_y = ((hidden_dim + 15) / 16) as u64;
            let num_tg_z = moe_config.mixture_size as u64;
            let threadgroups =
                metal::MTLSize::new(num_tg_x, num_tg_y, num_tg_z);
            encoder.dispatch_thread_groups(threadgroups, threads_per_tg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            eprintln!(
                "[MoE Init] Weight transformation complete, buffers ready"
            );

            (w1_buf, w3_buf)
        } else {
            eprintln!(
                "[MoE Init] Weights not fused, using source buffers directly"
            );
            let w3_buf = context
                .device
                .new_buffer(0, metal::MTLResourceOptions::StorageModeShared);
            (up_buf, w3_buf)
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

        Ok(Self {
            router,
            topk_kernel,
            counts_kernel,
            offsets_kernel,
            scatter_kernels,
            experts_kernel,
            finalize_kernel,
            moe_config: moe_config.clone(),
            model_dim,
            hidden_dim,
            data_type,
            fused_up_weights,
            w1_buf,
            w2_buf,
            w3_buf,
            up_biases_buf,
            down_biases_buf,
        })
    }

    fn gating_code_from_activation(
        activation: &Activation,
        fused: bool,
    ) -> u32 {
        if fused {
            match activation {
                Activation::GELU => 3,
                Activation::SILU { .. } => 2,
            }
        } else {
            match activation {
                Activation::GELU => 0,
                Activation::SILU { .. } => 1,
            }
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
            ArrayId::MoeTok2Row,
            ArrayId::MoeYPartial,
            ArrayId::MoeScatterPartials,
            ArrayId::MoeScatterBlockBases,
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

        let main_buf = unsafe { borrow0.mtl_buffer().clone() };
        let router_logits_buf = unsafe { borrow1.mtl_buffer().clone() };
        let topk_ids_buf = unsafe { borrow2.mtl_buffer().clone() };
        let topk_probs_buf = unsafe { borrow3.mtl_buffer().clone() };
        let counts_buf = unsafe { borrow4.mtl_buffer().clone() };
        let offsets_buf = unsafe { borrow5.mtl_buffer().clone() };
        let sumk_buf = unsafe { borrow6.mtl_buffer().clone() };
        let bucketed_ids_buf = unsafe { borrow7.mtl_buffer().clone() };
        let bucketed_probs_buf = unsafe { borrow8.mtl_buffer().clone() };
        let tok2row_buf = unsafe { borrow9.mtl_buffer().clone() };
        let y_partial_buf = unsafe { borrow10.mtl_buffer().clone() };
        let partials_buf = unsafe { borrow11.mtl_buffer().clone() };
        let block_bases_buf = unsafe { borrow12.mtl_buffer().clone() };

        let e = self.moe_config.mixture_size;
        let k = self.moe_config.num_experts_per_token;

        let root = command_buffer.root_command_buffer();
        let compute_encoder = root.new_compute_command_encoder();

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
                let threadgroups_x = (*mixture_size as u64 + num_simdgroups - 1) / num_simdgroups;
                let threadgroups_y = suffix_length as u64;
                compute_encoder.dispatch_thread_groups(
                    metal::MTLSize::new(threadgroups_x, threadgroups_y, 1),
                    metal::MTLSize::new(32 * num_simdgroups, 1, 1),
                );
            },
        }

        self.topk_kernel
            .encode(
                &compute_encoder,
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
                &compute_encoder,
                MoeBucketCountsArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    counts_buffer: &counts_buf,
                    t: suffix_length,
                    e,
                    k,
                },
            )
            .expect("MoE counts failed");

        self.offsets_kernel
            .encode(
                &compute_encoder,
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
                &compute_encoder,
                super::MoeBlockBasesArguments {
                    partials_buffer: &partials_buf,
                    block_bases_buffer: &block_bases_buf,
                    e,
                    num_blocks,
                    num_tiles,
                },
            )
            .expect("MoE block bases failed");

        self.scatter_kernels
            .encode_scatter_with_map(
                &compute_encoder,
                MoeScatterWithMapArguments {
                    base: super::MoeScatterArguments {
                        topk_ids_buffer: &topk_ids_buf,
                        topk_probs_buffer: &topk_probs_buf,
                        offsets_buffer: &offsets_buf,
                        block_bases_buffer: &block_bases_buf,
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

        let gating_code = Self::gating_code_from_activation(
            &self.moe_config.expert_config.activation,
            self.fused_up_weights,
        );

        self.experts_kernel
            .encode(
                &compute_encoder,
                MoeExpertsArguments {
                    x_buffer: &main_buf,
                    bucketed_token_ids: &bucketed_ids_buf,
                    expert_offsets: &offsets_buf,
                    w1_all: &self.w1_buf,
                    w3_all: &self.w3_buf,
                    w2_all: &self.w2_buf,
                    y_partial: &y_partial_buf,
                    up_biases: &self.up_biases_buf,
                    down_biases: &self.down_biases_buf,
                    t: suffix_length,
                    d_model: self.model_dim,
                    d_ff: self.hidden_dim,
                    e,
                    gating_code,
                    gate_clip_min: self.moe_config.expert_config.gate_clipping
                        [0]
                    .unwrap_or(f32::NEG_INFINITY),
                    gate_clip_max: self.moe_config.expert_config.gate_clipping
                        [1]
                    .unwrap_or(f32::INFINITY),
                    up_clip_min: self.moe_config.expert_config.up_clipping[0],
                    up_clip_max: self.moe_config.expert_config.up_clipping[1],
                    silu_alpha: self.moe_config.expert_config.activation.alpha(),
                },
            )
            .expect("MoE experts failed");

        self.finalize_kernel
            .encode(
                &compute_encoder,
                MoeFinalizeArguments {
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

        compute_encoder.end_encoding();
    }
}
