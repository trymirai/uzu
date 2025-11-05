#![allow(dead_code)]
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use super::{
    super::{MTLContext, MetalArray},
    buffers::ForwardPassBuffers,
    kv_cache::KVCache,
    model_shape::ModelShape,
};
use crate::{
    DataType, DeviceContext,
    backends::metal::forward_pass::traces::DecoderActivationTrace,
    config::{DecoderConfig, EmbeddingConfig, MLPConfig},
    device::array::Array,
    parameters::ParameterTree,
    session::parameter::SamplingMethod,
};

type ArrayCell = RefCell<MetalArray>;

pub enum EmbeddingsBuffers {
    Tied {
        /// [vocab_size, model_dim]
        weights: ArrayCell,
    },
    Untied {
        /// [vocab_size, model_dim]
        input_weights: ArrayCell,
        /// [vocab_size, model_dim]
        output_weights: ArrayCell,
    },
    QuantizedTied {
        /// [vocab_size, model_dim]
        weights: ArrayCell,
        /// [vocab_size]
        scales: ArrayCell,
    },
}

impl EmbeddingsBuffers {
    pub fn new(
        context: &MTLContext,
        embeddings_config: &EmbeddingConfig,
        model_shape: &ModelShape,
    ) -> Self {
        unsafe {
            match embeddings_config {
                EmbeddingConfig::Tied {
                    common: _,
                    precision: _,
                } => Self::Tied {
                    weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                EmbeddingConfig::Untied {
                    common: _,
                    precision: _,
                } => Self::Untied {
                    input_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                    output_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_output_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                EmbeddingConfig::QuantizedTied {
                    embedding_quantization_mode,
                    ..
                } => Self::QuantizedTied {
                    weights: RefCell::new(context.array_uninitialized(
                        &model_shape.quantized_embeddings_weights_shape(),
                        (*embedding_quantization_mode).into(),
                    )),
                    scales: RefCell::new(context.array_uninitialized(
                        &model_shape.quantized_embeddings_scales_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
            }
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) {
        let embeddings_tree = parameter_tree.subtree("embedding").unwrap();
        match self {
            EmbeddingsBuffers::Tied {
                weights,
            } => {
                let embeddings_view = embeddings_tree.leaf("weights").unwrap();
                weights.borrow_mut().copy_from_array(&embeddings_view);
            },
            EmbeddingsBuffers::Untied {
                input_weights,
                output_weights,
            } => {
                let mapping = vec![
                    ("input_weights", input_weights),
                    ("output_weights", output_weights),
                ];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from_array(&view);
                }
            },
            EmbeddingsBuffers::QuantizedTied {
                weights,
                scales,
            } => {
                let mapping = vec![("weights", weights), ("scales", scales)];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from_array(&view);
                }
            },
        }
    }
}

pub struct RopeBuffers {
    /// [rope_max_sequence_length, head_dim]
    pub cosines: ArrayCell,
    /// [rope_max_sequence_length, head_dim]
    pub sines: ArrayCell,
}

impl RopeBuffers {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
    ) -> Self {
        unsafe {
            let rotated_queries_shape = model_shape.rotated_queries_shape(1);
            let head_dim = rotated_queries_shape[2];
            let rope_max_sequence_length = model_shape.context_length();

            Self {
                cosines: RefCell::new(context.array_uninitialized(
                    &[rope_max_sequence_length, head_dim],
                    model_shape.activation_data_type(),
                )),
                sines: RefCell::new(context.array_uninitialized(
                    &[rope_max_sequence_length, head_dim],
                    model_shape.activation_data_type(),
                )),
            }
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        rope_name: String,
    ) {
        let rope_tree = parameter_tree.subtree(rope_name.as_str()).unwrap();

        let cosines_view = rope_tree.leaf("cosines").unwrap();
        self.cosines.borrow_mut().copy_from_array(&cosines_view);

        let sines_view = rope_tree.leaf("sines").unwrap();
        self.sines.borrow_mut().copy_from_array(&sines_view);
    }
}

pub struct SharedBuffers {
    pub embeddings: EmbeddingsBuffers,
    pub global_rope: RopeBuffers,
    pub local_rope: Option<RopeBuffers>,
    pub moe_expert_weights: Option<Vec<MoeExpertWeights>>,
    pub attention_sinks: Option<Vec<ArrayCell>>,
}

pub struct MoeExpertWeights {
    pub w1: ArrayCell,
    pub w2: ArrayCell,
    pub w3: ArrayCell,
}

impl SharedBuffers {
    pub fn new(
        context: &MTLContext,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let embeddings = EmbeddingsBuffers::new(
            context,
            &decoder_config.embedding_config,
            model_shape,
        );

        let global_rope = RopeBuffers::new(context, model_shape);

        let local_rope = if decoder_config.local_rope_config.is_some() {
            Some(RopeBuffers::new(context, model_shape))
        } else {
            None
        };

        let moe_expert_weights = if matches!(
            decoder_config.layer_config.mlp_config,
            crate::config::MLPConfig::MixtureOfExperts(_)
        ) {
            Some(Vec::new())
        } else {
            None
        };

        let attention_sinks =
            if decoder_config.layer_config.attention_config.has_sinks {
                let num_heads = decoder_config.num_heads;
                Some(
                    (0..decoder_config.num_layers)
                        .map(|_| unsafe {
                            RefCell::new(context.array_uninitialized(
                                &[num_heads],
                                DataType::F32,
                            ))
                        })
                        .collect(),
                )
            } else {
                None
            };

        Self {
            embeddings,
            global_rope,
            local_rope,
            moe_expert_weights,
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) {
        self.embeddings.update_data(parameter_tree);
        self.global_rope
            .update_data(parameter_tree, String::from("global_rope"));
        if let Some(local_rope) = &mut self.local_rope {
            local_rope.update_data(parameter_tree, String::from("local_rope"));
        }

        if let Some(sinks_vec) = &mut self.attention_sinks {
            for (layer_idx, sink_cell) in sinks_vec.iter_mut().enumerate() {
                let layer_tree = parameter_tree
                    .subtree(&format!("layers.{}", layer_idx))
                    .unwrap();
                let attn_tree = layer_tree.subtree("attention").unwrap();
                let sinks_arr = attn_tree.leaf("sinks").unwrap();
                let mut dst = sink_cell.borrow_mut();
                let dst_slice = dst.as_slice_mut::<f32>().unwrap();

                match sinks_arr.data_type() {
                    DataType::F32 => {
                        let src = sinks_arr.as_slice::<f32>().unwrap();
                        dst_slice.copy_from_slice(src);
                    },
                    DataType::BF16 => {
                        let src = sinks_arr.as_slice::<bf16>().unwrap();
                        for (dst_val, src_val) in
                            dst_slice.iter_mut().zip(src.iter())
                        {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    DataType::F16 => {
                        let src = sinks_arr.as_slice::<f16>().unwrap();
                        for (dst_val, src_val) in
                            dst_slice.iter_mut().zip(src.iter())
                        {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    other => {
                        panic!(
                            "Unsupported attention sink data type: {:?}",
                            other
                        );
                    },
                }
            }
        }
    }
}

struct AuxBuffers {
    suffix_length: usize,
    /// [suffix_length, model_dim]
    main: ArrayCell,
    /// [suffix_length, model_dim]
    shortcut: ArrayCell,
    /// [suffix_length, (num_heads + 2 * num_groups) * head_dim]
    qkv: ArrayCell,
    /// [suffix_length, num_heads * head_dim]
    attention_output: ArrayCell,
    /// [suffix_length, 2 * hidden_dim]
    mlp_fused_up: ArrayCell,
    /// [suffix_length, hidden_dim]
    mlp_hidden: ArrayCell,
    /// [num_heads, max_suffix_length, head_dim]
    rotated_queries: ArrayCell,
    /// [num_groups, max_suffix_length, head_dim]
    rotated_keys: ArrayCell,
    /// [num_heads * suffix_length * total_blocks_count * head_dim]
    attention_partials: ArrayCell,
    /// [num_heads * suffix_length * total_blocks_count]
    attention_sums: ArrayCell,
    /// [num_heads * suffix_length * total_blocks_count]
    attention_maxs: ArrayCell,

    moe_topk_ids: Option<ArrayCell>,
    moe_topk_probs: Option<ArrayCell>,
    moe_offsets: Option<ArrayCell>,
    moe_sumk: Option<ArrayCell>,
    moe_bucketed_token_ids: Option<ArrayCell>,
    moe_bucketed_probs: Option<ArrayCell>,
    moe_x_perm: Option<ArrayCell>,
    moe_tok2row: Option<ArrayCell>,
    moe_y_partial: Option<ArrayCell>,
    moe_hidden: Option<ArrayCell>,
    moe_two_pass_row_expert_map: Option<ArrayCell>,
    moe_tile_counts: Option<ArrayCell>,
    moe_tile_offsets: Option<ArrayCell>,
    moe_tile_map: Option<ArrayCell>,
    moe_total_tiles: Option<ArrayCell>,
    moe_dispatch_args: Option<ArrayCell>,
    moe_scatter_partials: Option<ArrayCell>,
    moe_scatter_block_bases: Option<ArrayCell>,
    moe_block_alloc: Option<ArrayCell>,
}

impl AuxBuffers {
    fn new(
        scratch: &ForwardPassBuffers,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let act_dtype = model_shape.activation_data_type();
        unsafe {
            Self {
                suffix_length,
                main: RefCell::new(MetalArray::new(
                    scratch.main.clone(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                shortcut: RefCell::new(MetalArray::new(
                    scratch.shortcut.clone(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                qkv: RefCell::new(MetalArray::new(
                    scratch.qkv.clone(),
                    &model_shape.qkv_shape(suffix_length),
                    act_dtype,
                )),
                attention_output: RefCell::new(MetalArray::new(
                    scratch.attention_output.clone(),
                    &model_shape.attention_output_shape(suffix_length),
                    act_dtype,
                )),
                mlp_fused_up: RefCell::new(MetalArray::new(
                    scratch.mlp_fused_up.clone(),
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    act_dtype,
                )),
                mlp_hidden: RefCell::new(MetalArray::new(
                    scratch.mlp_hidden.clone(),
                    &model_shape.mlp_hidden_shape(suffix_length),
                    act_dtype,
                )),
                rotated_queries: RefCell::new(MetalArray::new(
                    scratch.rotated_queries.clone(),
                    &model_shape.rotated_queries_shape(suffix_length),
                    act_dtype,
                )),
                rotated_keys: RefCell::new(MetalArray::new(
                    scratch.rotated_keys.clone(),
                    &model_shape.rotated_keys_shape(suffix_length),
                    act_dtype,
                )),
                attention_partials: RefCell::new(MetalArray::new(
                    scratch.attention_partials.clone(),
                    &model_shape.attention_partials_shape(suffix_length),
                    act_dtype,
                )),
                attention_sums: RefCell::new(MetalArray::new(
                    scratch.attention_sums.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    act_dtype,
                )),
                attention_maxs: RefCell::new(MetalArray::new(
                    scratch.attention_maxs.clone(),
                    &model_shape.attention_maxs_shape(suffix_length),
                    act_dtype,
                )),

                moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_topk_ids.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_topk_ids_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_topk_probs: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_topk_probs.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_topk_probs_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                act_dtype,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_offsets: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_offsets.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape
                                    .moe_offsets_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_sumk: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_sumk.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_sumk_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_bucketed_token_ids: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_bucketed_token_ids.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape
                                    .moe_bucketed_token_ids_shape(max_routed),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_bucketed_probs: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_bucketed_probs.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape
                                    .moe_bucketed_probs_shape(max_routed),
                                act_dtype,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_x_perm: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_x_perm.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_x_perm_shape(max_routed),
                                DataType::F16,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tok2row: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tok2row.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_tok2row_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                DataType::I32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_y_partial: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_y_partial.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_y_partial_shape(max_routed),
                                DataType::F16,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_hidden: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_hidden.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_hidden_shape(max_routed),
                                act_dtype,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_two_pass_row_expert_map: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_two_pass_row_expert_map.as_ref().map(
                            |buf| {
                                RefCell::new(MetalArray::new(
                                    buf.clone(),
                                    &[max_routed],
                                    DataType::U32,
                                ))
                            },
                        )
                    },
                    _ => None,
                },
                moe_tile_counts: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tile_counts.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_counts_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tile_offsets: match &decoder_config.layer_config.mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tile_offsets.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape
                                    .moe_offsets_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tile_map: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_tile_map.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_tile_map_shape(max_routed),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_total_tiles: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_total_tiles.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_total_tiles_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_dispatch_args: match &decoder_config.layer_config.mlp_config
                {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_dispatch_args.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &model_shape.moe_dispatch_args_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_scatter_partials: scratch
                    .moe_scatter_partials
                    .as_ref()
                    .map(|buf| {
                        let num_blocks = ((suffix_length + 255) / 256).max(1);
                        match &decoder_config.layer_config.mlp_config {
                            MLPConfig::MixtureOfExperts(moe) => {
                                let num_tiles =
                                    ((moe.mixture_size + 512 - 1) / 512).max(1);
                                let entries = num_blocks * num_tiles * 512;
                                RefCell::new(MetalArray::new(
                                    buf.clone(),
                                    &[entries],
                                    DataType::U32,
                                ))
                            },
                            _ => unreachable!(),
                        }
                    }),
                moe_scatter_block_bases: scratch
                    .moe_scatter_block_bases
                    .as_ref()
                    .map(|buf| {
                        let num_blocks = ((suffix_length + 255) / 256).max(1);
                        match &decoder_config.layer_config.mlp_config {
                            MLPConfig::MixtureOfExperts(moe) => {
                                let num_tiles =
                                    ((moe.mixture_size + 512 - 1) / 512).max(1);
                                let entries = num_blocks * num_tiles * 512;
                                RefCell::new(MetalArray::new(
                                    buf.clone(),
                                    &[entries],
                                    DataType::U32,
                                ))
                            },
                            _ => unreachable!(),
                        }
                    }),
                moe_block_alloc: scratch.moe_block_alloc.as_ref().map(|buf| {
                    let num_blocks = ((suffix_length + 255) / 256).max(1);
                    match &decoder_config.layer_config.mlp_config {
                        MLPConfig::MixtureOfExperts(moe) => {
                            let num_tiles =
                                ((moe.mixture_size + 512 - 1) / 512).max(1);
                            let entries = num_blocks * num_tiles * 512;
                            RefCell::new(MetalArray::new(
                                buf.clone(),
                                &[entries],
                                DataType::U32,
                            ))
                        },
                        _ => unreachable!(),
                    }
                }),
            }
        }
    }

    fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}

pub struct ForwardPassState {
    context: Rc<MTLContext>,
    /// [suffix_length] - u64
    token_ids: ArrayCell,
    /// [suffix_length] - i32
    token_positions: ArrayCell,
    /// [suffix_length, suffix_length + prefix_length]
    attention_bias: HashMap<Option<usize>, ArrayCell>,
    /// [suffix_length, vocabulary_size]
    logits: ArrayCell,
    pub kv_cache: Rc<RefCell<KVCache>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    aux_buffers: AuxBuffers,
    /// [suffix_length] - u32 sampling output buffer
    pub sampling_output: Option<ArrayCell>,
    /// [suffix_length, vocabulary_size] - u8 bitmask (0 = disallowed, 1 = allowed)
    pub token_bitmask: Option<ArrayCell>,
    /// Current sampling configuration for this forward pass
    pub sampling_method: Option<SamplingMethod>,
    pub traces: Option<Rc<RefCell<DecoderActivationTrace>>>,
}

impl ForwardPassState {
    pub fn new(
        context: Rc<MTLContext>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        scratch: &ForwardPassBuffers,
        kv_cache: Rc<RefCell<KVCache>>,
        shared_buffers: Rc<RefCell<SharedBuffers>>,
        token_ids: &[u64],
        token_positions: &[usize],
        trace: bool,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(
            suffix_length,
            token_positions.len(),
            "Tokens and token positions must have the same length, got {} and {} respectively",
            suffix_length,
            token_positions.len()
        );
        assert!(
            suffix_length <= kv_cache.borrow().max_suffix_length(),
            "KV cache size can only accomodate a suffix of length up to {}, but tried to use a suffix of length {}",
            kv_cache.borrow().max_suffix_length(),
            suffix_length
        );
        let aux_buffers = AuxBuffers::new(
            scratch,
            decoder_config,
            model_shape,
            suffix_length,
        );

        // --------------------
        // Token IDs
        // --------------------
        let mut token_ids_array = unsafe {
            MetalArray::new(
                scratch.token_ids.clone(),
                &[suffix_length],
                DataType::U64,
            )
        };
        context.copy_from_view(&mut token_ids_array, token_ids.into());
        let token_ids_refcell = RefCell::new(token_ids_array);

        // --------------------
        // Token Positions (i32)
        // --------------------
        let mut token_positions_array = unsafe {
            MetalArray::new(
                scratch.token_positions.clone(),
                &[suffix_length],
                DataType::I32,
            )
        };
        let token_positions_i32: Box<[i32]> =
            token_positions.iter().map(|p| *p as i32).collect();
        context.copy_from_view(
            &mut token_positions_array,
            token_positions_i32.as_ref().into(),
        );
        let token_positions_refcell = RefCell::new(token_positions_array);

        // --------------------
        // Attention Bias
        // --------------------
        let prefix_length = kv_cache.borrow().max_prefix_length();
        let attention_bias_shape =
            [suffix_length, suffix_length + prefix_length];
        let act_dtype = model_shape.activation_data_type();

        let mut attention_bias_map: HashMap<Option<usize>, MetalArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
                    let array = unsafe {
                        MetalArray::new(
                            buffer.clone(),
                            &attention_bias_shape,
                            act_dtype,
                        )
                    };
                    (*window_size, array)
                })
                .collect();

        kv_cache.borrow().fill_attention_bias(
            &mut attention_bias_map,
            token_positions,
            suffix_length,
            &context,
            external_bias_fn,
        );

        let attention_bias: HashMap<Option<usize>, ArrayCell> =
            attention_bias_map
                .into_iter()
                .map(|(k, v)| (k, RefCell::new(v)))
                .collect();

        // --------------------
        // Logits
        // --------------------
        let logits = RefCell::new(unsafe {
            MetalArray::new(
                scratch.logits.clone(),
                &model_shape.logits_shape(suffix_length),
                act_dtype,
            )
        });

        let sampling_output = RefCell::new(unsafe {
            MetalArray::new(
                scratch.sampling_output.clone(),
                &[suffix_length],
                DataType::U32,
            )
        });

        let vocab_size = decoder_config.vocab_size;
        let token_bitmask = RefCell::new(unsafe {
            MetalArray::new(
                scratch.token_bitmask.clone(),
                &[suffix_length, vocab_size],
                DataType::U8,
            )
        });

        let traces = if trace {
            Some(Rc::new(RefCell::new(DecoderActivationTrace::new(
                &context,
                model_shape,
                suffix_length,
            ))))
        } else {
            None
        };

        Self {
            context,
            token_ids: token_ids_refcell,
            token_positions: token_positions_refcell,
            attention_bias,
            logits,
            kv_cache,
            shared_buffers,
            aux_buffers,
            sampling_output: Some(sampling_output),
            token_bitmask: Some(token_bitmask),
            sampling_method: None,
            traces,
        }
    }

    fn array_cell(
        &self,
        id: ArrayId,
    ) -> RefCell<MetalArray> {
        match id {
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::Logits => self.logits.clone(),
            ArrayId::Main => self.aux_buffers.main.clone(),
            ArrayId::Shortcut => self.aux_buffers.shortcut.clone(),
            ArrayId::QKV => self.aux_buffers.qkv.clone(),
            ArrayId::AttentionOutput => {
                self.aux_buffers.attention_output.clone()
            },
            ArrayId::MlpFusedUp => self.aux_buffers.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.aux_buffers.mlp_hidden.clone(),
            ArrayId::Keys(layer_index) => {
                self.kv_cache.borrow().data[layer_index].keys.clone()
            },
            ArrayId::Values(layer_index) => {
                self.kv_cache.borrow().data[layer_index].values.clone()
            },
            ArrayId::RotatedQueries => self.aux_buffers.rotated_queries.clone(),
            ArrayId::RotatedKeys => self.aux_buffers.rotated_keys.clone(),
            ArrayId::AttentionPartials => {
                self.aux_buffers.attention_partials.clone()
            },
            ArrayId::AttentionSums => self.aux_buffers.attention_sums.clone(),
            ArrayId::AttentionMaxs => self.aux_buffers.attention_maxs.clone(),

            ArrayId::EmbeddingsInputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        weights,
                    } => weights.clone(),
                    EmbeddingsBuffers::Untied {
                        input_weights,
                        ..
                    } => input_weights.clone(),
                    EmbeddingsBuffers::QuantizedTied {
                        weights,
                        ..
                    } => weights.clone(),
                }
            },
            ArrayId::EmbeddingsOutputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        weights,
                    } => weights.clone(),
                    EmbeddingsBuffers::Untied {
                        output_weights,
                        ..
                    } => output_weights.clone(),
                    EmbeddingsBuffers::QuantizedTied {
                        weights,
                        ..
                    } => weights.clone(),
                }
            },
            ArrayId::EmbeddingsScales => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::QuantizedTied {
                        scales,
                        ..
                    } => scales.clone(),
                    _ => panic!("Expected EmbeddingsBuffers::QuantizedTied"),
                }
            },
            ArrayId::RopeCosines(rope_type) => match rope_type {
                RopeType::Global => {
                    self.shared_buffers.borrow().global_rope.cosines.clone()
                },
                RopeType::Local => self
                    .shared_buffers
                    .borrow()
                    .local_rope
                    .as_ref()
                    .expect("Local rope requested but local_rope buffers are not initialized")
                    .cosines
                    .clone(),
            },
            ArrayId::RopeSines(rope_type) => match rope_type {
                RopeType::Global => {
                    self.shared_buffers.borrow().global_rope.sines.clone()
                },
                RopeType::Local => self
                    .shared_buffers
                    .borrow()
                    .local_rope
                    .as_ref()
                    .expect("Local rope requested but local_rope buffers are not initialized")
                    .sines
                    .clone(),
            },

            ArrayId::MoeTopkIds => self.aux_buffers.moe_topk_ids.as_ref().expect("MoE topk ids buffer not initialized").clone(),
            ArrayId::MoeTopkProbs => self.aux_buffers.moe_topk_probs.as_ref().expect("MoE topk probs buffer not initialized").clone(),
            ArrayId::MoeOffsets => self.aux_buffers.moe_offsets.as_ref().expect("MoE offsets buffer not initialized").clone(),
            ArrayId::MoeSumK => self.aux_buffers.moe_sumk.as_ref().expect("MoE sumk buffer not initialized").clone(),
            ArrayId::MoeBucketedTokenIds => self.aux_buffers.moe_bucketed_token_ids.as_ref().expect("MoE bucketed token ids buffer not initialized").clone(),
            ArrayId::MoeBucketedProbs => self.aux_buffers.moe_bucketed_probs.as_ref().expect("MoE bucketed probs buffer not initialized").clone(),
            ArrayId::MoeXPerm => self.aux_buffers.moe_x_perm.as_ref().expect("MoE x_perm buffer not initialized").clone(),
            ArrayId::MoeTok2Row => self.aux_buffers.moe_tok2row.as_ref().expect("MoE tok2row buffer not initialized").clone(),
            ArrayId::MoeYPartial => self.aux_buffers.moe_y_partial.as_ref().expect("MoE y_partial buffer not initialized").clone(),
            ArrayId::MoeHidden => self.aux_buffers.moe_hidden.as_ref().expect("MoE hidden buffer not initialized").clone(),
            ArrayId::MoeTwoPassRowExpertMap => self.aux_buffers.moe_two_pass_row_expert_map.as_ref().expect("MoE two-pass row expert map buffer not initialized").clone(),
            ArrayId::MoeTileCounts => self.aux_buffers.moe_tile_counts.as_ref().expect("MoE tile counts buffer not initialized").clone(),
            ArrayId::MoeTileOffsets => self.aux_buffers.moe_tile_offsets.as_ref().expect("MoE tile offsets buffer not initialized").clone(),
            ArrayId::MoeTileMap => self.aux_buffers.moe_tile_map.as_ref().expect("MoE tile map buffer not initialized").clone(),
            ArrayId::MoeTotalTiles => self.aux_buffers.moe_total_tiles.as_ref().expect("MoE total tiles buffer not initialized").clone(),
            ArrayId::MoeDispatchArgs => self.aux_buffers.moe_dispatch_args.as_ref().expect("MoE dispatch args buffer not initialized").clone(),
            ArrayId::MoeScatterPartials => self.aux_buffers.moe_scatter_partials.as_ref().expect("MoE scatter partials buffer not initialized").clone(),
            ArrayId::MoeScatterBlockBases => self.aux_buffers.moe_scatter_block_bases.as_ref().expect("MoE scatter block bases buffer not initialized").clone(),
            ArrayId::MoeBlockAlloc => self.aux_buffers.moe_block_alloc.as_ref().expect("MoE block alloc buffer not initialized").clone(),

            ArrayId::AttentionSinks(layer_index) => {
                self.shared_buffers.borrow().attention_sinks.as_ref()
                    .expect("Attention sinks not initialized")[layer_index].clone()
            },
        }
    }

    pub fn hashmap_cell(
        &self,
        id: &HashMapId,
    ) -> RefCell<HashMap<Option<usize>, ArrayCell>> {
        match id {
            HashMapId::AttentionBias => {
                RefCell::new(self.attention_bias.clone())
            },
        }
    }

    pub fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        ids.iter().map(|id| self.hashmap_cell(id).into_inner()).collect()
    }

    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter().map(|id| self.array_cell(*id).clone()).collect()
    }

    pub fn copy_array(
        &self,
        source_array_id: ArrayId,
        destination_array: RefCell<MetalArray>,
    ) {
        destination_array
            .borrow_mut()
            .copy_from_array(&self.arrays(&[source_array_id])[0].borrow());
    }

    pub fn aux_buffers_suffix_length(&self) -> usize {
        self.aux_buffers.suffix_length()
    }

    pub fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }
}

impl Drop for ForwardPassState {
    fn drop(&mut self) {
        // Nothing extra to clean up now that heap is removed.
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum RopeType {
    Global,
    Local,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum ArrayId {
    TokenIds,
    TokenPositions,
    Logits,

    Main,
    Shortcut,
    QKV,
    AttentionOutput,
    MlpFusedUp,
    MlpHidden,

    Keys(usize),
    Values(usize),

    RotatedQueries,
    RotatedKeys,

    AttentionPartials,
    AttentionSums,
    AttentionMaxs,

    EmbeddingsInputWeights,
    EmbeddingsOutputWeights,
    EmbeddingsScales,
    RopeCosines(RopeType),
    RopeSines(RopeType),

    AttentionSinks(usize),

    MoeTopkIds,
    MoeTopkProbs,
    MoeOffsets,
    MoeSumK,
    MoeBucketedTokenIds,
    MoeBucketedProbs,
    MoeXPerm,
    MoeTok2Row,
    MoeYPartial,
    MoeHidden,
    MoeTwoPassRowExpertMap,
    MoeTileCounts,
    MoeTileOffsets,
    MoeTileMap,
    MoeTotalTiles,
    MoeDispatchArgs,
    MoeScatterPartials,
    MoeScatterBlockBases,
    MoeBlockAlloc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashMapId {
    AttentionBias,
}
