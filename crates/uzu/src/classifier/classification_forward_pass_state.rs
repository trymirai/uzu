use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    DataType, DeviceContext,
    backends::metal::{
        MTLContext, MetalArray,
        forward_pass::{
            ArrayId, ForwardPassBuffers, ForwardPassStateTrait, HashMapId,
            ModelShape, RopeType, SharedBuffers,
        },
    },
    config::DecoderConfig,
};

type ArrayCell = RefCell<MetalArray>;

/// Simplified forward pass state for classification models (BERT, etc.)
/// Unlike ForwardPassState which is designed for autoregressive LLMs, this:
/// - Doesn't allocate KV cache (classification uses bidirectional attention)
/// - Doesn't allocate vocabulary-sized logits buffer (only num_labels needed)
/// - Supports both transformer operations [seq_len, model_dim] and prediction head [batch, features]
pub struct ClassificationForwardPassState {
    context: Rc<MTLContext>,
    /// [suffix_length] - u64
    token_ids: ArrayCell,
    /// [suffix_length] - i32
    token_positions: ArrayCell,
    /// [suffix_length, suffix_length] - attention bias (bidirectional, no prefix)
    attention_bias: HashMap<Option<usize>, ArrayCell>,
    /// Shared buffers (embeddings, RoPE)
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    /// Auxiliary buffers for transformer operations
    aux_buffers: AuxBuffers,
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
}

impl AuxBuffers {
    fn new(
        scratch: &ForwardPassBuffers,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        Self {
            suffix_length,
            main: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.main.clone(),
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            shortcut: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.shortcut.clone(),
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            qkv: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.qkv.clone(),
                    &model_shape.qkv_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_output: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_output.clone(),
                    &model_shape.attention_output_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            mlp_fused_up: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.mlp_fused_up.clone(),
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            mlp_hidden: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.mlp_hidden.clone(),
                    &model_shape.mlp_hidden_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            rotated_queries: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.rotated_queries.clone(),
                    &model_shape.rotated_queries_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            rotated_keys: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.rotated_keys.clone(),
                    &model_shape.rotated_keys_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_partials: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_partials.clone(),
                    &model_shape.attention_partials_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_sums: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_sums.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    DataType::F32,
                )
            }),
            attention_maxs: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_maxs.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    DataType::F32,
                )
            }),
        }
    }

    fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}

impl ClassificationForwardPassState {
    pub fn new(
        context: Rc<MTLContext>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        scratch: &ForwardPassBuffers,
        shared_buffers: Rc<RefCell<SharedBuffers>>,
        token_ids: &[u64],
        token_positions: &[usize],
        bidirectional_attention: bool, // true for BERT-style bidirectional attention
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(
            suffix_length,
            token_positions.len(),
            "Tokens and token positions must have the same length"
        );

        let aux_buffers = AuxBuffers::new(
            scratch,
            decoder_config,
            model_shape,
            suffix_length,
        );

        // Token IDs
        let mut token_ids_array = unsafe {
            MetalArray::new(
                scratch.token_ids.clone(),
                &[suffix_length],
                DataType::U64,
            )
        };
        context.copy_from_view(&mut token_ids_array, token_ids.into());
        let token_ids_refcell = RefCell::new(token_ids_array);

        // Token Positions (i32)
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

        // Attention Bias (for bidirectional attention, all zeros = no masking)
        let act_dtype = model_shape.activation_data_type();
        let mut attention_bias_map: HashMap<Option<usize>, MetalArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
                    // For classification, we only have suffix (no prefix/KV cache)
                    let attention_bias_shape = [suffix_length, suffix_length];

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

        // Fill attention bias: For bidirectional attention, all values are 0 (no masking)
        // For causal attention, we'd set upper triangle to -inf
        for (_window, bias_array) in attention_bias_map.iter_mut() {
            if bidirectional_attention {
                // All zeros = no masking (full bidirectional attention)
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    0,                  // no prefix
                    |_row, _col| false, // never mask
                );
            } else {
                // Causal masking
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    0,
                    |row, col| row < col, // mask future tokens
                );
            }
        }

        let attention_bias: HashMap<Option<usize>, ArrayCell> =
            attention_bias_map
                .into_iter()
                .map(|(k, v)| (k, RefCell::new(v)))
                .collect();

        Self {
            context,
            token_ids: token_ids_refcell,
            token_positions: token_positions_refcell,
            attention_bias,
            shared_buffers,
            aux_buffers,
        }
    }

    fn array_cell(
        &self,
        id: ArrayId,
    ) -> ArrayCell {
        match id {
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::Main => self.aux_buffers.main.clone(),
            ArrayId::Shortcut => self.aux_buffers.shortcut.clone(),
            ArrayId::QKV => self.aux_buffers.qkv.clone(),
            ArrayId::AttentionOutput => {
                self.aux_buffers.attention_output.clone()
            },
            ArrayId::MlpFusedUp => self.aux_buffers.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.aux_buffers.mlp_hidden.clone(),
            ArrayId::RotatedQueries => self.aux_buffers.rotated_queries.clone(),
            ArrayId::RotatedKeys => self.aux_buffers.rotated_keys.clone(),
            ArrayId::AttentionPartials => {
                self.aux_buffers.attention_partials.clone()
            },
            ArrayId::AttentionSums => self.aux_buffers.attention_sums.clone(),
            ArrayId::AttentionMaxs => self.aux_buffers.attention_maxs.clone(),

            // Shared buffers access
            ArrayId::EmbeddingsInputWeights => {
                use crate::backends::metal::forward_pass::EmbeddingsBuffers;
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
                use crate::backends::metal::forward_pass::EmbeddingsBuffers;
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
                use crate::backends::metal::forward_pass::EmbeddingsBuffers;
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
                    .expect("Local rope requested but not initialized")
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
                    .expect("Local rope requested but not initialized")
                    .sines
                    .clone(),
            },

            ArrayId::Logits => {
                panic!(
                    "ClassificationForwardPassState doesn't support Logits array - use Main for classifier output"
                )
            },
            _ => panic!("Unsupported ArrayId for classifier: {:?}", id),
        }
    }

    fn hashmap_cell(
        &self,
        id: &HashMapId,
    ) -> &HashMap<Option<usize>, ArrayCell> {
        match id {
            HashMapId::AttentionBias => &self.attention_bias,
        }
    }

    pub fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        ids.iter().map(|id| self.hashmap_cell(id).clone()).collect()
    }

    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter().map(|id| self.array_cell(*id)).collect()
    }

    pub fn aux_buffers_suffix_length(&self) -> usize {
        self.aux_buffers.suffix_length()
    }

    pub fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }
}

impl ForwardPassStateTrait for ClassificationForwardPassState {
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter().map(|id| self.array_cell(*id)).collect()
    }

    fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        ids.iter().map(|id| self.hashmap_cell(id).clone()).collect()
    }

    fn aux_buffers_suffix_length(&self) -> usize {
        self.aux_buffers.suffix_length()
    }

    fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }

    fn shared_buffers(&self) -> &Rc<RefCell<SharedBuffers>> {
        &self.shared_buffers
    }

    fn kv_cache(
        &self
    ) -> Option<&Rc<RefCell<crate::backends::metal::forward_pass::KVCache>>>
    {
        None
    }

    fn sampling_output(&self) -> Option<&ArrayCell> {
        None
    }

    fn traces(
        &self,
    ) -> Option<
        &Rc<
            RefCell<crate::backends::metal::forward_pass::traces::DecoderActivationTrace>,
        >,
    >{
        None
    }
}
