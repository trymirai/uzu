#![allow(dead_code)]
use std::{cell::RefCell, collections::HashMap, rc::Rc};

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
    config::{DecoderConfig, EmbeddingConfig},
    parameters::ParameterTree,
};

type ArrayCell = RefCell<MetalArray>;

pub enum EmbeddingsBuffers {
    Tied {
        /// [vocab_size, model_dim]
        embeddings: ArrayCell,
    },
    Untied {
        /// [vocab_size, model_dim]
        input: ArrayCell,
        /// [vocab_size, model_dim]
        output: ArrayCell,
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
                    embeddings: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                EmbeddingConfig::Untied {
                    common: _,
                    precision: _,
                } => Self::Untied {
                    input: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                    output: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_output_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                _ => {
                    unimplemented!()
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
                embeddings,
            } => {
                let embeddings_view =
                    embeddings_tree.leaf("token_embeddings").unwrap();
                embeddings.borrow_mut().copy_from_array(&embeddings_view);
            },
            EmbeddingsBuffers::Untied {
                input,
                output,
            } => {
                let mapping =
                    vec![("input_weights", input), ("output_weights", output)];
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

        Self {
            embeddings,
            global_rope,
            local_rope,
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
    /// Current sampling configuration for this forward pass
    pub sampling_config:
        Option<crate::session::sampling_config::SamplingConfig>,
    pub traces: Option<Rc<RefCell<DecoderActivationTrace>>>,
}

impl ForwardPassState {
    pub fn new(
        context: Rc<MTLContext>,
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
        let aux_buffers = AuxBuffers::new(scratch, model_shape, suffix_length);

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
            sampling_config: None,
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

            ArrayId::EmbeddingsInput => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        embeddings,
                    } => embeddings.clone(),
                    EmbeddingsBuffers::Untied {
                        input,
                        output: _,
                    } => input.clone(),
                }
            },
            ArrayId::EmbeddingsOutput => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        embeddings,
                    } => embeddings.clone(),
                    EmbeddingsBuffers::Untied {
                        input: _,
                        output,
                    } => output.clone(),
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
                    .unwrap()
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
                    .unwrap()
                    .sines
                    .clone(),
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

    Keys(usize),
    Values(usize),

    RotatedQueries,
    RotatedKeys,

    AttentionPartials,
    AttentionSums,
    AttentionMaxs,

    EmbeddingsInput,
    EmbeddingsOutput,
    RopeCosines(RopeType),
    RopeSines(RopeType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashMapId {
    AttentionBias,
}
