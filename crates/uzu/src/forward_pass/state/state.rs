#[cfg(feature = "tracing")]
use std::ops::{Deref, DerefMut};
use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayView2;

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{Backend, Context, Encoder, kernel::kv_cache_update::KVCacheUpdate},
    config::DecoderConfig,
    forward_pass::{
        cache_layers::{CacheLayer, CacheLayers},
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, CommonAuxBuffers, LanguageModelGeneratorAuxBuffers, RopeType, SharedBuffers},
    },
    session::parameter::SamplingMethod,
};

pub enum ForwardPassMode<B: Backend> {
    LanguageModelGenerator(LanguageModelGeneratorModeState<B>),
    Classifier(ClassifierModeState<B>),
}

pub struct MaterializedTransformerLayer<B: Backend> {
    pub keys: Option<Array<B>>,
    pub values: Option<Array<B>>,
}

pub struct LanguageModelGeneratorModeState<B: Backend> {
    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,
    pub materialized_transformer_layers: Box<[Option<MaterializedTransformerLayer<B>>]>,
    pub token_seeds: Array<B>,
    pub logits: Array<B>,
    pub sampling_output: Option<Array<B>>,
    pub sampling_method: Option<SamplingMethod>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<B>>>,
    pub active_row_count: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub is_prefilling: bool,
}

pub struct ClassifierModeState<B: Backend> {
    pub pooling: Array<B>,
    pub dense: Array<B>,
    pub norm: Array<B>,
    pub classifier_logits: Array<B>,
    pub active_row_count: usize,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<B>>>,
}

pub struct ForwardPassState<B: Backend> {
    context: Rc<B::Context>,
    token_ids: Array<B>,
    pub token_subtrie_ranges: Option<Array<B>>,
    token_positions: Array<B>,
    token_parents: Array<B>,
    token_bitmask: Option<Array<B>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    pub common_aux: CommonAuxBuffers<B>,
    pub llm_aux: Option<LanguageModelGeneratorAuxBuffers<B>>,
    mode: ForwardPassMode<B>,
}

impl<B: Backend> ForwardPassState<B> {
    // ========================================================================
    // Common initialization helpers
    // ========================================================================

    fn init_token_ids(
        scratch: &ScratchBuffers<B>,
        token_ids: &[u64],
    ) -> Array<B> {
        let suffix_length = token_ids.len();
        let mut token_ids_array = scratch.token_ids.view(&[suffix_length]);
        token_ids_array.copy_from_view(token_ids.into());
        token_ids_array
    }

    fn init_token_positions(
        scratch: &ScratchBuffers<B>,
        token_positions: &[usize],
    ) -> Array<B> {
        let suffix_length = token_positions.len();
        let mut token_positions_array = scratch.token_positions.view(&[suffix_length]);
        let token_positions_i32: Box<[i32]> = token_positions.iter().map(|p| *p as i32).collect();
        token_positions_array.copy_from_view(token_positions_i32.as_ref().into());
        token_positions_array
    }

    fn init_token_parents(
        scratch: &ScratchBuffers<B>,
        token_positions: &[usize],
        sampling_start: usize,
        sampling_length: usize,
    ) -> Array<B> {
        let suffix_length = token_positions.len();
        let mut token_parents_array = scratch.token_parents.view(&[suffix_length]);
        {
            let parents = token_parents_array.as_slice_mut::<i32>();
            parents.fill(-1);

            if sampling_length > 0 {
                let root_pos = token_positions.get(sampling_start).copied().unwrap_or(0);

                let mut stack: Vec<usize> = Vec::new();

                for local_idx in 0..sampling_length {
                    let abs_idx = sampling_start + local_idx;
                    let Some(&pos) = token_positions.get(abs_idx) else {
                        break;
                    };
                    let depth = pos.saturating_sub(root_pos);

                    let parent_local: i32 = if depth == 0 {
                        -1
                    } else if let Some(&p) = stack.get(depth - 1) {
                        p as i32
                    } else {
                        debug_assert!(false, "invalid trie depth ordering: depth={depth}, stack_len={}", stack.len());
                        -1
                    };
                    parents[abs_idx] = parent_local;

                    if stack.len() <= depth {
                        stack.resize(depth + 1, 0);
                    }
                    stack[depth] = local_idx;
                    stack.truncate(depth + 1);
                }
            }
        }
        token_parents_array
    }

    // ========================================================================
    // LLM Constructor
    // ========================================================================

    #[allow(clippy::too_many_arguments)]
    pub fn new_llm(
        context: Rc<B::Context>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        scratch: &ScratchBuffers<B>,
        cache_layers: Rc<RefCell<CacheLayers<B>>>,
        shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
        token_ids: &[u64],
        token_subtrie_ranges: Option<&[[u32; 3]]>,
        token_positions: &[usize],
        token_bitmask: Option<&[u32]>,
        token_seeds: &[u64],
        active_row_count: usize,
        sampling_start: usize,
        sampling_length: usize,
        is_prefilling: bool,
        skip_token_ids_copy: bool,
        async_positions: Option<(Rc<RefCell<B::Buffer>>, usize)>,
        async_seeds: Option<(Rc<RefCell<B::Buffer>>, usize)>,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len(), "Tokens and positions must have same length");
        assert!(suffix_length <= cache_layers.borrow().max_suffix_length(), "Suffix length exceeds KV cache capacity");

        // Token IDs - optionally skip copy for async path
        let token_ids_cell = if skip_token_ids_copy {
            scratch.token_ids.view(&[suffix_length])
        } else {
            Self::init_token_ids(scratch, token_ids)
        };

        // Token subtrie ranges
        let token_subtrie_ranges_cell = token_subtrie_ranges.map(|subtrie_ranges| {
            let subtrie_ranges_shape = model_shape.subtrie_ranges_shape(suffix_length);
            let mut subtrie_ranges_array = scratch.token_subtrie_ranges.view(&subtrie_ranges_shape);
            subtrie_ranges_array.as_slice_mut::<u32>().fill(0);
            subtrie_ranges_array.copy_from_view(
                ArrayView2::from_shape((subtrie_ranges.len(), 3), bytemuck::cast_slice::<_, u32>(subtrie_ranges))
                    .unwrap(),
            );
            subtrie_ranges_array
        });

        // Token positions - use async buffer if provided
        let token_positions_cell = if let Some((async_buf, offset)) = async_positions {
            unsafe {
                Array::from_parts(async_buf, offset * std::mem::size_of::<i32>(), &[suffix_length], DataType::I32)
            }
        } else {
            Self::init_token_positions(scratch, token_positions)
        };

        // Trie parent indices (relative, within the sampling segment).
        // Only meaningful when sampling_length > 0.
        let token_parents_cell = Self::init_token_parents(scratch, token_positions, sampling_start, sampling_length);

        // Token bitmask
        let token_bitmask_cell = token_bitmask.map(|bitmask| {
            let bitmask_shape = model_shape.bitmask_shape(suffix_length);
            let mut bitmask_array = scratch.token_bitmask.view(&bitmask_shape);
            bitmask_array.as_slice_mut::<u32>().fill(0);
            bitmask_array.copy_from_view(bitmask.into());
            bitmask_array
        });

        // Token seeds - use async buffer if provided
        let token_seeds_cell = if let Some((async_buf, offset)) = async_seeds {
            unsafe {
                Array::from_parts(async_buf, offset * std::mem::size_of::<u64>(), &[suffix_length], DataType::U64)
            }
        } else {
            let mut token_seeds_array = scratch.token_seeds.view(&[suffix_length]);
            token_seeds_array.copy_from_view(token_seeds.into());
            token_seeds_array
        };

        // Logits
        let logits_cell = scratch.logits.view(&model_shape.logits_shape(suffix_length));

        // Sampling output
        let sampling_output = Some(scratch.sampling_output.view(&[suffix_length]));

        // Common aux buffers
        let common_aux = CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // LLM-specific aux buffers
        let llm_aux = Some(LanguageModelGeneratorAuxBuffers::new(scratch, decoder_config, model_shape, suffix_length));

        let materialized_transformer_layers = {
            let cache_layers = cache_layers.borrow();
            cache_layers
                .data
                .iter()
                .enumerate()
                .map(|(layer_index, layer)| match layer {
                    CacheLayer::Transformer(layer) if layer.uses_compressed_storage() => {
                        let skip_prefix_key_materialization = suffix_length == 1
                            && layer.keys.is_none()
                            && matches!(
                                layer.state,
                                crate::forward_pass::kv_cache_layer::KVCacheLayerState::Full { .. }
                            )
                            && layer.supports_compressed_prefix_attention_scores_for_single_decode();
                        let skip_prefix_value_materialization = suffix_length == 1
                            && layer.values.is_none()
                            && matches!(
                                layer.state,
                                crate::forward_pass::kv_cache_layer::KVCacheLayerState::Full { .. }
                            )
                            && layer.supports_value_row_decoding_for_single_decode()
                            && (layer.sparse_value.is_none() || active_row_count == 1);
                        let mut keys = layer.keys.is_none().then(|| {
                            context.create_array(
                                &layer.shape,
                                layer.data_type,
                                &format!("state_materialized_keys_{layer_index}"),
                            )
                        });
                        let mut values = layer.values.is_none().then(|| {
                            context.create_array(
                                &layer.shape,
                                layer.data_type,
                                &format!("state_materialized_values_{layer_index}"),
                            )
                        });
                        if let Some(materialized_keys) = keys.as_mut() {
                            if !skip_prefix_key_materialization {
                                layer.materialize_keys_into(materialized_keys);
                            }
                        }
                        if let Some(materialized_values) = values.as_mut() {
                            if !skip_prefix_value_materialization {
                                layer.materialize_values_into(materialized_values);
                            }
                        }

                        (keys.is_some() || values.is_some()).then_some(MaterializedTransformerLayer {
                            keys,
                            values,
                        })
                    },
                    _ => None,
                })
                .collect()
        };

        // Traces
        #[cfg(feature = "tracing")]
        let traces = Rc::new(RefCell::new(ActivationTrace::new_llm(context.as_ref(), model_shape, suffix_length)));

        let mode = ForwardPassMode::LanguageModelGenerator(LanguageModelGeneratorModeState {
            cache_layers,
            materialized_transformer_layers,
            token_seeds: token_seeds_cell,
            logits: logits_cell,
            sampling_output,
            sampling_method: None,
            #[cfg(feature = "tracing")]
            traces,
            active_row_count,
            sampling_start,
            sampling_length,
            is_prefilling,
        });
        Self {
            context,
            token_ids: token_ids_cell,
            token_subtrie_ranges: token_subtrie_ranges_cell,
            token_positions: token_positions_cell,
            token_parents: token_parents_cell,
            token_bitmask: token_bitmask_cell,
            shared_buffers,
            common_aux,
            llm_aux,
            mode,
        }
    }

    // ========================================================================
    // Classifier Constructor
    // ========================================================================

    #[allow(clippy::too_many_arguments)]
    pub fn new_classifier(
        context: Rc<B::Context>,
        model_shape: &ModelShape,
        scratch: &ScratchBuffers<B>,
        shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
        token_ids: &[u64],
        token_positions: &[usize],
        num_labels: usize,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let token_ids_cell = Self::init_token_ids(scratch, token_ids);
        let token_positions_cell = Self::init_token_positions(scratch, token_positions);
        let token_parents_cell = Self::init_token_parents(scratch, token_positions, 0, 0);

        // Common aux buffers
        let common_aux = CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // Classifier-specific buffers
        let model_dim = model_shape.main_shape(1)[1];
        let classifier_state =
            Self::init_classifier_buffers(context.as_ref(), model_shape, model_dim, num_labels, suffix_length);

        let mode = ForwardPassMode::Classifier(classifier_state);

        Self {
            context,
            token_ids: token_ids_cell,
            token_subtrie_ranges: None,
            token_positions: token_positions_cell,
            token_parents: token_parents_cell,
            token_bitmask: None,
            shared_buffers,
            common_aux,
            llm_aux: None,
            mode,
        }
    }

    #[cfg(feature = "tracing")]
    fn init_classifier_buffers(
        context: &B::Context,
        model_shape: &ModelShape,
        model_dim: usize,
        num_labels: usize,
        suffix_length: usize,
    ) -> ClassifierModeState<B> {
        let data_type = model_shape.activation_data_type();
        let batch_dim = 1;

        let create_buffer = |size: usize| -> Array<B> {
            let buffer_size = size * data_type.size_in_bytes();
            let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
            unsafe { Array::from_parts(Rc::new(RefCell::new(buffer)), 0, &[batch_dim, size / batch_dim], data_type) }
        };

        ClassifierModeState {
            pooling: create_buffer(batch_dim * model_dim),
            dense: create_buffer(batch_dim * model_dim),
            norm: create_buffer(batch_dim * model_dim),
            classifier_logits: {
                let buffer_size = batch_dim * num_labels * data_type.size_in_bytes();
                let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
                unsafe { Array::from_parts(Rc::new(RefCell::new(buffer)), 0, &[batch_dim, num_labels], data_type) }
            },
            active_row_count: suffix_length,
            traces: Rc::new(RefCell::new(ActivationTrace::new_classifier(
                context,
                model_shape,
                suffix_length,
                num_labels,
            ))),
        }
    }

    #[cfg(not(feature = "tracing"))]
    fn init_classifier_buffers(
        context: &B::Context,
        model_shape: &ModelShape,
        model_dim: usize,
        num_labels: usize,
        suffix_length: usize,
    ) -> ClassifierModeState<B> {
        let data_type = model_shape.activation_data_type();
        let batch_dim = 1;

        let create_buffer = |dims: &[usize]| -> Array<B> {
            let size: usize = dims.iter().product();
            let buffer_size = size * data_type.size_in_bytes();
            let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
            unsafe { Array::from_parts(Rc::new(RefCell::new(buffer)), 0, dims, data_type) }
        };

        ClassifierModeState {
            pooling: create_buffer(&[batch_dim, model_dim]),
            dense: create_buffer(&[batch_dim, model_dim]),
            norm: create_buffer(&[batch_dim, model_dim]),
            classifier_logits: create_buffer(&[batch_dim, num_labels]),
            active_row_count: suffix_length,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    pub fn context(&self) -> &B::Context {
        self.context.as_ref()
    }

    pub fn aux_buffers_suffix_length(&self) -> usize {
        self.common_aux.suffix_length
    }

    pub fn token_bitmask(&self) -> Option<&Array<B>> {
        self.token_bitmask.as_ref()
    }

    pub fn llm_state(&self) -> &LanguageModelGeneratorModeState<B> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    pub fn classifier_state(&self) -> &ClassifierModeState<B> {
        match &self.mode {
            ForwardPassMode::Classifier(state) => state,
            _ => panic!("Not in classifier mode"),
        }
    }

    pub fn array(
        &self,
        id: ArrayId,
    ) -> Array<B> {
        match id {
            // Common arrays
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::TokenParents => self.token_parents.clone(),
            ArrayId::TokenBitmask => self.token_bitmask.clone().expect("Token bitmask not available"),
            ArrayId::Main => self.common_aux.main.clone(),
            ArrayId::Shortcut => self.common_aux.shortcut.clone(),
            ArrayId::QKV => self.common_aux.qkv.clone(),
            ArrayId::Gate => self.common_aux.gate.clone().expect("Gate buffer not available"),
            ArrayId::AttentionOutput => self.common_aux.attention_output.clone(),
            ArrayId::MlpFusedUp => self.common_aux.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.common_aux.mlp_hidden.clone(),
            ArrayId::RotatedQueries => self.common_aux.rotated_queries.clone(),
            ArrayId::RotatedKeys => self.common_aux.rotated_keys.clone(),
            ArrayId::ExtractedValues => self.common_aux.extracted_values.clone(),
            ArrayId::AttentionPartials => self.common_aux.attention_partials.clone(),
            ArrayId::AttentionSums => self.common_aux.attention_sums.clone(),
            ArrayId::AttentionMaxs => self.common_aux.attention_maxs.clone(),

            // Shared buffer arrays
            ArrayId::RopeCosines(_) | ArrayId::RopeSines(_) => {
                self.shared_buffer_array(id).expect("Shared buffer array should be available")
            },

            // LLM-specific arrays
            ArrayId::Logits => self.llm_state().logits.clone(),
            ArrayId::TokenSeeds => self.llm_state().token_seeds.clone(),
            ArrayId::Keys(layer_index) => {
                if let Some(layer) = self.llm_state().materialized_transformer_layers[layer_index].as_ref() {
                    if let Some(keys) = &layer.keys {
                        return keys.clone();
                    }
                }
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_transformer()
                    .expect("Expected transformer layer")
                    .dense_keys()
                    .borrow()
                    .clone()
            },
            ArrayId::Values(layer_index) => {
                if let Some(layer) = self.llm_state().materialized_transformer_layers[layer_index].as_ref() {
                    if let Some(values) = &layer.values {
                        return values.clone();
                    }
                }
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_transformer()
                    .expect("Expected transformer layer")
                    .dense_values()
                    .borrow()
                    .clone()
            },
            ArrayId::AttentionSinks(layer_index) => {
                self.shared_buffers.borrow().attention_sinks.as_ref().expect("Attention sinks not initialized")
                    [layer_index]
                    .clone()
            },

            // SSM arrays (LLM only)
            ArrayId::SsmInProj => {
                self.llm_aux.as_ref().and_then(|aux| aux.ssm_inproj.clone()).expect("SSM inproj not initialized")
            },
            ArrayId::SsmPacked(_) => {
                self.llm_aux.as_ref().and_then(|aux| aux.ssm_packed.clone()).expect("SSM packed not initialized")
            },
            ArrayId::SsmConvState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_state_space().expect("Expected SSM layer").conv_state.clone()
            },
            ArrayId::SsmState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_state_space().expect("Expected SSM layer").ssm_state.clone()
            },
            ArrayId::SsmX(_) => self.llm_aux.as_ref().and_then(|aux| aux.ssm_x.clone()).expect("SSM x not initialized"),
            ArrayId::SsmB(_) => self.llm_aux.as_ref().and_then(|aux| aux.ssm_b.clone()).expect("SSM b not initialized"),
            ArrayId::SsmC(_) => self.llm_aux.as_ref().and_then(|aux| aux.ssm_c.clone()).expect("SSM c not initialized"),
            ArrayId::SsmDt(_) => {
                self.llm_aux.as_ref().and_then(|aux| aux.ssm_dt.clone()).expect("SSM dt not initialized")
            },
            ArrayId::SsmZ(_) => self.llm_aux.as_ref().and_then(|aux| aux.ssm_z.clone()).expect("SSM z not initialized"),
            ArrayId::ShortConvState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_short_conv().expect("Expected ShortConv layer").conv_state.clone()
            },
            ArrayId::ShortConvSuffixState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_short_conv().expect("Expected ShortConv layer").suffix_state.clone()
            },
            ArrayId::DeltaNetConvState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_delta_net().expect("Expected DeltaNet layer").conv_state.clone()
            },
            ArrayId::DeltaNetSsmState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_delta_net().expect("Expected DeltaNet layer").ssm_state.clone()
            },

            // MoE arrays (LLM only)
            ArrayId::MoeTopkIds => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_topk_ids.clone()).expect("MoE topk_ids not initialized")
            },
            ArrayId::MoeTopkProbs => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_topk_probs.clone())
                .expect("MoE topk_probs not initialized"),
            ArrayId::MoeOffsets => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_offsets.clone()).expect("MoE offsets not initialized")
            },
            ArrayId::MoeSumK => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_sumk.clone()).expect("MoE sumk not initialized")
            },
            ArrayId::MoeBucketedTokenIds => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_bucketed_token_ids.clone())
                .expect("MoE bucketed_token_ids not initialized"),
            ArrayId::MoeBucketedProbs => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_bucketed_probs.clone())
                .expect("MoE bucketed_probs not initialized"),
            ArrayId::MoeXPerm => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_x_perm.clone()).expect("MoE x_perm not initialized")
            },
            ArrayId::MoeTok2Row => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_tok2row.clone()).expect("MoE tok2row not initialized")
            },
            ArrayId::MoeYPartial => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_y_partial.clone()).expect("MoE y_partial not initialized")
            },
            ArrayId::MoeHidden => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_hidden.clone()).expect("MoE hidden not initialized")
            },
            ArrayId::MoeTwoPassRowExpertMap => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_two_pass_row_expert_map.clone())
                .expect("MoE two_pass_row_expert_map not initialized"),
            ArrayId::MoeTileCounts => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_tile_counts.clone())
                .expect("MoE tile_counts not initialized"),
            ArrayId::MoeTileOffsets => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_tile_offsets.clone())
                .expect("MoE tile_offsets not initialized"),
            ArrayId::MoeTileMap => {
                self.llm_aux.as_ref().and_then(|aux| aux.moe_tile_map.clone()).expect("MoE tile_map not initialized")
            },
            ArrayId::MoeTotalTiles => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_total_tiles.clone())
                .expect("MoE total_tiles not initialized"),
            ArrayId::MoeDispatchArgs => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_dispatch_args.clone())
                .expect("MoE dispatch_args not initialized"),
            ArrayId::MoeScatterPartials => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_scatter_partials.clone())
                .expect("MoE scatter_partials not initialized"),
            ArrayId::MoeScatterBlockBases => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_scatter_block_bases.clone())
                .expect("MoE scatter_block_bases not initialized"),
            ArrayId::MoeBlockAlloc => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_block_alloc.clone())
                .expect("MoE block_alloc not initialized"),

            // Classifier-specific arrays
            ArrayId::ClassifierPooling => self.classifier_state().pooling.clone(),
            ArrayId::ClassifierPredictionHeadDense => self.classifier_state().dense.clone(),
            ArrayId::ClassifierPredictionHeadNorm => self.classifier_state().norm.clone(),
            ArrayId::ClassifierPredictionHeadLogits => self.classifier_state().classifier_logits.clone(),
        }
    }

    fn shared_buffer_array(
        &self,
        id: ArrayId,
    ) -> Option<Array<B>> {
        let shared = self.shared_buffers.borrow();
        match id {
            ArrayId::RopeCosines(rope_type) => Some(match rope_type {
                RopeType::Global => shared.global_rope.as_ref().expect("Global rope not initialized").cosines.clone(),
                RopeType::Local => shared.local_rope.as_ref().expect("Local rope not initialized").cosines.clone(),
            }),
            ArrayId::RopeSines(rope_type) => Some(match rope_type {
                RopeType::Global => shared.global_rope.as_ref().expect("Global rope not initialized").sines.clone(),
                RopeType::Local => shared.local_rope.as_ref().expect("Local rope not initialized").sines.clone(),
            }),
            _ => None,
        }
    }

    pub fn conv_padded_buffer(&self) -> Option<Array<B>> {
        self.llm_aux.as_ref().and_then(|aux| aux.ssm_conv_padded.clone())
    }

    pub fn sparse_value_buffers_for(
        &self,
        keys: &Array<B>,
        values: &Array<B>,
    ) -> (Array<B>, Array<B>) {
        let mut shared = self.shared_buffers.borrow_mut();
        if shared.sparse_value_keys.is_none() {
            shared.sparse_value_keys = Some(self.context.create_array_uninitialized(
                keys.shape(),
                keys.data_type(),
                "shared_buffers_sparse_value_keys",
            ));
        }
        if shared.sparse_value_values.is_none() {
            shared.sparse_value_values = Some(self.context.create_array_uninitialized(
                values.shape(),
                values.data_type(),
                "shared_buffers_sparse_value_values",
            ));
        }
        (
            shared.sparse_value_keys.as_ref().expect("SparseValue key scratch must exist").clone(),
            shared.sparse_value_values.as_ref().expect("SparseValue value scratch must exist").clone(),
        )
    }

    // ========================================================================
    // Public API Methods (formerly trait methods)
    // ========================================================================

    pub fn active_row_count(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.active_row_count,
            ForwardPassMode::Classifier(state) => state.active_row_count,
        }
    }

    pub fn set_active_row_count(
        &mut self,
        active_row_count: usize,
    ) {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.active_row_count = active_row_count,
            ForwardPassMode::Classifier(state) => state.active_row_count = active_row_count,
        }
    }

    /// Start index (within the suffix batch) for which we need logits/sampling.
    pub fn sampling_start(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_start,
            ForwardPassMode::Classifier(_) => 0,
        }
    }

    /// Number of batch items for which we need logits/sampling.
    pub fn sampling_length(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_length,
            ForwardPassMode::Classifier(_) => self.common_aux.suffix_length,
        }
    }

    pub fn is_prefilling(&self) -> bool {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.is_prefilling,
            ForwardPassMode::Classifier(_) => true,
        }
    }

    pub fn cache_layers(&self) -> Option<&Rc<RefCell<CacheLayers<B>>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => Some(&state.cache_layers),
            ForwardPassMode::Classifier(_) => None,
        }
    }

    pub fn update_cache_after_acceptance(
        &self,
        context: &B::Context,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        let ForwardPassMode::LanguageModelGenerator(state) = &self.mode else {
            return;
        };
        let short_conv_commit_index = accepted_suffix_indices.last().copied().unwrap_or(0);
        let mut cache_layers = state.cache_layers.borrow_mut();
        assert_eq!(
            cache_layers.data.len(),
            state.materialized_transformer_layers.len(),
            "materialized transformer layer count must match cache layer count",
        );

        for (layer, materialized) in cache_layers.data.iter_mut().zip(state.materialized_transformer_layers.iter()) {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(
                    context,
                    accepted_suffix_indices,
                    suffix_start,
                    materialized.as_ref().and_then(|layer| layer.keys.as_ref()),
                    materialized.as_ref().and_then(|layer| layer.values.as_ref()),
                    encoder,
                    kv_cache_update,
                );
            } else if let Some(layer) = layer.as_short_conv_mut() {
                layer.commit_from_suffix_state_if_valid(short_conv_commit_index);
            }
        }
    }

    pub fn sampling_output(&self) -> Option<&Array<B>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_output.as_ref(),
            ForwardPassMode::Classifier(_) => None,
        }
    }

    #[cfg(feature = "tracing")]
    pub fn traces(&self) -> &Rc<RefCell<ActivationTrace<B>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => &state.traces,
            ForwardPassMode::Classifier(state) => &state.traces,
        }
    }

    pub fn sampling_method_mut(&mut self) -> Option<&mut Option<SamplingMethod>> {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => Some(&mut state.sampling_method),
            ForwardPassMode::Classifier(_) => None,
        }
    }

    pub fn sampling_method(&self) -> Option<SamplingMethod> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_method,
            ForwardPassMode::Classifier(_) => None,
        }
    }

    #[cfg(feature = "tracing")]
    pub fn encode_copy_array(
        &self,
        encoder: &mut Encoder<B>,
        source_array_id: ArrayId,
        destination_array: Array<B>,
    ) {
        let source_array = self.array(source_array_id);

        let src_buf_rc = source_array.buffer();
        let dst_buf_rc = destination_array.buffer();

        let copy_size_bytes = destination_array.size();
        debug_assert_eq!(destination_array.size(), source_array.size());

        encoder.encode_copy(
            src_buf_rc.borrow().deref(),
            0..copy_size_bytes,
            dst_buf_rc.borrow_mut().deref_mut(),
            0..copy_size_bytes,
        );
    }
}
