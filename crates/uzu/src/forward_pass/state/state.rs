use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    DataType, DecoderConfig,
    array::{Array, ArrayCell, ArrayCellExt},
    backends::common::{Backend, CommandBuffer, Context, CopyEncoder},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, CommonAuxBuffers, HashMapId, LanguageModelGeneratorAuxBuffers, RopeType, SharedBuffers},
    },
    session::parameter::SamplingMethod,
    utils::attention::fill_attention_bias,
};

pub enum ForwardPassMode<B: Backend> {
    LanguageModelGenerator(LanguageModelGeneratorModeState<B>),
    Classifier(ClassifierModeState<B>),
}

pub struct LanguageModelGeneratorModeState<B: Backend> {
    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,
    pub token_seeds: ArrayCell<B>,
    pub logits: ArrayCell<B>,
    pub sampling_output: Option<ArrayCell<B>>,
    pub sampling_method: Option<SamplingMethod>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<B>>>,
    pub active_suffix_length: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub is_prefilling: bool,
}

pub struct ClassifierModeState<B: Backend> {
    pub pooling: ArrayCell<B>,
    pub dense: ArrayCell<B>,
    pub norm: ArrayCell<B>,
    pub classifier_logits: ArrayCell<B>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<B>>>,
}

pub struct ForwardPassState<B: Backend> {
    context: Rc<B::Context>,
    token_ids: ArrayCell<B>,
    token_positions: ArrayCell<B>,
    token_parents: ArrayCell<B>,
    token_bitmask: Option<ArrayCell<B>>,
    attention_bias: HashMap<Option<usize>, ArrayCell<B>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    common_aux: CommonAuxBuffers<B>,
    llm_aux: Option<LanguageModelGeneratorAuxBuffers<B>>,
    mode: ForwardPassMode<B>,
}

impl<B: Backend> ForwardPassState<B> {
    // ========================================================================
    // Common initialization helpers
    // ========================================================================

    fn init_token_ids(
        scratch: &ScratchBuffers<B>,
        token_ids: &[u64],
    ) -> ArrayCell<B> {
        let suffix_length = token_ids.len();
        let token_ids_array = scratch.token_ids.view(&[suffix_length]);
        token_ids_array.borrow_mut().copy_from_view(token_ids.into());
        token_ids_array
    }

    fn init_token_positions(
        scratch: &ScratchBuffers<B>,
        token_positions: &[usize],
    ) -> ArrayCell<B> {
        let suffix_length = token_positions.len();
        let token_positions_array = scratch.token_positions.view(&[suffix_length]);
        let token_positions_i32: Box<[i32]> = token_positions.iter().map(|p| *p as i32).collect();
        token_positions_array.borrow_mut().copy_from_view(token_positions_i32.as_ref().into());
        token_positions_array
    }

    fn init_token_parents(
        scratch: &ScratchBuffers<B>,
        token_positions: &[usize],
        sampling_start: usize,
        sampling_length: usize,
    ) -> ArrayCell<B> {
        let suffix_length = token_positions.len();
        let token_parents_array = scratch.token_parents.view(&[suffix_length]);
        {
            let mut token_parents = token_parents_array.borrow_mut();
            let parents = token_parents.as_slice_mut::<i32>();
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
        token_positions: &[usize],
        token_bitmask: Option<&[u32]>,
        token_seeds: &[u64],
        active_suffix_length: usize,
        sampling_start: usize,
        sampling_length: usize,
        is_prefilling: bool,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
        skip_token_ids_copy: bool,
        should_fill_attention_bias: bool,
        async_positions: Option<(&B::NativeBuffer, usize)>,
        async_seeds: Option<(&B::NativeBuffer, usize)>,
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

        // Token positions - use async buffer if provided
        let token_positions_cell = if let Some((async_buf, offset)) = async_positions {
            let array = unsafe {
                Array::from_parts(
                    async_buf.clone(),
                    offset * std::mem::size_of::<i32>(),
                    &[suffix_length],
                    DataType::I32,
                )
            };
            RefCell::new(array)
        } else {
            Self::init_token_positions(scratch, token_positions)
        };

        // Trie parent indices (relative, within the sampling segment).
        // Only meaningful when sampling_length > 0.
        let token_parents_cell = Self::init_token_parents(scratch, token_positions, sampling_start, sampling_length);

        // Token bitmask
        let token_bitmask_cell = token_bitmask.map(|bitmask| {
            let bitmask_shape = model_shape.bitmask_shape(suffix_length);
            let bitmask_array = scratch.token_bitmask.view(&bitmask_shape);
            bitmask_array.borrow_mut().as_slice_mut::<u32>().fill(0);
            bitmask_array.borrow_mut().copy_from_view(bitmask.into());
            bitmask_array
        });

        // Token seeds - use async buffer if provided
        let token_seeds_cell = if let Some((async_buf, offset)) = async_seeds {
            let array = unsafe {
                Array::from_parts(
                    async_buf.clone(),
                    offset * std::mem::size_of::<u64>(),
                    &[suffix_length],
                    DataType::U64,
                )
            };
            RefCell::new(array)
        } else {
            let token_seeds_array = scratch.token_seeds.view(&[suffix_length]);
            token_seeds_array.borrow_mut().copy_from_view(token_seeds.into());
            token_seeds_array
        };

        // Logits
        let logits_cell = scratch.logits.view(&model_shape.logits_shape(suffix_length));

        // Sampling output
        let sampling_output = Some(scratch.sampling_output.view(&[suffix_length]));

        // Attention bias (causal + sliding window)
        let attention_bias = Self::init_llm_attention_bias(
            scratch,
            &cache_layers,
            suffix_length,
            token_positions,
            external_bias_fn,
            should_fill_attention_bias,
        );

        // Common aux buffers
        let common_aux = CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // LLM-specific aux buffers
        let llm_aux = Some(LanguageModelGeneratorAuxBuffers::new(scratch, decoder_config, model_shape, suffix_length));

        // Traces
        #[cfg(feature = "tracing")]
        let traces = Rc::new(RefCell::new(ActivationTrace::new_llm(context.as_ref(), model_shape, suffix_length)));

        let mode = ForwardPassMode::LanguageModelGenerator(LanguageModelGeneratorModeState {
            cache_layers,
            token_seeds: token_seeds_cell,
            logits: logits_cell,
            sampling_output,
            sampling_method: None,
            #[cfg(feature = "tracing")]
            traces,
            active_suffix_length,
            sampling_start,
            sampling_length,
            is_prefilling,
        });
        Self {
            context,
            token_ids: token_ids_cell,
            token_positions: token_positions_cell,
            token_parents: token_parents_cell,
            token_bitmask: token_bitmask_cell,
            attention_bias,
            shared_buffers,
            common_aux,
            llm_aux,
            mode,
        }
    }

    fn init_llm_attention_bias(
        scratch: &ScratchBuffers<B>,
        cache_layers: &Rc<RefCell<CacheLayers<B>>>,
        suffix_length: usize,
        token_positions: &[usize],
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
        should_fill_attention_bias: bool,
    ) -> HashMap<Option<usize>, ArrayCell<B>> {
        let cache_ref = cache_layers.borrow();
        let mut attention_bias_map: HashMap<Option<usize>, Array<B>> = scratch
            .attention_window_size_to_bias
            .iter()
            .map(|(window_size, buffer)| {
                let prefix_length = window_size.unwrap_or(cache_ref.max_prefix_length());
                let attention_bias_shape = [suffix_length, suffix_length + prefix_length];
                let array = buffer.borrow().view(&attention_bias_shape);
                (*window_size, array)
            })
            .collect();
        drop(cache_ref);

        // Use cache_layers' fill_attention_bias which properly handles
        // both causal masking and sliding window constraints
        // Skip fill for async decode passes after the first one (bias already set)
        if should_fill_attention_bias {
            cache_layers.borrow().fill_attention_bias(
                &mut attention_bias_map,
                token_positions,
                suffix_length,
                external_bias_fn,
            );
        }

        attention_bias_map.into_iter().map(|(k, v)| (k, RefCell::new(v))).collect()
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
        bidirectional_attention: bool,
        num_labels: usize,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let token_ids_cell = Self::init_token_ids(scratch, token_ids);
        let token_positions_cell = Self::init_token_positions(scratch, token_positions);
        let token_parents_cell = Self::init_token_parents(scratch, token_positions, 0, 0);

        // Attention bias (bidirectional or causal)
        let attention_bias = Self::init_classifier_attention_bias(scratch, suffix_length, bidirectional_attention);

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
            token_positions: token_positions_cell,
            token_parents: token_parents_cell,
            token_bitmask: None,
            attention_bias,
            shared_buffers,
            common_aux,
            llm_aux: None,
            mode,
        }
    }

    fn init_classifier_attention_bias(
        scratch: &ScratchBuffers<B>,
        suffix_length: usize,
        bidirectional_attention: bool,
    ) -> HashMap<Option<usize>, ArrayCell<B>> {
        let mut attention_bias_map: HashMap<Option<usize>, Array<B>> = scratch
            .attention_window_size_to_bias
            .iter()
            .map(|(window_size, buffer)| {
                let attention_bias_shape = [suffix_length, suffix_length];
                let array = buffer.borrow().view(&attention_bias_shape);
                (*window_size, array)
            })
            .collect();

        for (window, bias_array) in attention_bias_map.iter_mut() {
            if bidirectional_attention {
                if let Some(window_size) = window {
                    let half_window = (window_size / 2) as isize;
                    fill_attention_bias(bias_array, suffix_length, 0, |row, col| {
                        let distance = (row as isize) - (col as isize);
                        distance.abs() > half_window
                    });
                } else {
                    fill_attention_bias(bias_array, suffix_length, 0, |_row, _col| false);
                }
            } else {
                fill_attention_bias(bias_array, suffix_length, 0, |row, col| row < col);
            }
        }

        attention_bias_map.into_iter().map(|(k, v)| (k, RefCell::new(v))).collect()
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
        let batch_size = 1;

        let create_buffer = |size: usize| -> ArrayCell<B> {
            let buffer_size = size * data_type.size_in_bytes();
            let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
            RefCell::new(unsafe { Array::from_parts(buffer, 0, &[batch_size, size / batch_size], data_type) })
        };

        ClassifierModeState {
            pooling: create_buffer(batch_size * model_dim),
            dense: create_buffer(batch_size * model_dim),
            norm: create_buffer(batch_size * model_dim),
            classifier_logits: {
                let buffer_size = batch_size * num_labels * data_type.size_in_bytes();
                let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
                RefCell::new(unsafe { Array::from_parts(buffer, 0, &[batch_size, num_labels], data_type) })
            },
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
        _suffix_length: usize,
    ) -> ClassifierModeState<B> {
        let data_type = model_shape.activation_data_type();
        let batch_size = 1;

        let create_buffer = |dims: &[usize]| -> ArrayCell<B> {
            let size: usize = dims.iter().product();
            let buffer_size = size * data_type.size_in_bytes();
            let buffer = context.create_buffer(buffer_size).expect("Failed to create buffer");
            RefCell::new(unsafe { Array::from_parts(buffer, 0, dims, data_type) })
        };

        ClassifierModeState {
            pooling: create_buffer(&[batch_size, model_dim]),
            dense: create_buffer(&[batch_size, model_dim]),
            norm: create_buffer(&[batch_size, model_dim]),
            classifier_logits: create_buffer(&[batch_size, num_labels]),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    pub fn mtl_context(&self) -> &B::Context {
        self.context.as_ref()
    }

    pub fn aux_buffers_suffix_length(&self) -> usize {
        self.common_aux.suffix_length
    }

    pub fn is_llm(&self) -> bool {
        matches!(self.mode, ForwardPassMode::LanguageModelGenerator(_))
    }

    pub fn is_classifier(&self) -> bool {
        matches!(self.mode, ForwardPassMode::Classifier(_))
    }

    pub fn token_bitmask(&self) -> Option<&ArrayCell<B>> {
        self.token_bitmask.as_ref()
    }

    pub fn llm_state(&self) -> &LanguageModelGeneratorModeState<B> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    pub fn llm_state_mut(&mut self) -> &mut LanguageModelGeneratorModeState<B> {
        match &mut self.mode {
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

    pub fn array_cell(
        &self,
        id: ArrayId,
    ) -> ArrayCell<B> {
        match id {
            // Common arrays
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::TokenParents => self.token_parents.clone(),
            ArrayId::TokenBitmask => self.token_bitmask.clone().expect("Token bitmask not available"),
            ArrayId::Main => self.common_aux.main.clone(),
            ArrayId::Shortcut => self.common_aux.shortcut.clone(),
            ArrayId::QKV => self.common_aux.qkv.clone(),
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
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_transformer().expect("Expected transformer layer").keys.clone()
            },
            ArrayId::Values(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index].as_transformer().expect("Expected transformer layer").values.clone()
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
    ) -> Option<ArrayCell<B>> {
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

    pub fn conv_padded_buffer(&self) -> Option<ArrayCell<B>> {
        self.llm_aux.as_ref().and_then(|aux| aux.ssm_conv_padded.clone())
    }

    pub fn short_conv_padded_buffer(&self) -> Option<ArrayCell<B>> {
        self.llm_aux.as_ref().and_then(|aux| aux.short_conv_padded.clone())
    }

    // ========================================================================
    // Public API Methods (formerly trait methods)
    // ========================================================================

    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell<B>]> {
        ids.iter().map(|id| self.array_cell(*id)).collect()
    }

    pub fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell<B>>]> {
        ids.iter()
            .map(|id| match id {
                HashMapId::AttentionBias => self.attention_bias.clone(),
            })
            .collect()
    }

    pub fn active_suffix_length(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.active_suffix_length,
            ForwardPassMode::Classifier(_) => self.common_aux.suffix_length,
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

    pub fn sampling_output(&self) -> Option<&ArrayCell<B>> {
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

    pub fn copy_array(
        &self,
        source_array_id: ArrayId,
        destination_array: RefCell<Array<B>>,
    ) {
        destination_array.borrow_mut().copy_from_array(&self.arrays(&[source_array_id])[0].borrow());
    }

    pub fn encode_copy_array(
        &self,
        command_buffer: &B::CommandBuffer,
        source_array_id: ArrayId,
        destination_array: RefCell<Array<B>>,
    ) {
        let source_ref = self.arrays(&[source_array_id])[0].clone();
        let src_borrow = source_ref.borrow();
        let dst_borrow = destination_array.borrow();

        let src_buf = src_borrow.buffer();
        let dst_buf = dst_borrow.buffer();

        let copy_size_bytes = dst_borrow.size();
        debug_assert_eq!(dst_borrow.size(), src_borrow.size());

        command_buffer.with_copy_encoder(|encoder| {
            encoder.encode_copy(src_buf, dst_buf, copy_size_bytes);
        });
    }
}
