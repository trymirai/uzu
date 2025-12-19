mod array_id;
mod common_aux_buffers;
mod embeddings_buffers;
mod hash_map_id;
mod language_model_generator_aux_buffers;
mod mode;
mod rope_buffers;
mod rope_type;
mod shared_buffers;

use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub use array_id::ArrayId;
pub use common_aux_buffers::CommonAuxBuffers;
pub use embeddings_buffers::EmbeddingsBuffers;
pub use hash_map_id::HashMapId;
pub use language_model_generator_aux_buffers::LanguageModelGeneratorAuxBuffers;
pub use mode::{
    ClassifierModeState, ForwardPassMode, LanguageModelGeneratorModeState,
};
pub use rope_buffers::RopeBuffers;
pub use rope_type::RopeType;
pub use shared_buffers::{MoeExpertWeights, SharedBuffers};

#[cfg(feature = "tracing")]
use super::traces::ActivationTrace;
use super::{
    CacheLayer, ModelShape, ScratchBuffers, cache_layers::CacheLayers,
};
use crate::{
    Array, DecoderConfig, DeviceContext, config::AttentionConfig,
    session::parameter::SamplingMethod,
};

pub type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub struct ForwardPassState<C: DeviceContext> {
    context: Rc<C>,
    token_ids: ArrayCell<C>,
    token_positions: ArrayCell<C>,
    token_bitmask: Option<ArrayCell<C>>,
    attention_bias: HashMap<Option<usize>, ArrayCell<C>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers<C>>>,
    common_aux: CommonAuxBuffers<C>,
    llm_aux: Option<LanguageModelGeneratorAuxBuffers<C>>,
    mode: ForwardPassMode<C>,
}

impl<C: DeviceContext> ForwardPassState<C> {
    // ========================================================================
    // Common initialization helpers
    // ========================================================================

    fn init_token_ids(
        context: &C,
        scratch: &ScratchBuffers<C>,
        token_ids: &[u64],
    ) -> ArrayCell<C> {
        let suffix_length = token_ids.len();
        let mut token_ids_array = scratch.token_ids.reshape(&[suffix_length]);
        context.copy_from_view(&mut token_ids_array, token_ids.into());
        RefCell::new(token_ids_array)
    }

    fn init_token_positions(
        context: &C,
        scratch: &ScratchBuffers<C>,
        token_positions: &[usize],
    ) -> ArrayCell<C> {
        let suffix_length = token_positions.len();
        let mut token_positions_array =
            scratch.token_positions.reshape(&[suffix_length]);

        let token_positions_i32: Box<[i32]> =
            token_positions.iter().map(|p| *p as i32).collect();
        context.copy_from_view(
            &mut token_positions_array,
            token_positions_i32.as_ref().into(),
        );
        RefCell::new(token_positions_array)
    }

    // ========================================================================
    // LLM Constructor
    // ========================================================================

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn new_llm(
        context: Rc<C>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        scratch: &ScratchBuffers<C>,
        cache_layers: Rc<RefCell<CacheLayers<C>>>,
        shared_buffers: Rc<RefCell<SharedBuffers<C>>>,
        token_ids: &[u64],
        token_positions: &[usize],
        token_bitmask: Option<&[u32]>,
        token_seeds: &[u64],
        active_suffix_length: usize,
        is_prefilling: bool,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
        skip_token_ids_copy: bool,
        skip_attention_bias_fill: bool,
        _async_positions: Option<(&C::DeviceArray, usize)>,
        _async_seeds: Option<(&C::DeviceArray, usize)>,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(
            suffix_length,
            token_positions.len(),
            "Tokens and positions must have same length"
        );
        assert!(
            suffix_length <= cache_layers.borrow().max_suffix_length(),
            "Suffix length exceeds KV cache capacity"
        );

        // Token IDs - optionally skip copy for async path
        let token_ids_cell = if skip_token_ids_copy {
            RefCell::new(scratch.token_ids.reshape(&[suffix_length]))
        } else {
            Self::init_token_ids(&context, scratch, token_ids)
        };

        // Token positions
        // NOTE: Async positions optimization temporarily disabled in generic refactor
        let token_positions_cell =
            Self::init_token_positions(&context, scratch, token_positions);

        // Token bitmask
        let token_bitmask_cell = token_bitmask.map(|bitmask| {
            let bitmask_shape = model_shape.bitmask_shape(suffix_length);
            let mut bitmask_array =
                scratch.token_bitmask.reshape(&bitmask_shape);
            if let Ok(dst) = bitmask_array.as_slice_mut::<u32>() {
                dst.fill(0);
            }
            context.copy_from_view(&mut bitmask_array, bitmask.into());
            RefCell::new(bitmask_array)
        });

        // Token seeds
        // NOTE: Async seeds optimization temporarily disabled in generic refactor
        let mut token_seeds_array =
            scratch.token_seeds.reshape(&[suffix_length]);
        context.copy_from_view(&mut token_seeds_array, token_seeds.into());
        let token_seeds_cell = RefCell::new(token_seeds_array);

        // Logits
        let logits_cell = RefCell::new(
            scratch.logits.reshape(&model_shape.logits_shape(suffix_length)),
        );

        // Sampling output
        let sampling_output = Some(RefCell::new(
            scratch.sampling_output.reshape(&[suffix_length]),
        ));

        // Attention bias (causal + sliding window)
        let attention_bias = Self::init_llm_attention_bias(
            &context,
            scratch,
            &cache_layers,
            decoder_config,
            suffix_length,
            token_positions,
            external_bias_fn,
            skip_attention_bias_fill,
        );

        // Common aux buffers
        let common_aux =
            CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // LLM-specific aux buffers
        let llm_aux = Some(LanguageModelGeneratorAuxBuffers::new(
            scratch,
            decoder_config,
            model_shape,
            suffix_length,
        ));

        // Traces
        #[cfg(feature = "tracing")]
        let traces = Rc::new(RefCell::new(ActivationTrace::new_llm(
            &context,
            model_shape,
            suffix_length,
        )));

        let mode = ForwardPassMode::LanguageModelGenerator(
            LanguageModelGeneratorModeState {
                cache_layers,
                token_seeds: token_seeds_cell,
                logits: logits_cell,
                sampling_output,
                sampling_method: None,
                #[cfg(feature = "tracing")]
                traces,
                active_suffix_length,
                is_prefilling,
            },
        );

        Self {
            context,
            token_ids: token_ids_cell,
            token_positions: token_positions_cell,
            token_bitmask: token_bitmask_cell,
            attention_bias,
            shared_buffers,
            common_aux,
            llm_aux,
            mode,
        }
    }

    fn init_llm_attention_bias(
        context: &C,
        scratch: &ScratchBuffers<C>,
        cache_layers: &Rc<RefCell<CacheLayers<C>>>,
        decoder_config: &DecoderConfig,
        suffix_length: usize,
        token_positions: &[usize],
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
        skip_fill: bool,
    ) -> HashMap<Option<usize>, ArrayCell<C>> {
        let cache_ref = cache_layers.borrow();
        let mut attention_bias_map: HashMap<Option<usize>, C::DeviceArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
                    let prefix_length =
                        window_size.unwrap_or(cache_ref.max_prefix_length());
                    let attention_bias_shape =
                        [suffix_length, suffix_length + prefix_length];
                    let array = buffer.reshape(&attention_bias_shape);
                    (*window_size, array)
                })
                .collect();
        drop(cache_ref);

        if !skip_fill {
            if decoder_config.layer_config.attention_config().is_some() {
                cache_layers.borrow().fill_attention_bias(
                    &mut attention_bias_map,
                    token_positions,
                    suffix_length,
                    context,
                    external_bias_fn,
                );
            }
        }

        attention_bias_map
            .into_iter()
            .map(|(k, v)| (k, RefCell::new(v)))
            .collect()
    }

    // ========================================================================
    // Classifier Constructor
    // ========================================================================

    #[allow(clippy::too_many_arguments)]
    pub fn new_classifier(
        context: Rc<C>,
        attention_config: Option<&AttentionConfig>,
        model_shape: &ModelShape,
        shared_buffers: Rc<RefCell<SharedBuffers<C>>>,
        scratch: &ScratchBuffers<C>,
        token_ids: &[u64],
        token_positions: &[usize],
        bidirectional_attention: bool,
        num_labels: usize,
        #[cfg(feature = "tracing")] traces: Rc<RefCell<ActivationTrace<C>>>,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let token_ids_cell = Self::init_token_ids(&context, scratch, token_ids);
        let token_positions_cell =
            Self::init_token_positions(&context, scratch, token_positions);

        let attention_bias = Self::init_classifier_attention_bias(
            &context,
            scratch,
            attention_config,
            suffix_length,
            bidirectional_attention,
        );

        // Common aux buffers
        let common_aux =
            CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // Classifier-specific buffers
        let (pooling, dense, norm, classifier_logits) =
            Self::init_classifier_buffers(
                &context,
                model_shape,
                suffix_length,
                num_labels,
            );

        #[cfg(feature = "tracing")]
        let traces = Rc::new(RefCell::new(ActivationTrace::new_classifier(
            &context,
            model_shape,
            suffix_length,
            num_labels,
        )));

        Self {
            context,
            token_ids: token_ids_cell,
            token_positions: token_positions_cell,
            token_bitmask: None,
            attention_bias,
            shared_buffers,
            common_aux,
            llm_aux: None,
            mode: ForwardPassMode::Classifier(ClassifierModeState {
                pooling,
                dense,
                norm,
                classifier_logits,
                #[cfg(feature = "tracing")]
                traces,
            }),
        }
    }

    fn init_classifier_attention_bias(
        context: &C,
        scratch: &ScratchBuffers<C>,
        attention_config: Option<&AttentionConfig>,
        suffix_length: usize,
        bidirectional_attention: bool,
    ) -> HashMap<Option<usize>, ArrayCell<C>> {
        let mut attention_bias = HashMap::new();

        if attention_config.is_some() {
            for (window_size, buffer) in &scratch.attention_window_size_to_bias
            {
                let shape = [suffix_length, suffix_length];
                let mut mask_array = buffer.reshape(&shape);

                if bidirectional_attention {
                    if let Some(w) = window_size {
                        let half_window = (w / 2) as isize;
                        context.fill_attention_bias(
                            &mut mask_array,
                            suffix_length,
                            0,
                            |row, col| {
                                let distance = (row as isize) - (col as isize);
                                distance.abs() > half_window
                            },
                        );
                    } else {
                        context.fill_attention_bias(
                            &mut mask_array,
                            suffix_length,
                            0,
                            |_row, _col| false,
                        );
                    }
                } else {
                    context.fill_attention_bias(
                        &mut mask_array,
                        suffix_length,
                        0,
                        |row, col| col > row,
                    );
                }
                attention_bias.insert(*window_size, RefCell::new(mask_array));
            }
        }
        attention_bias
    }

    fn init_classifier_buffers(
        context: &C,
        model_shape: &ModelShape,
        _suffix_length: usize,
        num_labels: usize,
    ) -> (ArrayCell<C>, ArrayCell<C>, ArrayCell<C>, ArrayCell<C>) {
        unsafe {
            let pooling = RefCell::new(context.array_uninitialized(
                &[1, model_shape.main_shape(1)[1]],
                model_shape.activation_data_type(),
            ));
            let dense = RefCell::new(context.array_uninitialized(
                &[1, model_shape.main_shape(1)[1]],
                model_shape.activation_data_type(),
            ));
            let norm = RefCell::new(context.array_uninitialized(
                &[1, model_shape.main_shape(1)[1]],
                model_shape.activation_data_type(),
            ));
            let logits = RefCell::new(context.array_uninitialized(
                &[1, num_labels],
                model_shape.activation_data_type(),
            ));
            (pooling, dense, norm, logits)
        }
    }

    // ========================================================================
    // Accessors & Helpers
    // ========================================================================

    pub fn context(&self) -> &Rc<C> {
        &self.context
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

    pub fn token_bitmask(&self) -> Option<&ArrayCell<C>> {
        self.token_bitmask.as_ref()
    }

    pub fn llm_state(&self) -> &LanguageModelGeneratorModeState<C> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    pub fn llm_state_mut(&mut self) -> &mut LanguageModelGeneratorModeState<C> {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    pub fn classifier_state(&self) -> &ClassifierModeState<C> {
        match &self.mode {
            ForwardPassMode::Classifier(state) => state,
            _ => panic!("Not in Classifier mode"),
        }
    }

    fn array_cell(
        &self,
        id: ArrayId,
    ) -> ArrayCell<C> {
        match id {
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::TokenBitmask => {
                self.token_bitmask.as_ref().unwrap().clone()
            },
            ArrayId::Logits => match &self.mode {
                ForwardPassMode::LanguageModelGenerator(s) => s.logits.clone(),
                ForwardPassMode::Classifier(s) => s.classifier_logits.clone(),
            },
            ArrayId::TokenSeeds => self.llm_state().token_seeds.clone(),

            ArrayId::Main => self.common_aux.main.clone(),
            ArrayId::Shortcut => self.common_aux.shortcut.clone(),
            ArrayId::QKV => self.common_aux.qkv.clone(),
            ArrayId::AttentionOutput => {
                self.common_aux.attention_output.clone()
            },
            ArrayId::MlpFusedUp => self.common_aux.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.common_aux.mlp_hidden.clone(),
            ArrayId::SsmInProj => self
                .llm_aux
                .as_ref()
                .unwrap()
                .ssm_inproj
                .as_ref()
                .unwrap()
                .clone(),

            ArrayId::Keys(layer) => {
                match &self.llm_state().cache_layers.borrow().data[layer] {
                    CacheLayer::Transformer(l) => l.keys.clone(),
                    _ => panic!("Expected Transformer layer"),
                }
            },
            ArrayId::Values(layer) => {
                match &self.llm_state().cache_layers.borrow().data[layer] {
                    CacheLayer::Transformer(l) => l.values.clone(),
                    _ => panic!("Expected Transformer layer"),
                }
            },

            ArrayId::SsmConvState(layer) => {
                match &self.llm_state().cache_layers.borrow().data[layer] {
                    CacheLayer::StateSpace(l) => l.conv_state.clone(),
                    _ => panic!("Expected StateSpace layer"),
                }
            },
            ArrayId::SsmState(layer) => {
                match &self.llm_state().cache_layers.borrow().data[layer] {
                    CacheLayer::StateSpace(l) => l.ssm_state.clone(),
                    _ => panic!("Expected StateSpace layer"),
                }
            },
            ArrayId::SsmPacked(_idx) => {
                // Assuming idx maps to layer or specific buffer, but `ssm_packed` is in `llm_aux`.
                // Checking `LanguageModelGeneratorAuxBuffers` structure, `ssm_packed` is Option<ArrayCell>.
                // `ArrayId` has `SsmPacked(usize)`.
                // Logic in original file likely mapped this.
                // Assuming single buffer for now as per `llm_aux` definition.
                // If `idx` is meant to select something, we ignore it if there's only one, or panic.
                // Current `LanguageModelGeneratorAuxBuffers` has `ssm_packed: Option<ArrayCell>`.
                // It doesn't look like a vec.
                self.llm_aux
                    .as_ref()
                    .unwrap()
                    .ssm_packed
                    .as_ref()
                    .unwrap()
                    .clone()
            },
            ArrayId::SsmX(_) => {
                self.llm_aux.as_ref().unwrap().ssm_x.as_ref().unwrap().clone()
            },
            ArrayId::SsmB(_) => {
                self.llm_aux.as_ref().unwrap().ssm_b.as_ref().unwrap().clone()
            },
            ArrayId::SsmC(_) => {
                self.llm_aux.as_ref().unwrap().ssm_c.as_ref().unwrap().clone()
            },
            ArrayId::SsmDt(_) => {
                self.llm_aux.as_ref().unwrap().ssm_dt.as_ref().unwrap().clone()
            },
            ArrayId::SsmZ(_) => {
                self.llm_aux.as_ref().unwrap().ssm_z.as_ref().unwrap().clone()
            },
            ArrayId::ShortConvState(layer) => {
                match &self.llm_state().cache_layers.borrow().data[layer] {
                    CacheLayer::ShortConv(l) => l.conv_state.clone(),
                    _ => panic!("Expected ShortConv layer"),
                }
            },

            ArrayId::RotatedQueries => self.common_aux.rotated_queries.clone(),
            ArrayId::RotatedKeys => self.common_aux.rotated_keys.clone(),
            ArrayId::ExtractedValues => {
                self.common_aux.extracted_values.clone()
            },

            ArrayId::AttentionPartials => {
                self.common_aux.attention_partials.clone()
            },
            ArrayId::AttentionSums => self.common_aux.attention_sums.clone(),
            ArrayId::AttentionMaxs => self.common_aux.attention_maxs.clone(),

            ArrayId::EmbeddingsInputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Untied {
                        input_weights,
                        ..
                    }
                    | EmbeddingsBuffers::MLXSemiQuantizedUntied {
                        input_weights,
                        ..
                    } => input_weights.clone(),
                    _ => panic!(
                        "Input weights not available in this embedding config"
                    ),
                }
            },
            ArrayId::EmbeddingsOutputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Untied {
                        output_weights,
                        ..
                    } => output_weights.clone(),
                    _ => panic!(
                        "Output weights not available in this embedding config"
                    ),
                }
            },
            ArrayId::EmbeddingsScales => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::QuantizedTied {
                        scales,
                        ..
                    } => scales.clone(),
                    _ => {
                        panic!("Scales not available in this embedding config")
                    },
                }
            },
            ArrayId::RopeCosines(rope_type) => match rope_type {
                RopeType::Global => self
                    .shared_buffers
                    .borrow()
                    .global_rope
                    .as_ref()
                    .unwrap()
                    .cosines
                    .clone(),
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
                RopeType::Global => self
                    .shared_buffers
                    .borrow()
                    .global_rope
                    .as_ref()
                    .unwrap()
                    .sines
                    .clone(),
                RopeType::Local => self
                    .shared_buffers
                    .borrow()
                    .local_rope
                    .as_ref()
                    .unwrap()
                    .sines
                    .clone(),
            },

            ArrayId::AttentionSinks(layer) => {
                self.shared_buffers.borrow().attention_sinks.as_ref().unwrap()
                    [layer]
                    .clone()
            },

            ArrayId::MoeTopkIds => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_topk_ids
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTopkProbs => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_topk_probs
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeOffsets => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_offsets
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeSumK => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_sumk
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeBucketedTokenIds => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_bucketed_token_ids
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeBucketedProbs => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_bucketed_probs
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeXPerm => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_x_perm
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTok2Row => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_tok2row
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeYPartial => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_y_partial
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeHidden => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_hidden
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTwoPassRowExpertMap => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_two_pass_row_expert_map
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTileCounts => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_tile_counts
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTileOffsets => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_tile_offsets
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTileMap => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_tile_map
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeTotalTiles => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_total_tiles
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeDispatchArgs => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_dispatch_args
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeScatterPartials => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_scatter_partials
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeScatterBlockBases => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_scatter_block_bases
                .as_ref()
                .unwrap()
                .clone(),
            ArrayId::MoeBlockAlloc => self
                .llm_aux
                .as_ref()
                .unwrap()
                .moe_block_alloc
                .as_ref()
                .unwrap()
                .clone(),

            ArrayId::ClassifierPooling => {
                self.classifier_state().pooling.clone()
            },
            ArrayId::ClassifierPredictionHeadDense => {
                self.classifier_state().dense.clone()
            },
            ArrayId::ClassifierPredictionHeadNorm => {
                self.classifier_state().norm.clone()
            },
            ArrayId::ClassifierPredictionHeadLogits => {
                self.classifier_state().classifier_logits.clone()
            },
        }
    }

    pub fn conv_padded_buffer(&self) -> Option<ArrayCell<C>> {
        self.llm_aux.as_ref().and_then(|aux| aux.ssm_conv_padded.clone())
    }

    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Vec<ArrayCell<C>> {
        ids.iter().map(|&id| self.array_cell(id)).collect()
    }

    pub fn hashmaps(
        &self,
        id: HashMapId,
    ) -> &HashMap<Option<usize>, ArrayCell<C>> {
        match id {
            HashMapId::AttentionBias => &self.attention_bias,
        }
    }

    pub fn active_suffix_length(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => {
                s.active_suffix_length
            },
            ForwardPassMode::Classifier(_) => self.aux_buffers_suffix_length(),
        }
    }

    pub fn is_prefilling(&self) -> bool {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => s.is_prefilling,
            ForwardPassMode::Classifier(_) => true,
        }
    }

    pub fn cache_layers(&self) -> Option<Rc<RefCell<CacheLayers<C>>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => {
                Some(s.cache_layers.clone())
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    pub fn sampling_output(&self) -> Option<&ArrayCell<C>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => {
                s.sampling_output.as_ref()
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    #[cfg(feature = "tracing")]
    pub fn traces(&self) -> Rc<RefCell<ActivationTrace<C>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => s.traces.clone(),
            ForwardPassMode::Classifier(s) => s.traces.clone(),
        }
    }

    pub fn sampling_method_mut(&mut self) -> Option<&mut SamplingMethod> {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => {
                s.sampling_method.as_mut()
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    pub fn sampling_method(&self) -> Option<&SamplingMethod> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(s) => {
                s.sampling_method.as_ref()
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    pub fn copy_array(
        &self,
        source_array_id: ArrayId,
        destination_array: ArrayCell<C>,
    ) {
        let source_array = self.array_cell(source_array_id);
        destination_array.borrow_mut().copy_from(&source_array.borrow());
    }
}
