//! Unified forward pass state for both LLM and classifier models.
//!
//! This module consolidates `LLMForwardPassState` and `ClassificationForwardPassState`
//! into a single implementation to eliminate code duplication.

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
use super::{ModelShape, ScratchBuffers, cache_layers::CacheLayers};
use crate::{
    Array, DataType, DecoderConfig, DeviceContext,
    backends::metal::{MTLContext, MetalArray},
    session::parameter::SamplingMethod,
};

/// Type alias for array storage.
pub type ArrayCell = RefCell<MetalArray>;

/// Unified forward pass state for both LLM and classifier models.
pub struct ForwardPassState {
    context: Rc<MTLContext>,
    token_ids: ArrayCell,
    token_positions: ArrayCell,
    token_bitmask: Option<ArrayCell>,
    attention_bias: HashMap<Option<usize>, ArrayCell>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    common_aux: CommonAuxBuffers,
    llm_aux: Option<LanguageModelGeneratorAuxBuffers>,
    mode: ForwardPassMode,
}

impl ForwardPassState {
    // ========================================================================
    // Common initialization helpers
    // ========================================================================

    fn init_token_ids(
        context: &MTLContext,
        scratch: &ScratchBuffers,
        token_ids: &[u64],
    ) -> ArrayCell {
        let suffix_length = token_ids.len();
        let mut token_ids_array = unsafe {
            MetalArray::new(
                scratch.token_ids.clone(),
                &[suffix_length],
                DataType::U64,
            )
        };
        context.copy_from_view(&mut token_ids_array, token_ids.into());
        RefCell::new(token_ids_array)
    }

    fn init_token_positions(
        context: &MTLContext,
        scratch: &ScratchBuffers,
        token_positions: &[usize],
    ) -> ArrayCell {
        let suffix_length = token_positions.len();
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
        RefCell::new(token_positions_array)
    }

    // ========================================================================
    // LLM Constructor
    // ========================================================================

    /// Create a new forward pass state for LLM (autoregressive) mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_llm(
        context: Rc<MTLContext>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        scratch: &ScratchBuffers,
        cache_layers: Rc<RefCell<CacheLayers>>,
        shared_buffers: Rc<RefCell<SharedBuffers>>,
        token_ids: &[u64],
        token_positions: &[usize],
        token_bitmask: Option<&[u32]>,
        token_seeds: &[u64],
        active_suffix_length: usize,
        is_prefilling: bool,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
        skip_token_ids_copy: bool,
        async_positions: Option<(&metal::Buffer, usize)>,
        async_seeds: Option<(&metal::Buffer, usize)>,
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
            RefCell::new(unsafe {
                MetalArray::new(
                    scratch.token_ids.clone(),
                    &[suffix_length],
                    DataType::U64,
                )
            })
        } else {
            Self::init_token_ids(&context, scratch, token_ids)
        };

        // Token positions - use async buffer if provided
        let token_positions_cell =
            if let Some((async_buf, offset)) = async_positions {
                let array = unsafe {
                    MetalArray::new_with_offset(
                        async_buf.clone(),
                        &[suffix_length],
                        DataType::I32,
                        offset * std::mem::size_of::<i32>(),
                    )
                };
                RefCell::new(array)
            } else {
                Self::init_token_positions(&context, scratch, token_positions)
            };

        // Token bitmask
        let token_bitmask_cell = token_bitmask.map(|bitmask| {
            let bitmask_shape = model_shape.bitmask_shape(suffix_length);
            let mut bitmask_array = unsafe {
                MetalArray::new(
                    scratch.token_bitmask.clone(),
                    &bitmask_shape,
                    DataType::U32,
                )
            };
            context.copy_from_view(&mut bitmask_array, bitmask.into());
            RefCell::new(bitmask_array)
        });

        // Token seeds - use async buffer if provided
        let token_seeds_cell = if let Some((async_buf, offset)) = async_seeds {
            let array = unsafe {
                MetalArray::new_with_offset(
                    async_buf.clone(),
                    &[suffix_length],
                    DataType::U64,
                    offset * std::mem::size_of::<u64>(),
                )
            };
            RefCell::new(array)
        } else {
            let mut token_seeds_array = unsafe {
                MetalArray::new(
                    scratch.token_seeds.clone(),
                    &[suffix_length],
                    DataType::U64,
                )
            };
            context.copy_from_view(&mut token_seeds_array, token_seeds.into());
            RefCell::new(token_seeds_array)
        };

        // Logits
        let logits_cell = RefCell::new(unsafe {
            MetalArray::new(
                scratch.logits.clone(),
                &model_shape.logits_shape(suffix_length),
                model_shape.activation_data_type(),
            )
        });

        // Sampling output
        let sampling_output = Some(RefCell::new(unsafe {
            MetalArray::new(
                scratch.sampling_output.clone(),
                &[suffix_length],
                DataType::U32,
            )
        }));

        // Attention bias (causal)
        let act_dtype = model_shape.activation_data_type();
        let attention_bias = Self::init_llm_attention_bias(
            &context,
            scratch,
            &cache_layers,
            suffix_length,
            act_dtype,
            is_prefilling,
            external_bias_fn,
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
        context: &MTLContext,
        scratch: &ScratchBuffers,
        cache_layers: &Rc<RefCell<CacheLayers>>,
        suffix_length: usize,
        act_dtype: DataType,
        is_prefilling: bool,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> HashMap<Option<usize>, ArrayCell> {
        let cache_ref = cache_layers.borrow();
        let mut attention_bias_map: HashMap<Option<usize>, MetalArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
                    let prefix_length =
                        window_size.unwrap_or(cache_ref.max_prefix_length());
                    let attention_bias_shape =
                        [suffix_length, suffix_length + prefix_length];
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
        drop(cache_ref);

        let cache_ref = cache_layers.borrow();
        for (window, bias_array) in attention_bias_map.iter_mut() {
            let prefix_length = window.unwrap_or(cache_ref.max_prefix_length());
            if let Some(bias_fn) = external_bias_fn {
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    prefix_length,
                    bias_fn,
                );
            } else {
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    prefix_length,
                    |row, col| {
                        if is_prefilling {
                            row + prefix_length < col
                        } else {
                            prefix_length < col
                        }
                    },
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

    /// Create a new forward pass state for classifier (bidirectional) mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_classifier(
        context: Rc<MTLContext>,
        model_shape: &ModelShape,
        scratch: &ScratchBuffers,
        shared_buffers: Rc<RefCell<SharedBuffers>>,
        token_ids: &[u64],
        token_positions: &[usize],
        bidirectional_attention: bool,
        num_labels: usize,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let token_ids_cell = Self::init_token_ids(&context, scratch, token_ids);
        let token_positions_cell =
            Self::init_token_positions(&context, scratch, token_positions);

        // Attention bias (bidirectional or causal)
        let act_dtype = model_shape.activation_data_type();
        let attention_bias = Self::init_classifier_attention_bias(
            &context,
            scratch,
            suffix_length,
            act_dtype,
            bidirectional_attention,
        );

        // Common aux buffers
        let common_aux =
            CommonAuxBuffers::new(scratch, model_shape, suffix_length);

        // Classifier-specific buffers
        let model_dim = model_shape.main_shape(1)[1];
        let classifier_state = Self::init_classifier_buffers(
            &context,
            model_shape,
            model_dim,
            num_labels,
            suffix_length,
        );

        let mode = ForwardPassMode::Classifier(classifier_state);

        Self {
            context,
            token_ids: token_ids_cell,
            token_positions: token_positions_cell,
            token_bitmask: None,
            attention_bias,
            shared_buffers,
            common_aux,
            llm_aux: None,
            mode,
        }
    }

    fn init_classifier_attention_bias(
        context: &MTLContext,
        scratch: &ScratchBuffers,
        suffix_length: usize,
        act_dtype: DataType,
        bidirectional_attention: bool,
    ) -> HashMap<Option<usize>, ArrayCell> {
        let mut attention_bias_map: HashMap<Option<usize>, MetalArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
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

        for (window, bias_array) in attention_bias_map.iter_mut() {
            if bidirectional_attention {
                if let Some(window_size) = window {
                    let half_window = (window_size / 2) as isize;
                    context.fill_attention_bias(
                        bias_array,
                        suffix_length,
                        0,
                        |row, col| {
                            let distance = (row as isize) - (col as isize);
                            distance.abs() > half_window
                        },
                    );
                } else {
                    context.fill_attention_bias(
                        bias_array,
                        suffix_length,
                        0,
                        |_row, _col| false,
                    );
                }
            } else {
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    0,
                    |row, col| row < col,
                );
            }
        }

        attention_bias_map
            .into_iter()
            .map(|(k, v)| (k, RefCell::new(v)))
            .collect()
    }

    #[cfg(feature = "tracing")]
    fn init_classifier_buffers(
        context: &MTLContext,
        model_shape: &ModelShape,
        model_dim: usize,
        num_labels: usize,
        suffix_length: usize,
    ) -> ClassifierModeState {
        let data_type = model_shape.activation_data_type();
        let batch_size = 1;

        let create_buffer = |size: usize| -> ArrayCell {
            let buffer_size = (size * data_type.size_in_bytes()) as u64;
            let buffer = context.device.new_buffer(
                buffer_size,
                metal::MTLResourceOptions::StorageModeShared,
            );
            RefCell::new(unsafe {
                MetalArray::new(
                    buffer,
                    &[batch_size, size / batch_size],
                    data_type,
                )
            })
        };

        ClassifierModeState {
            pooling: create_buffer(batch_size * model_dim),
            dense: create_buffer(batch_size * model_dim),
            norm: create_buffer(batch_size * model_dim),
            classifier_logits: {
                let buffer_size =
                    (batch_size * num_labels * data_type.size_in_bytes())
                        as u64;
                let buffer = context.device.new_buffer(
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                RefCell::new(unsafe {
                    MetalArray::new(
                        buffer,
                        &[batch_size, num_labels],
                        data_type,
                    )
                })
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
        context: &MTLContext,
        model_shape: &ModelShape,
        model_dim: usize,
        num_labels: usize,
        _suffix_length: usize,
    ) -> ClassifierModeState {
        let data_type = model_shape.activation_data_type();
        let batch_size = 1;

        let create_buffer = |dims: &[usize]| -> ArrayCell {
            let size: usize = dims.iter().product();
            let buffer_size = (size * data_type.size_in_bytes()) as u64;
            let buffer = context.device.new_buffer(
                buffer_size,
                metal::MTLResourceOptions::StorageModeShared,
            );
            RefCell::new(unsafe { MetalArray::new(buffer, dims, data_type) })
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

    pub fn mtl_context(&self) -> &Rc<MTLContext> {
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

    /// Get token bitmask if available.
    pub fn token_bitmask(&self) -> Option<&ArrayCell> {
        self.token_bitmask.as_ref()
    }

    /// Get LLM-specific state (panics if not in LLM mode).
    pub fn llm_state(&self) -> &LanguageModelGeneratorModeState {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    /// Get mutable LLM-specific state (panics if not in LLM mode).
    pub fn llm_state_mut(&mut self) -> &mut LanguageModelGeneratorModeState {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state,
            _ => panic!("Not in LLM mode"),
        }
    }

    /// Get classifier-specific state (panics if not in classifier mode).
    pub fn classifier_state(&self) -> &ClassifierModeState {
        match &self.mode {
            ForwardPassMode::Classifier(state) => state,
            _ => panic!("Not in classifier mode"),
        }
    }

    /// Get array cell by ID.
    pub fn array_cell(
        &self,
        id: ArrayId,
    ) -> ArrayCell {
        match id {
            // Common arrays
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::TokenBitmask => {
                self.token_bitmask.clone().expect("Token bitmask not available")
            },
            ArrayId::Main => self.common_aux.main.clone(),
            ArrayId::Shortcut => self.common_aux.shortcut.clone(),
            ArrayId::QKV => self.common_aux.qkv.clone(),
            ArrayId::AttentionOutput => {
                self.common_aux.attention_output.clone()
            },
            ArrayId::MlpFusedUp => self.common_aux.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.common_aux.mlp_hidden.clone(),
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

            // Shared buffer arrays
            ArrayId::EmbeddingsInputWeights
            | ArrayId::EmbeddingsOutputWeights
            | ArrayId::EmbeddingsScales
            | ArrayId::RopeCosines(_)
            | ArrayId::RopeSines(_) => self
                .shared_buffer_array(id)
                .expect("Shared buffer array should be available"),

            // LLM-specific arrays
            ArrayId::Logits => self.llm_state().logits.clone(),
            ArrayId::TokenSeeds => self.llm_state().token_seeds.clone(),
            ArrayId::Keys(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_transformer()
                    .expect("Expected transformer layer")
                    .keys
                    .clone()
            },
            ArrayId::Values(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_transformer()
                    .expect("Expected transformer layer")
                    .values
                    .clone()
            },
            ArrayId::AttentionSinks(layer_index) => self
                .shared_buffers
                .borrow()
                .attention_sinks
                .as_ref()
                .expect("Attention sinks not initialized")[layer_index]
                .clone(),

            // SSM arrays (LLM only)
            ArrayId::SsmInProj => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_inproj.clone())
                .expect("SSM inproj not initialized"),
            ArrayId::SsmPacked(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_packed.clone())
                .expect("SSM packed not initialized"),
            ArrayId::SsmConvState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_state_space()
                    .expect("Expected SSM layer")
                    .conv_state
                    .clone()
            },
            ArrayId::SsmState(layer_index) => {
                let cache = self.llm_state().cache_layers.borrow();
                cache.data[layer_index]
                    .as_state_space()
                    .expect("Expected SSM layer")
                    .ssm_state
                    .clone()
            },
            ArrayId::SsmX(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_x.clone())
                .expect("SSM x not initialized"),
            ArrayId::SsmB(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_b.clone())
                .expect("SSM b not initialized"),
            ArrayId::SsmC(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_c.clone())
                .expect("SSM c not initialized"),
            ArrayId::SsmDt(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_dt.clone())
                .expect("SSM dt not initialized"),
            ArrayId::SsmZ(_) => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.ssm_z.clone())
                .expect("SSM z not initialized"),

            // MoE arrays (LLM only)
            ArrayId::MoeTopkIds => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_topk_ids.clone())
                .expect("MoE topk_ids not initialized"),
            ArrayId::MoeTopkProbs => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_topk_probs.clone())
                .expect("MoE topk_probs not initialized"),
            ArrayId::MoeOffsets => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_offsets.clone())
                .expect("MoE offsets not initialized"),
            ArrayId::MoeSumK => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_sumk.clone())
                .expect("MoE sumk not initialized"),
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
            ArrayId::MoeXPerm => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_x_perm.clone())
                .expect("MoE x_perm not initialized"),
            ArrayId::MoeTok2Row => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_tok2row.clone())
                .expect("MoE tok2row not initialized"),
            ArrayId::MoeYPartial => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_y_partial.clone())
                .expect("MoE y_partial not initialized"),
            ArrayId::MoeHidden => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_hidden.clone())
                .expect("MoE hidden not initialized"),
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
            ArrayId::MoeTileMap => self
                .llm_aux
                .as_ref()
                .and_then(|aux| aux.moe_tile_map.clone())
                .expect("MoE tile_map not initialized"),
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

    fn shared_buffer_array(
        &self,
        id: ArrayId,
    ) -> Option<ArrayCell> {
        let shared = self.shared_buffers.borrow();
        match id {
            ArrayId::EmbeddingsInputWeights => Some(match &shared.embeddings {
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
                EmbeddingsBuffers::MLXSemiQuantizedUntied {
                    input_weights,
                    ..
                } => input_weights.clone(),
            }),
            ArrayId::EmbeddingsOutputWeights => {
                Some(match &shared.embeddings {
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
                    EmbeddingsBuffers::MLXSemiQuantizedUntied {
                        packed_output_weights,
                        ..
                    } => packed_output_weights.clone(),
                })
            },
            ArrayId::EmbeddingsScales => Some(match &shared.embeddings {
                EmbeddingsBuffers::QuantizedTied {
                    scales,
                    ..
                } => scales.clone(),
                EmbeddingsBuffers::MLXSemiQuantizedUntied {
                    output_scales,
                    ..
                } => output_scales.clone(),
                _ => panic!(
                    "Expected QuantizedTied or MLXSemiQuantizedUntied embeddings"
                ),
            }),
            ArrayId::RopeCosines(rope_type) => Some(match rope_type {
                RopeType::Global => shared
                    .global_rope
                    .as_ref()
                    .expect("Global rope not initialized")
                    .cosines
                    .clone(),
                RopeType::Local => shared
                    .local_rope
                    .as_ref()
                    .expect("Local rope not initialized")
                    .cosines
                    .clone(),
            }),
            ArrayId::RopeSines(rope_type) => Some(match rope_type {
                RopeType::Global => shared
                    .global_rope
                    .as_ref()
                    .expect("Global rope not initialized")
                    .sines
                    .clone(),
                RopeType::Local => shared
                    .local_rope
                    .as_ref()
                    .expect("Local rope not initialized")
                    .sines
                    .clone(),
            }),
            _ => None,
        }
    }

    pub fn conv_padded_buffer(&self) -> Option<ArrayCell> {
        self.llm_aux.as_ref().and_then(|aux| aux.ssm_conv_padded.clone())
    }

    // ========================================================================
    // Public API Methods (formerly trait methods)
    // ========================================================================

    /// Access arrays by their IDs.
    pub fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter().map(|id| self.array_cell(*id)).collect()
    }

    /// Access hashmaps (e.g., attention bias) by their IDs.
    pub fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        ids.iter()
            .map(|id| match id {
                HashMapId::AttentionBias => self.attention_bias.clone(),
            })
            .collect()
    }

    /// Get the active suffix length (number of tokens to process).
    pub fn active_suffix_length(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                state.active_suffix_length
            },
            ForwardPassMode::Classifier(_) => self.common_aux.suffix_length,
        }
    }

    /// Check if the current pass is a prefill pass.
    pub fn is_prefilling(&self) -> bool {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                state.is_prefilling
            },
            ForwardPassMode::Classifier(_) => true,
        }
    }

    /// Get cache layers (LLM only - returns None for classifiers).
    pub fn cache_layers(&self) -> Option<&Rc<RefCell<CacheLayers>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                Some(&state.cache_layers)
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    /// Get sampling output buffer (LLM only - returns None for classifiers).
    pub fn sampling_output(&self) -> Option<&ArrayCell> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                state.sampling_output.as_ref()
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    /// Get activation traces (available for both LLM and classifier when tracing feature enabled).
    #[cfg(feature = "tracing")]
    pub fn traces(&self) -> &Rc<RefCell<ActivationTrace>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => &state.traces,
            ForwardPassMode::Classifier(state) => &state.traces,
        }
    }

    /// Get mutable reference to sampling method (LLM only).
    pub fn sampling_method_mut(
        &mut self
    ) -> Option<&mut Option<SamplingMethod>> {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                Some(&mut state.sampling_method)
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    /// Get sampling method (LLM only).
    pub fn sampling_method(&self) -> Option<SamplingMethod> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => {
                state.sampling_method
            },
            ForwardPassMode::Classifier(_) => None,
        }
    }

    /// Copy array from source to destination.
    pub fn copy_array(
        &self,
        source_array_id: ArrayId,
        destination_array: RefCell<MetalArray>,
    ) {
        destination_array
            .borrow_mut()
            .copy_from_array(&self.arrays(&[source_array_id])[0].borrow());
    }

    /// Encode a GPU-to-GPU copy using a Blit encoder.
    pub fn encode_copy_array(
        &self,
        command_buffer: &mpsgraph::CommandBuffer,
        source_array_id: ArrayId,
        destination_array: RefCell<MetalArray>,
    ) {
        let source_ref = self.arrays(&[source_array_id])[0].clone();
        let mut src_borrow = source_ref.borrow_mut();
        let mut dst_borrow = destination_array.borrow_mut();

        let src_buf = unsafe { src_borrow.mtl_buffer().clone() };
        let dst_buf = unsafe { dst_borrow.mtl_buffer().clone() };

        let copy_size_bytes = dst_borrow.size_in_bytes() as u64;
        debug_assert_eq!(
            dst_borrow.size_in_bytes(),
            src_borrow.size_in_bytes()
        );

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let blit_encoder = mtl_command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &src_buf,
            0,
            &dst_buf,
            0,
            copy_size_bytes,
        );
        blit_encoder.end_encoding();
    }
}
