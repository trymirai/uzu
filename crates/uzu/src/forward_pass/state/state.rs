use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayView2;

#[cfg(feature = "tracing")]
use crate::forward_pass::cache_layers::CacheLayer;
use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    backends::common::{Allocation, Backend},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        state::{RopeType, SharedBuffers},
    },
    session::parameter::SamplingMethod,
};

pub enum ForwardPassMode<B: Backend> {
    LanguageModelGenerator(LanguageModelGeneratorModeState<B>),
    Classifier {
        active_row_count: usize,
    },
}

pub struct LanguageModelGeneratorModeState<B: Backend> {
    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,
    pub sampling_method: Option<SamplingMethod>,
    pub active_row_count: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub is_prefilling: bool,
}

pub struct ForwardPassState<B: Backend> {
    context: Rc<B::Context>,
    model_shape: ModelShape,
    token_ids: Array<B>,
    pub token_subtrie_ranges: Option<Array<B>>,
    token_positions: Array<B>,
    token_parents: Array<B>,
    pub shared_buffers: Rc<SharedBuffers<B>>,
    mode: ForwardPassMode<B>,
}

impl<B: Backend> ForwardPassState<B> {
    fn init_token_ids(
        context: &B::Context,
        token_ids: &[u64],
    ) -> Array<B> {
        context.create_array_from(&[token_ids.len()], token_ids, "forward_pass_token_ids")
    }

    fn init_token_positions(
        context: &B::Context,
        token_positions: &[usize],
    ) -> Array<B> {
        let positions_i32: Box<[i32]> = token_positions.iter().map(|p| *p as i32).collect();
        context.create_array_from(&[token_positions.len()], positions_i32.as_ref(), "forward_pass_token_positions")
    }

    fn init_token_parents(
        context: &B::Context,
        token_positions: &[usize],
        sampling_start: usize,
        sampling_length: usize,
    ) -> Array<B> {
        let suffix_length = token_positions.len();
        let mut parents = vec![-1i32; suffix_length];

        if sampling_length > 0 {
            let root_pos = token_positions.get(sampling_start).copied().unwrap_or(0);
            let mut stack: Vec<usize> = Vec::new();

            for local_idx in 0..sampling_length {
                let abs_idx = sampling_start + local_idx;
                let Some(&pos) = token_positions.get(abs_idx) else {
                    break;
                };
                let depth = pos.saturating_sub(root_pos);
                let parent_local = if depth == 0 {
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

        context.create_array_from(&[suffix_length], &parents, "forward_pass_token_parents")
    }

    fn init_token_subtrie_ranges(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
        token_subtrie_ranges: &[[u32; 3]],
    ) -> Array<B> {
        let shape = model_shape.subtrie_ranges_shape(suffix_length);
        let mut array = context.create_array_zeros(&shape, DataType::U32, "forward_pass_token_subtrie_ranges");
        let source = ArrayView2::from_shape(
            (token_subtrie_ranges.len(), 3),
            bytemuck::cast_slice::<_, u32>(token_subtrie_ranges),
        )
        .unwrap();
        array.copy_from_view(source);
        array
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_llm(
        context: Rc<B::Context>,
        model_shape: &ModelShape,
        cache_layers: Rc<RefCell<CacheLayers<B>>>,
        shared_buffers: Rc<SharedBuffers<B>>,
        token_ids: &[u64],
        token_subtrie_ranges: Option<&[[u32; 3]]>,
        token_positions: &[usize],
        token_ids_allocation: Option<Array<B>>,
        token_positions_allocation: Option<Array<B>>,
        active_row_count: usize,
        sampling_start: usize,
        sampling_length: usize,
        is_prefilling: bool,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len(), "Tokens and positions must have same length");
        assert!(suffix_length <= cache_layers.borrow().max_suffix_length(), "Suffix length exceeds KV cache capacity");

        let token_ids_allocation =
            token_ids_allocation.unwrap_or_else(|| Self::init_token_ids(context.as_ref(), token_ids));
        let token_subtrie_ranges_allocation = token_subtrie_ranges
            .map(|ranges| Self::init_token_subtrie_ranges(context.as_ref(), model_shape, suffix_length, ranges));
        let token_positions_allocation =
            token_positions_allocation.unwrap_or_else(|| Self::init_token_positions(context.as_ref(), token_positions));
        let token_parents_allocation =
            Self::init_token_parents(context.as_ref(), token_positions, sampling_start, sampling_length);

        let mode = ForwardPassMode::LanguageModelGenerator(LanguageModelGeneratorModeState {
            cache_layers,
            sampling_method: None,
            active_row_count,
            sampling_start,
            sampling_length,
            is_prefilling,
        });

        Self {
            context,
            model_shape: model_shape.clone(),
            token_ids: token_ids_allocation,
            token_subtrie_ranges: token_subtrie_ranges_allocation,
            token_positions: token_positions_allocation,
            token_parents: token_parents_allocation,
            shared_buffers,
            mode,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_classifier(
        context: Rc<B::Context>,
        model_shape: &ModelShape,
        shared_buffers: Rc<SharedBuffers<B>>,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let token_ids_allocation = Self::init_token_ids(context.as_ref(), token_ids);
        let token_positions_allocation = Self::init_token_positions(context.as_ref(), token_positions);
        let token_parents_allocation = Self::init_token_parents(context.as_ref(), token_positions, 0, 0);

        Self {
            context,
            model_shape: model_shape.clone(),
            token_ids: token_ids_allocation,
            token_subtrie_ranges: None,
            token_positions: token_positions_allocation,
            token_parents: token_parents_allocation,
            shared_buffers,
            mode: ForwardPassMode::Classifier {
                active_row_count: suffix_length,
            },
        }
    }

    pub fn context(&self) -> &B::Context {
        self.context.as_ref()
    }

    pub fn model_shape(&self) -> &ModelShape {
        &self.model_shape
    }

    pub fn token_ids(&self) -> &Allocation<B> {
        self.token_ids.allocation()
    }

    pub fn token_positions(&self) -> &Allocation<B> {
        self.token_positions.allocation()
    }

    pub fn token_parents(&self) -> &Allocation<B> {
        self.token_parents.allocation()
    }

    pub fn token_subtrie_ranges(&self) -> Option<&Allocation<B>> {
        self.token_subtrie_ranges.as_ref().map(Array::allocation)
    }

    pub fn rope_cosines(
        &self,
        rope_type: RopeType,
    ) -> &Allocation<B> {
        match rope_type {
            RopeType::Global => &self.shared_buffers.global_rope.as_ref().expect("Global rope not initialized").cosines,
            RopeType::Local => &self.shared_buffers.local_rope.as_ref().expect("Local rope not initialized").cosines,
        }
    }

    pub fn rope_sines(
        &self,
        rope_type: RopeType,
    ) -> &Allocation<B> {
        match rope_type {
            RopeType::Global => &self.shared_buffers.global_rope.as_ref().expect("Global rope not initialized").sines,
            RopeType::Local => &self.shared_buffers.local_rope.as_ref().expect("Local rope not initialized").sines,
        }
    }

    pub fn rope_max_sequence_length(&self) -> usize {
        self.model_shape.context_length()
    }

    pub fn rope_dim(&self) -> usize {
        self.model_shape.rope_dim()
    }

    pub fn attention_sinks(
        &self,
        layer_index: usize,
    ) -> Option<&Allocation<B>> {
        self.shared_buffers.attention_sinks.as_ref().map(|sinks| &sinks[layer_index])
    }

    pub fn vocab_size(&self) -> usize {
        self.model_shape.logits_shape(self.token_count())[1]
    }

    pub fn active_row_count(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.active_row_count,
            ForwardPassMode::Classifier {
                active_row_count,
            } => *active_row_count,
        }
    }

    pub fn sampling_start(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_start,
            ForwardPassMode::Classifier {
                ..
            } => 0,
        }
    }

    pub fn sampling_length(&self) -> usize {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_length,
            ForwardPassMode::Classifier {
                ..
            } => self.token_count(),
        }
    }

    pub fn is_prefilling(&self) -> bool {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.is_prefilling,
            ForwardPassMode::Classifier {
                ..
            } => true,
        }
    }

    pub fn cache_layers(&self) -> Option<&Rc<RefCell<CacheLayers<B>>>> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => Some(&state.cache_layers),
            ForwardPassMode::Classifier {
                ..
            } => None,
        }
    }

    #[cfg(feature = "tracing")]
    pub fn with_cache_layer<R>(
        &self,
        layer_index: usize,
        f: impl FnOnce(&CacheLayer<B>) -> R,
    ) -> R {
        let cache_layers = self.cache_layers().expect("Cache layers are only available in LLM mode");
        let cache = cache_layers.borrow();
        f(&cache.data[layer_index])
    }

    pub fn sampling_method_mut(&mut self) -> Option<&mut Option<SamplingMethod>> {
        match &mut self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => Some(&mut state.sampling_method),
            ForwardPassMode::Classifier {
                ..
            } => None,
        }
    }

    pub fn sampling_method(&self) -> Option<SamplingMethod> {
        match &self.mode {
            ForwardPassMode::LanguageModelGenerator(state) => state.sampling_method,
            ForwardPassMode::Classifier {
                ..
            } => None,
        }
    }

    fn token_count(&self) -> usize {
        self.token_ids.as_buffer_range().1.len() / size_for_shape(&[1], DataType::U64)
    }
}
