use ndarray::ArrayView2;

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{Allocation, Backend},
    encodable_block::DecoderArguments,
    forward_pass::{cache_layers::CacheLayers, model_shape::ModelShape, state::SharedBuffers},
};

pub struct TokenInputs<B: Backend> {
    token_ids: Array<B>,
    token_subtrie_ranges: Option<Array<B>>,
    token_positions: Array<B>,
    token_parents: Array<B>,
}

pub struct LlmDecoderPass<B: Backend> {
    token_inputs: TokenInputs<B>,
    batch_dim: usize,
    sampling_start: usize,
    sampling_length: usize,
}

impl<B: Backend> TokenInputs<B> {
    pub fn new_llm(
        context: &B::Context,
        model_shape: &ModelShape,
        token_ids: &[u64],
        token_subtrie_ranges: Option<&[[u32; 3]]>,
        token_positions: &[usize],
        token_ids_array: Option<Array<B>>,
        token_positions_array: Option<Array<B>>,
        sampling_start: usize,
        sampling_length: usize,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len(), "Tokens and positions must have same length");

        Self {
            token_ids: token_ids_array.unwrap_or_else(|| Self::init_token_ids(context, token_ids)),
            token_subtrie_ranges: token_subtrie_ranges
                .map(|ranges| Self::init_token_subtrie_ranges(context, model_shape, suffix_length, ranges)),
            token_positions: token_positions_array
                .unwrap_or_else(|| Self::init_token_positions(context, token_positions)),
            token_parents: Self::init_token_parents(context, token_positions, sampling_start, sampling_length),
        }
    }

    pub fn new_classifier(
        context: &B::Context,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len(), "Tokens and positions must have same length");

        Self {
            token_ids: Self::init_token_ids(context, token_ids),
            token_subtrie_ranges: None,
            token_positions: Self::init_token_positions(context, token_positions),
            token_parents: Self::init_token_parents(context, token_positions, 0, 0),
        }
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
        let positions_i32: Box<[i32]> = token_positions.iter().map(|position| *position as i32).collect();
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
                } else if let Some(&parent) = stack.get(depth - 1) {
                    parent as i32
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
        .expect("Invalid token subtrie range shape");
        array.copy_from_view(source);
        array
    }
}

impl<B: Backend> LlmDecoderPass<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        token_ids: &[u64],
        token_subtrie_ranges: Option<&[[u32; 3]]>,
        token_positions: &[usize],
        token_ids_array: Option<Array<B>>,
        token_positions_array: Option<Array<B>>,
        batch_dim: usize,
        sampling_start: usize,
        sampling_length: usize,
    ) -> Self {
        Self {
            token_inputs: TokenInputs::new_llm(
                context,
                model_shape,
                token_ids,
                token_subtrie_ranges,
                token_positions,
                token_ids_array,
                token_positions_array,
                sampling_start,
                sampling_length,
            ),
            batch_dim,
            sampling_start,
            sampling_length,
        }
    }

    pub fn sampling_length(&self) -> usize {
        self.sampling_length
    }

    pub fn decoder_arguments<'a>(
        &'a self,
        model_shape: &ModelShape,
        shared_buffers: &'a SharedBuffers<B>,
        cache_layers: Option<&'a mut CacheLayers<B>>,
        #[cfg(feature = "tracing")] trace: Option<&'a ActivationTrace<B>>,
    ) -> DecoderArguments<'a, B> {
        DecoderArguments {
            activation_data_type: model_shape.activation_data_type(),
            token_ids: self.token_inputs.token_ids(),
            token_positions: self.token_inputs.token_positions(),
            token_parents: self.token_inputs.token_parents(),
            token_subtrie_ranges: self.token_inputs.token_subtrie_ranges(),
            shared_buffers,
            cache_layers,
            batch_dim: self.batch_dim,
            sampling_start: self.sampling_start,
            sampling_length: self.sampling_length,
            rope_max_sequence_length: model_shape.context_length(),
            rope_dim: model_shape.rope_dim(),
            #[cfg(feature = "tracing")]
            trace,
        }
    }
}
