use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        kernel::{NGramConvInjectKernel, NGramGateKernel, NGramHashKernel},
    },
    config::{
        normalization::NormalizationConfig, residual::gated::GatedResidualConfig,
        stream_op::ngram_injection::NGramInjectionConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding_table::EmbeddingTable,
        linear::Linear,
        stream_op::{StreamOpEncodeError, StreamOpNewError},
    },
    safetensors::ParameterTree,
    utils::size_for_shape::size_for_shape,
};

pub struct NGramInjectionState<B: Backend> {
    /// last `ngram_size - 1` token ids
    token_history: B::GlobalGpuBuffer,
    conv_state: B::GlobalGpuBuffer,
    pending_suffix_length: Option<u32>,
    is_empty: bool,
}

impl<B: Backend> NGramInjectionState<B> {
    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
    ) {
        let Some(suffix_length) = self.pending_suffix_length.take() else {
            return;
        };
        let accepted_index = *accepted_indices.last().expect("n-gram injection state attempted to accept zero indices");
        assert!(accepted_index == suffix_length - 1, "attempted a partial n-gram injection accept");
    }
}

pub struct NGramInjection<B: Backend> {
    ngram_embedding: EmbeddingTable<B>,
    /// `embedding_dim -> stream_count * model_dim`
    key_projection: Box<dyn Linear<B>>,
    /// `embedding_dim -> model_dim`
    value_projection: Box<dyn Linear<B>>,
    /// `[stream_count, model_dim]`, `f32`
    key_scales: B::GlobalGpuBuffer,
    /// `[stream_count, model_dim]`, `f32`
    query_scales: B::GlobalGpuBuffer,
    /// `[stream_count, model_dim]`, `f32`
    conv_scales: B::GlobalGpuBuffer,
    /// `[stream_count * model_dim, conv_kernel_size]`, `f32`
    conv_weights: B::GlobalGpuBuffer,
    multipliers: Box<[u64]>,
    head_vocab_sizes: Box<[u32]>,
    head_offsets: Box<[u32]>,
    hash_kernel: <B::Kernels as Kernels>::NGramHashKernel,
    gate_kernel: <B::Kernels as Kernels>::NGramGateKernel,
    conv_kernel: <B::Kernels as Kernels>::NGramConvInjectKernel,
    model_dim: u32,
    stream_count: u32,
    embedding_dim: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    eos_token_id: u32,
    conv_kernel_size: u32,
    norm_epsilon: f32,
    norm_scale_offset: f32,
    data_type: DataType,
}

impl<B: Backend> NGramInjection<B> {
    pub async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &NGramInjectionConfig,
        residual_config: &GatedResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, StreamOpNewError<B>> {
        let hash_config = &config.hash_config;
        let stream_count = residual_config.stream_count;
        let num_heads = (hash_config.ngram_size - 1) * hash_config.heads_per_ngram;
        let unsupported = |reason: String| Err(StreamOpNewError::UnsupportedConfiguration(reason));

        if hash_config.ngram_size < 2 || hash_config.multipliers.len() != hash_config.ngram_size as usize {
            return unsupported(format!(
                "ngram_size {} with {} multipliers",
                hash_config.ngram_size,
                hash_config.multipliers.len()
            ));
        }
        if hash_config.head_vocab_sizes.len() != num_heads as usize {
            return unsupported(format!(
                "expected {num_heads} head vocabulary sizes, got {}",
                hash_config.head_vocab_sizes.len()
            ));
        }
        if !config.embedding_dim.is_multiple_of(num_heads) {
            return unsupported(format!(
                "embedding_dim {} is not divisible by {num_heads} heads",
                config.embedding_dim
            ));
        }
        if config.conv_kernel_size < 2 || config.conv_config.has_biases {
            return unsupported(format!(
                "conv with kernel size {} and has_biases = {}",
                config.conv_kernel_size, config.conv_config.has_biases
            ));
        }
        let NormalizationConfig {
            epsilon,
            scale_offset,
            subtract_mean: false,
            has_scale: true,
            has_biases: false,
            ..
        } = config.norm_config
        else {
            return unsupported(format!("normalization {:?}", config.norm_config));
        };

        let head_offsets = hash_config
            .head_vocab_sizes
            .iter()
            .scan(0u32, |offset, &size| {
                let head_offset = *offset;
                *offset += size;
                Some(head_offset)
            })
            .collect::<Box<[u32]>>();
        let num_rows = hash_config.head_vocab_sizes.iter().sum::<u32>();

        let ngram_embedding = EmbeddingTable::load(
            num_rows,
            config.embedding_dim / num_heads,
            data_type,
            &parameter_tree.subtree("ngram_embedding"),
            context,
        )
        .await?;
        let key_projection = <dyn Linear<B>>::new(
            config.embedding_dim,
            stream_count * model_dim,
            data_type,
            false,
            &parameter_tree.subtree("key_projection"),
            context,
        )
        .await?;
        let value_projection = <dyn Linear<B>>::new(
            config.embedding_dim,
            model_dim,
            data_type,
            false,
            &parameter_tree.subtree("value_projection"),
            context,
        )
        .await?;

        let read_stream_scales = async |name: &str| {
            parameter_tree.subtree(name).read_buffer("scales", &[stream_count, model_dim], DataType::F32).await
        };
        let key_scales = read_stream_scales("key_norm").await?;
        let query_scales = read_stream_scales("query_norm").await?;
        let conv_scales = read_stream_scales("conv_norm").await?;
        let conv_weights = parameter_tree
            .subtree("conv")
            .read_buffer("weights", &[stream_count * model_dim, config.conv_kernel_size], DataType::F32)
            .await?;

        let hash_kernel = <B::Kernels as Kernels>::NGramHashKernel::new(context).map_err(StreamOpNewError::Backend)?;
        let gate_kernel =
            <B::Kernels as Kernels>::NGramGateKernel::new(context, data_type).map_err(StreamOpNewError::Backend)?;
        let conv_kernel = <B::Kernels as Kernels>::NGramConvInjectKernel::new(context, data_type)
            .map_err(StreamOpNewError::Backend)?;

        Ok(Self {
            ngram_embedding,
            key_projection,
            value_projection,
            key_scales,
            query_scales,
            conv_scales,
            conv_weights,
            multipliers: hash_config.multipliers.clone(),
            head_vocab_sizes: hash_config.head_vocab_sizes.clone(),
            head_offsets,
            hash_kernel,
            gate_kernel,
            conv_kernel,
            model_dim,
            stream_count,
            embedding_dim: config.embedding_dim,
            ngram_size: hash_config.ngram_size,
            heads_per_ngram: hash_config.heads_per_ngram,
            eos_token_id: hash_config.eos_token_id,
            conv_kernel_size: config.conv_kernel_size,
            norm_epsilon: epsilon,
            norm_scale_offset: scale_offset.unwrap_or(0.0),
            data_type,
        })
    }

    fn num_heads(&self) -> u32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    fn channels(&self) -> u32 {
        self.stream_count * self.model_dim
    }

    pub fn create_empty_state(
        &self,
        context: &B::Context,
    ) -> Result<NGramInjectionState<B>, B::Error> {
        let token_history = context.create_buffer_gpu(size_for_shape(&[self.ngram_size - 1], DataType::U32))?;
        let conv_state =
            context.create_buffer_gpu(size_for_shape(&[self.conv_kernel_size - 1, self.channels()], self.data_type))?;
        Ok(NGramInjectionState {
            token_history,
            conv_state,
            pending_suffix_length: None,
            is_empty: true,
        })
    }

    /// Takes and returns a gated stream, `[tokens, stream_count, model_dim]`.
    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology<B>,
        state: &mut NGramInjectionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, StreamOpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("n-gram injection");

        if !batch_dim.full_accept() {
            return Err(StreamOpEncodeError::SpeculationNotSupported);
        }
        assert!(state.pending_suffix_length.is_none(), "n-gram injection called with state with unaccepted tokens");

        let num_tokens = batch_dim.size();
        let history_length = self.ngram_size - 1;
        let token_bytes = size_for_shape(&[1], DataType::U32);
        let history_bytes = history_length as usize * token_bytes;
        let tokens_bytes = num_tokens as usize * token_bytes;

        if state.is_empty {
            let eos_history = command_buffer
                .allocate_constant_from_slice(&vec![self.eos_token_id; history_length as usize])
                .map_err(StreamOpEncodeError::Backend)?;
            command_buffer.encode_copy(&eos_history, &mut state.token_history);
            command_buffer.encode_fill(&mut state.conv_state, 0);
            state.is_empty = false;
        }

        let mut tokens_with_history = command_buffer
            .allocate_scratch_for_shape(&[history_length + num_tokens], DataType::U32)
            .map_err(StreamOpEncodeError::Backend)?;
        command_buffer.encode_copy(&state.token_history, (&mut tokens_with_history).subrange_mut(..history_bytes));
        command_buffer
            .encode_copy(token_ids.subrange(..tokens_bytes), (&mut tokens_with_history).subrange_mut(history_bytes..));
        command_buffer.encode_copy(
            (&tokens_with_history).subrange(tokens_bytes..tokens_bytes + history_bytes),
            &mut state.token_history,
        );

        let num_heads = self.num_heads();
        let mut ngram_ids = command_buffer
            .allocate_scratch_for_shape(&[num_tokens, num_heads], DataType::U32)
            .map_err(StreamOpEncodeError::Backend)?;
        self.hash_kernel.encode(
            &tokens_with_history,
            &self.multipliers,
            &self.head_vocab_sizes,
            &self.head_offsets,
            &mut ngram_ids,
            num_tokens,
            self.ngram_size,
            self.heads_per_ngram,
            self.eos_token_id,
            command_buffer,
        );

        let mut embeddings = command_buffer
            .allocate_scratch_for_shape(&[num_tokens, self.embedding_dim], self.data_type)
            .map_err(StreamOpEncodeError::Backend)?;
        self.ngram_embedding.encode_lookup(&ngram_ids, &mut embeddings, num_tokens * num_heads, 1.0, command_buffer);

        let mut key_input = command_buffer.allocate_scratch(embeddings.size()).map_err(StreamOpEncodeError::Backend)?;
        command_buffer.encode_copy(&embeddings, &mut key_input);
        let keys = self.key_projection.encode(key_input, num_tokens, command_buffer)?;
        let values = self.value_projection.encode(embeddings, num_tokens, command_buffer)?;

        let stream_shape = [num_tokens, self.stream_count, self.model_dim];
        let mut gated = command_buffer
            .allocate_scratch_for_shape(&stream_shape, self.data_type)
            .map_err(StreamOpEncodeError::Backend)?;
        let mut conv_input = command_buffer
            .allocate_scratch_for_shape(&stream_shape, self.data_type)
            .map_err(StreamOpEncodeError::Backend)?;
        self.gate_kernel.encode(
            &keys,
            &stream,
            &values,
            &self.key_scales,
            &self.query_scales,
            &self.conv_scales,
            &mut gated,
            &mut conv_input,
            num_tokens,
            self.stream_count,
            self.model_dim,
            self.norm_epsilon,
            self.norm_scale_offset,
            command_buffer,
        );

        let mut next_conv_state =
            command_buffer.allocate_scratch(state.conv_state.size()).map_err(StreamOpEncodeError::Backend)?;
        self.conv_kernel.encode(
            &conv_input,
            &state.conv_state,
            &self.conv_weights,
            &gated,
            &mut stream,
            &mut next_conv_state,
            num_tokens,
            self.channels(),
            self.conv_kernel_size,
            command_buffer,
        );
        command_buffer.encode_copy(&next_conv_state, &mut state.conv_state);

        state.pending_suffix_length = Some(num_tokens);
        Ok(stream)
    }
}
