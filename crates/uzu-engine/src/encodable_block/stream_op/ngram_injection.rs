use std::{io, mem};

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        kernel::{NGramConvInjectKernel, NGramConvInjectTrieKernel, NGramGateKernel},
    },
    config::{
        normalization::NormalizationConfig,
        residual::gated::GatedResidualConfig,
        stream_op::ngram_injection::{NGramHashConfig, NGramInjectionConfig},
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding_table_disk::EmbeddingTableDisk,
        linear::Linear,
        stream_op::{StreamOpEncodeError, StreamOpNewError},
    },
    safetensors::ParameterTree,
    utils::size_for_shape::size_for_shape,
};

// TODO(arch): this is essentially ctrl+c ctrl+v `ShortConvStateSuffixStatus`.
// We should probably merge both.
enum NGramSuffixStatus<B: Backend> {
    Flat {
        suffix_length: u32,
    },
    Trie {
        conv_states: B::ScratchBuffer,
    },
}

pub struct NGramInjectionState<B: Backend> {
    /// Up to the last `ngram_size - 1` accepted token ids, most recent last.
    history: Vec<u32>,
    history_length: usize,
    /// Token ids of the batch whose rows were prepared and not yet accepted.
    pending_tokens: Vec<u32>,
    /// Embedding rows of the pending batch, `[tokens, embedding_dim]` in the model data type, until encoded.
    pending_rows: Vec<u8>,
    conv_state: B::GlobalGpuBuffer,
    pending_conv_state: B::GlobalGpuBuffer,
    suffix_status: Option<NGramSuffixStatus<B>>,
    is_empty: bool,
}

impl<B: Backend> NGramInjectionState<B> {
    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let pending_tokens = mem::take(&mut self.pending_tokens);
        self.pending_rows.clear();
        if !pending_tokens.is_empty() {
            self.history.extend(accepted_indices.iter().map(|&index| pending_tokens[index as usize]));
            let excess = self.history.len().saturating_sub(self.history_length);
            self.history.drain(..excess);
        }

        let Some(suffix_status) = self.suffix_status.take() else {
            return Ok(());
        };
        let accepted_index = *accepted_indices.last().expect("n-gram injection state attempted to accept zero indices");

        match suffix_status {
            NGramSuffixStatus::Flat {
                suffix_length,
            } => {
                assert!(accepted_index == suffix_length - 1, "attempted a partial flat n-gram injection accept");
                mem::swap(&mut self.conv_state, &mut self.pending_conv_state);
            },
            NGramSuffixStatus::Trie {
                conv_states,
            } => {
                let slot_size = self.conv_state.size();
                let slot_offset = accepted_index as usize * slot_size;
                command_buffer
                    .encode_copy(conv_states.subrange(slot_offset..slot_offset + slot_size), &mut self.conv_state);
            },
        }
        self.is_empty = false;
        Ok(())
    }
}

pub struct NGramInjection<B: Backend> {
    ngram_embedding: EmbeddingTableDisk,
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
    hash_config: NGramHashConfig,
    gate_kernel: <B::Kernels as Kernels>::NGramGateKernel,
    conv_kernel: <B::Kernels as Kernels>::NGramConvInjectKernel,
    conv_trie_kernel: <B::Kernels as Kernels>::NGramConvInjectTrieKernel,
    model_dim: u32,
    stream_count: u32,
    embedding_dim: u32,
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

        let num_rows = hash_config.head_vocab_sizes.iter().sum::<u32>();

        let ngram_embedding = EmbeddingTableDisk::load(
            num_rows,
            config.embedding_dim / num_heads,
            data_type,
            &parameter_tree.subtree("ngram_embedding"),
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

        let gate_kernel =
            <B::Kernels as Kernels>::NGramGateKernel::new(context, data_type).map_err(StreamOpNewError::Backend)?;
        let conv_kernel = <B::Kernels as Kernels>::NGramConvInjectKernel::new(context, data_type)
            .map_err(StreamOpNewError::Backend)?;
        let conv_trie_kernel = <B::Kernels as Kernels>::NGramConvInjectTrieKernel::new(context, data_type)
            .map_err(StreamOpNewError::Backend)?;

        Ok(Self {
            ngram_embedding,
            key_projection,
            value_projection,
            key_scales,
            query_scales,
            conv_scales,
            conv_weights,
            hash_config: hash_config.clone(),
            gate_kernel,
            conv_kernel,
            conv_trie_kernel,
            model_dim,
            stream_count,
            embedding_dim: config.embedding_dim,
            conv_kernel_size: config.conv_kernel_size,
            norm_epsilon: epsilon,
            norm_scale_offset: scale_offset.unwrap_or(0.0),
            data_type,
        })
    }

    fn num_heads(&self) -> u32 {
        num_heads(&self.hash_config)
    }

    fn channels(&self) -> u32 {
        self.stream_count * self.model_dim
    }

    pub fn create_empty_state(
        &self,
        context: &B::Context,
    ) -> Result<NGramInjectionState<B>, B::Error> {
        let conv_state_size = size_for_shape(&[self.conv_kernel_size - 1, self.channels()], self.data_type);
        Ok(NGramInjectionState {
            history: Vec::new(),
            history_length: self.hash_config.ngram_size as usize - 1,
            pending_tokens: Vec::new(),
            pending_rows: Vec::new(),
            conv_state: context.create_buffer_gpu(conv_state_size)?,
            pending_conv_state: context.create_buffer_gpu(conv_state_size)?,
            suffix_status: None,
            is_empty: true,
        })
    }

    /// Needs to be called before [`Self::encode`]
    // TODO(arch): use typestate?
    pub fn prepare(
        &self,
        state: &mut NGramInjectionState<B>,
        token_ids: &[u32],
        parents: Option<&[i32]>,
    ) -> io::Result<()> {
        assert!(state.pending_tokens.is_empty(), "n-gram rows prepared with unaccepted tokens");

        let mut ngram_ids = vec![0u32; token_ids.len() * self.num_heads() as usize];
        hash(&self.hash_config, &state.history, token_ids, parents, &mut ngram_ids);
        state.pending_tokens = token_ids.to_vec();

        self.ngram_embedding.lookup(&ngram_ids, &mut state.pending_rows)
    }

    // TODO(arch): use typestate with `Self::prepare`?
    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        state: &mut NGramInjectionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, StreamOpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("n-gram injection");

        assert!(state.suffix_status.is_none(), "n-gram injection called with state with unaccepted tokens");

        let num_tokens = batch_dim.size();
        if state.pending_tokens.is_empty() {
            panic!("Should have prepared ngram rows earlier");
        }
        let prepared = state.pending_tokens.len();
        if batch_dim.full_accept() {
            assert_eq!(prepared, num_tokens as usize, "n-gram rows were prepared for another batch");
        } else {
            assert!(prepared <= num_tokens as usize, "n-gram rows were prepared for more nodes than the trie has");
        }
        let mut rows = mem::take(&mut state.pending_rows);
        let row_size = size_for_shape(&[1, self.embedding_dim], self.data_type);
        assert_eq!(rows.len(), prepared * row_size, "wrong rows size");
        // pad back trie
        rows.resize(num_tokens as usize * row_size, 0);

        if state.is_empty {
            command_buffer.encode_fill(&mut state.conv_state, 0);
        }

        let rows = command_buffer.allocate_constant_from_slice(&rows).map_err(StreamOpEncodeError::Backend)?;
        let mut embeddings = command_buffer
            .allocate_scratch_for_shape(&[num_tokens, self.embedding_dim], self.data_type)
            .map_err(StreamOpEncodeError::Backend)?;
        command_buffer.encode_copy(&rows, &mut embeddings);

        // TODO: this copy could be avoided if `Linear::encode` borrowed its
        // input (or if the key and value projections were fused into one linear
        // whose output is split afterwards).
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

        state.suffix_status = Some(if batch_dim.full_accept() {
            self.conv_kernel.encode(
                &conv_input,
                &state.conv_state,
                &self.conv_weights,
                &gated,
                &mut stream,
                &mut state.pending_conv_state,
                num_tokens,
                self.channels(),
                self.conv_kernel_size,
                command_buffer,
            );
            NGramSuffixStatus::Flat {
                suffix_length: num_tokens,
            }
        } else {
            let nodes = batch_dim.nodes().expect("trie n-gram injection requires trie nodes");
            let mut conv_states = command_buffer
                .allocate_scratch_for_shape(&[num_tokens, self.conv_kernel_size - 1, self.channels()], self.data_type)
                .map_err(StreamOpEncodeError::Backend)?;
            self.conv_trie_kernel.encode(
                &conv_input,
                &state.conv_state,
                nodes,
                &self.conv_weights,
                &gated,
                &mut stream,
                &mut conv_states,
                num_tokens,
                self.channels(),
                self.conv_kernel_size,
                command_buffer,
            );
            NGramSuffixStatus::Trie {
                conv_states,
            }
        });
        Ok(stream)
    }
}

fn num_heads(config: &NGramHashConfig) -> u32 {
    (config.ngram_size - 1) * config.heads_per_ngram
}

fn hash(
    config: &NGramHashConfig,
    history: &[u32],
    tokens: &[u32],
    parents: Option<&[i32]>,
    ngram_ids: &mut [u32],
) {
    let num_heads = num_heads(config) as usize;
    let eos_token_id = config.eos_token_id as u64;
    assert_eq!(ngram_ids.len(), tokens.len() * num_heads, "n-gram ids must have one row per token and head");
    if let Some(parents) = parents {
        assert_eq!(parents.len(), tokens.len(), "n-gram parents must have one entry per token");
    }

    let parent = |index: usize| -> Option<usize> {
        let parent = parents.map(|parents| parents[index]).unwrap_or(index as i32 - 1);
        usize::try_from(parent).ok()
    };

    enum Cursor {
        Batch(usize),
        History(usize),
    }

    // Tokens preceding the current one, nearest first.
    let mut context = vec![0u32; config.ngram_size as usize - 1];
    for (token_index, &token) in tokens.iter().enumerate() {
        let mut cursor = Cursor::Batch(token_index);
        for slot in &mut context {
            cursor = match cursor {
                Cursor::Batch(index) => parent(index).map(Cursor::Batch).unwrap_or(Cursor::History(1)),
                Cursor::History(steps_back) => Cursor::History(steps_back + 1),
            };
            *slot = match cursor {
                Cursor::Batch(index) => tokens[index],
                Cursor::History(steps_back) => {
                    history.len().checked_sub(steps_back).map(|index| history[index]).unwrap_or(config.eos_token_id)
                },
            };
        }

        let mut head_offset = 0u64;
        for head in 0..num_heads {
            let order = head / config.heads_per_ngram as usize + 2;
            let mut hash = (token as u64).wrapping_mul(config.multipliers[0]);
            // Everything before history start is EOS
            let mut crossed_eos = false;
            for shift in 1..order {
                let shifted_token = if crossed_eos {
                    eos_token_id
                } else {
                    context[shift - 1] as u64
                };
                crossed_eos |= shifted_token == eos_token_id;
                hash ^= shifted_token.wrapping_mul(config.multipliers[shift]);
            }
            let head_vocab_size = config.head_vocab_sizes[head] as u64;
            ngram_ids[token_index * num_heads + head] = (hash % head_vocab_size + head_offset) as u32;
            head_offset += head_vocab_size;
        }
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/stream_op/ngram_injection.rs"]
mod tests;
