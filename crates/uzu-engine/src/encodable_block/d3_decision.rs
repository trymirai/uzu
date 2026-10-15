use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::trie::TrieNode,
        kernel::TensorCastKernel,
    },
    config::{model::d3_decision::D3DecisionModelConfig, weight_matrix::AnyWeightMatrixSpec},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError, TransformerState},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum D3DecisionError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingTableError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Readout error: {0}")]
    Readout(#[from] LinearBlockError<B>),
    #[error("Unsupported configuration: {0}")]
    Configuration(String),
}

pub struct D3Decision<B: Backend> {
    input_embedding: EmbeddingTable<B>,
    transformer: Transformer<B>,
    readout: Box<dyn Linear<B>>,
    cast: <B::Kernels as Kernels>::TensorCastKernel,
    model_dim: u32,
}

pub struct D3DecisionEncodeOutput<B: Backend> {
    pub logits: B::ScratchBuffer,
    pub hidden: B::ScratchBuffer,
}

#[derive(Debug, Error)]
pub enum D3DecisionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerEncodeError<B>),
    #[error("Readout error: {0}")]
    Readout(#[from] LinearEncodeError<B>),
}

impl<B: Backend> D3Decision<B> {
    pub fn new(
        context: &B::Context,
        config: &D3DecisionModelConfig,
        tree: &ParameterTree<B>,
    ) -> Result<Self, D3DecisionError<B>> {
        config.validate().map_err(D3DecisionError::Configuration)?;
        let model_dim = config.transformer_config.model_dim;
        let input_embedding = EmbeddingTable::load(
            config.vocab_size,
            model_dim,
            DataType::BF16,
            &tree.subtree("input_embedding"),
            context,
        )?;
        let transformer =
            Transformer::new(DataType::BF16, None, &config.transformer_config, &tree.subtree("transformer"), context)?;
        let readout_tree = tree.subtree("readout");
        if !matches!(
            readout_tree.subtree("weights").metadata::<AnyWeightMatrixSpec>("spec")?,
            AnyWeightMatrixSpec::FullPrecisionSpec(_)
        ) {
            return Err(D3DecisionError::Configuration("D3 readout must remain full precision".into()));
        }
        let readout = <dyn Linear<B>>::new_mixed_precision(
            model_dim,
            config.decision.codes.len() as u32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            false,
            &readout_tree,
            context,
        )?;
        let cast = <B::Kernels as Kernels>::TensorCastKernel::new(context, DataType::BF16, DataType::F32)
            .map_err(D3DecisionError::Backend)?;
        Ok(Self {
            input_embedding,
            transformer,
            readout,
            cast,
            model_dim,
        })
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub fn create_empty_state(
        &self,
        sequence_length: u32,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let mut state = self.transformer.create_empty_state_for_prefill(sequence_length, context)?;
        state.prepare(0, sequence_length)?;
        Ok(state)
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        sequence_length: u32,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<D3DecisionEncodeOutput<B>, D3DecisionEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("D3 decision");
        let mut embedded = command_buffer
            .allocate_scratch_for_shape(&[sequence_length, self.model_dim], DataType::BF16)
            .map_err(D3DecisionEncodeError::Backend)?;
        self.input_embedding.encode_lookup(token_ids, &mut embedded, sequence_length, 1.0, command_buffer);
        let nodes = (0..sequence_length)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: sequence_length - 1,
                height: index,
            })
            .collect::<Box<[_]>>();
        // A single complete prompt is required: future keys must be visible to
        // each noncausal attention layer, even when the prompt exceeds 1024 tokens.
        let hidden_bf16 = self
            .transformer
            .encode(
                embedded,
                None::<&B::ScratchBuffer>,
                &BatchTopology::new(&nodes, true),
                Some((sequence_length - 1..sequence_length).into()),
                None,
                Some(state),
                command_buffer,
            )?
            .output
            .expect("D3 always requests a final normalized hidden state");
        let mut hidden = command_buffer
            .allocate_scratch_for_shape(&[self.model_dim], DataType::F32)
            .map_err(D3DecisionEncodeError::Backend)?;
        self.cast.encode(&hidden_bf16, &mut hidden, self.model_dim, command_buffer);
        let mut readout_input =
            command_buffer.allocate_scratch(hidden.size()).map_err(D3DecisionEncodeError::Backend)?;
        command_buffer.encode_copy(&hidden, &mut readout_input);
        let logits = self.readout.encode(readout_input, 1, command_buffer)?;
        Ok(D3DecisionEncodeOutput {
            logits,
            hidden,
        })
    }
}
