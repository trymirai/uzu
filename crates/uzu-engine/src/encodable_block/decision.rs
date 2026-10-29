use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding, gpu_types::trie::TrieNode},
    config::model::decision_model::DecisionModelConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{Linear, LinearEncodeError, LinearMatmul, LinearMatmulError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum DecisionError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding table error: {0}")]
    EmbeddingTable(#[from] EmbeddingTableError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Linear matmul error: {0}")]
    LinearMatmul(#[from] LinearMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
}

pub struct Decision<B: Backend> {
    input_data_type: DataType,
    input_scale: f32,
    input_embedding: EmbeddingTable<B>,
    transformer: Transformer<B>,
    readout: LinearMatmul<B>,
}

impl<B: Backend> Decision<B> {
    pub async fn new(
        input_data_type: DataType,
        output_data_type: DataType,
        config: &DecisionModelConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, DecisionError<B>> {
        let model_dim = config.decoder_config.transformer_config.model_dim;
        let embedding_tree = parameter_tree.subtree("embedding");
        let input_embedding = EmbeddingTable::load(
            config.decoder_config.vocab_size,
            model_dim,
            input_data_type,
            &embedding_tree.subtree("input_embedding"),
            context,
        )
        .await?;

        let transformer = Transformer::new(
            input_data_type,
            output_data_type,
            None,
            &config.decoder_config.transformer_config,
            &parameter_tree.subtree("transformer"),
            context,
        )
        .await?;

        let readout_tree = embedding_tree.subtree("output_embedding");
        let readout_spec = readout_tree.metadata("spec")?;
        let readout = LinearMatmul::load(
            model_dim,
            config.decision.codes.len() as u32,
            output_data_type,
            output_data_type,
            output_data_type,
            readout_spec,
            None,
            &readout_tree,
            None,
            context,
        )
        .await?;

        Ok(Self {
            input_data_type,
            input_scale: config.decoder_config.embedding_config.input_scale().unwrap_or(1.0),
            input_embedding,
            transformer,
            readout,
        })
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    /// Runs the whole prompt in one stateless pass and returns the `[1, answer_count]` logits of its last token.
    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        token_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, DecisionError<B>> {
        assert!(token_count > 0, "decision needs at least one token");

        let command_buffer = &mut *command_buffer.debug_group("decision");

        let mut embedded = command_buffer
            .allocate_scratch_for_shape(&[token_count, self.transformer.model_dim()], self.input_data_type)
            .map_err(DecisionError::Backend)?;
        self.input_embedding.encode_lookup(token_ids, &mut embedded, token_count, self.input_scale, command_buffer);

        let nodes = (0..token_count)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: token_count - 1,
                height: index,
            })
            .collect::<Box<[TrieNode]>>();

        let hidden = self
            .transformer
            .encode(
                embedded,
                None::<&B::ScratchBuffer>,
                &BatchTopology::new(&nodes, true),
                Some((token_count - 1..token_count).into()),
                None,
                None,
                command_buffer,
            )?
            .output
            .expect("transformer returns rows for an output range");

        let logits = self.readout.encode(hidden, 1, command_buffer)?;

        Ok(logits)
    }
}
