use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::trie::TrieNode,
        kernel::PoolingMeanKernel,
    },
    config::classifier::{ClassifierConfig, PoolingType},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        prediction_head::{PredictionHead, PredictionHeadError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum ClassifierError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
    #[error("Transformer error: {0}")]
    PredictionHead(#[from] PredictionHeadError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct Classifier<B: Backend> {
    hidden_dim: u32,
    data_type: DataType,
    embedding: Embedding<B>,
    embedding_norm: Normalization<B>,
    transformer: Transformer<B>,
    pooling: <B::Kernels as Kernels>::PoolingMeanKernel,
    prediction_head: PredictionHead<B>,
}

impl<B: Backend> Classifier<B> {
    pub async fn new(
        data_type: DataType,
        config: &ClassifierConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ClassifierError<B>> {
        let (embedding, _) = Embedding::new(
            config.vocab_size,
            config.transformer_config.model_dim,
            data_type,
            &config.embedding_config,
            &parameter_tree.subtree("embedding"),
            context,
        )
        .await?;

        let embedding_norm = Normalization::new(
            config.transformer_config.model_dim,
            data_type,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.embedding_norm_config,
            &parameter_tree.subtree("embedding_norm"),
            context,
        )
        .await?;

        let transformer = Transformer::new(
            data_type,
            None,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
            context,
        )
        .await?;

        if config.classifier_pooling != PoolingType::Mean {
            return Err(ClassifierError::UnsupportedConfiguration(format!(
                "config.classifier_pooling={:?} (only PoolingType::Mean is supported)",
                config.classifier_pooling
            )));
        }
        let pooling =
            <B::Kernels as Kernels>::PoolingMeanKernel::new(context, data_type).map_err(ClassifierError::Backend)?;

        let prediction_head = PredictionHead::new(
            config.hidden_dim,
            config.num_labels,
            data_type,
            &config.prediction_head_config,
            &parameter_tree.subtree("prediction_head"),
            context,
        )
        .await?;

        Ok(Self {
            hidden_dim: config.hidden_dim,
            data_type,
            embedding,
            embedding_norm,
            transformer,
            pooling,
            prediction_head,
        })
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, ClassifierError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("classifier");

        let embedded = self.embedding.encode_lookup(token_ids, batch_dim, command_buffer)?;

        let hidden = self
            .embedding_norm
            .encode(&embedded, 0, batch_dim, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(ClassifierError::Backend)?;

        let nodes = (0..batch_dim)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: batch_dim - 1,
                height: index,
            })
            .collect::<Box<[TrieNode]>>();
        let hidden = self
            .transformer
            .encode(
                hidden,
                None::<&B::ScratchBuffer>,
                &BatchTopology::new(&nodes, true),
                Some((0..batch_dim).into()),
                None,
                None,
                command_buffer,
            )?
            .output
            .unwrap();

        let mut pooled = command_buffer
            .allocate_scratch_for_shape(&[self.hidden_dim], self.data_type)
            .map_err(ClassifierError::Backend)?;
        self.pooling.encode(&hidden, &mut pooled, batch_dim, self.hidden_dim, 1, command_buffer);

        let logits = self.prediction_head.encode(pooled, 1, command_buffer)?;

        Ok(logits)
    }
}
