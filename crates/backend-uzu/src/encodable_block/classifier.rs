use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend, Encoder, Kernels, gpu_types::trie::TrieNode, kernel::PoolingMeanKernel},
    config::classifier::{ClassifierConfig, PoolingType},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
        prediction_head::{PredictionHead, PredictionHeadError},
        transformer::{Transformer, TransformerNewError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum ClassifierError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer error: {0}")]
    PredictionHead(#[from] PredictionHeadError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct Classifier<B: Backend> {
    hidden_dim: usize,
    data_type: DataType,
    embedding: Embedding<B>,
    embedding_norm: Normalization<B>,
    transformer: Transformer<B>,
    pooling: <B::Kernels as Kernels>::PoolingMeanKernel,
    prediction_head: PredictionHead<B>,
}

impl<B: Backend> Classifier<B> {
    pub fn new(
        context: &B::Context,
        config: &ClassifierConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, ClassifierError<B>> {
        let (embedding, _) = Embedding::new(
            context,
            config.vocab_size as u32,
            config.transformer_config.model_dim as u32,
            &config.embedding_config,
            &parameter_tree.subtree("embedding")?,
            data_type,
        )?;

        let embedding_norm = Normalization::new(
            config.transformer_config.model_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.embedding_norm_config,
            &parameter_tree.subtree("embedding_norm")?,
            context,
        )?;

        let transformer = Transformer::new(
            context,
            None,
            data_type,
            &config.transformer_config,
            &parameter_tree.subtree("transformer")?,
        )?;

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
            &parameter_tree.subtree("prediction_head")?,
            context,
        )?;

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

    pub fn max_context_length(&self) -> Option<usize> {
        self.transformer.max_context_length()
    }

    pub fn encode(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, ClassifierError<B>> {
        let embedded = self.embedding.encode_lookup(token_ids, batch_dim, encoder)?;

        let hidden =
            self.embedding_norm.encode(&embedded, 0, batch_dim, None, encoder).map_err(ClassifierError::Backend)?;

        let nodes = (0..batch_dim)
            .map(|index| TrieNode {
                trie_start: index as u32,
                trie_end: (batch_dim - 1) as u32,
                height: index as u32,
            })
            .collect::<Box<[TrieNode]>>();
        let hidden = self
            .transformer
            .encode(hidden, None, &BatchTopology::new(&nodes, true), Some(0..batch_dim), None, encoder, &[])
            .map_err(ClassifierError::Backend)?
            .output
            .unwrap();

        let mut pooled = encoder
            .allocate_scratch(size_for_shape(&[self.hidden_dim], self.data_type))
            .map_err(ClassifierError::Backend)?;
        self.pooling.encode(&hidden, &mut pooled, batch_dim as u32, self.hidden_dim as u32, 1, encoder);

        let logits = self.prediction_head.encode(pooled, 1, encoder).map_err(ClassifierError::Backend)?;

        Ok(logits)
    }
}
