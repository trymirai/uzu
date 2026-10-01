use crate::{
    backends::common::{Backend, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE},
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{embedding::EmbeddingError, weight_matrix::WeightMatrix},
    parameters::ParameterTree,
};

pub struct EmbeddingResource<B: Backend> {
    pub matrix: WeightMatrix<B>,
    pub output_hadamard_factors: Option<B::GlobalBuffer>,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub data_type: DataType,
}

impl<B: Backend> EmbeddingResource<B> {
    pub fn load_input(
        config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<B>,
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
    ) -> Result<Self, EmbeddingError<B>> {
        let subtree = match config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => "embedding",
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => "input_embedding",
        };
        Self::load(&parameter_tree.subtree(subtree), vocab_size, model_dim, data_type)
    }

    pub fn load(
        tree: &ParameterTree<B>,
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
    ) -> Result<Self, EmbeddingError<B>> {
        let (matrix, output_hadamard_factors) = match tree.metadata::<AnyWeightMatrixSpec>("spec")? {
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                adapter_spec: None,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode: IncoherenceProcessingMode::Output,
                ..
            }) => (
                WeightMatrix::load(
                    &tree.subtree("quantized"),
                    *quantization_spec,
                    Layout::InputOutput,
                    model_dim,
                    vocab_size,
                    data_type,
                )?,
                Some(Self::load_output_hadamard_factors(tree, model_dim)?),
            ),
            spec => (WeightMatrix::load(tree, spec, Layout::InputOutput, model_dim, vocab_size, data_type)?, None),
        };
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(EmbeddingError::UnsupportedConfiguration(
                "output-hadamard factors require a quantized table".into(),
            ));
        }

        Ok(Self {
            matrix,
            output_hadamard_factors,
            vocab_size,
            model_dim,
            data_type,
        })
    }

    pub fn load_output_hadamard_factors(
        tree: &ParameterTree<B>,
        model_dim: u32,
    ) -> Result<B::GlobalBuffer, EmbeddingError<B>> {
        Ok(tree
            .subtree("incoherence_signs")
            .leaf("output_signs")?
            .validate(&[model_dim], DataType::I32)?
            .read_buffer()?)
    }
}
