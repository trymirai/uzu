use crate::{
    backends::common::{
        Backend,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, d4s4},
    },
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            d4s4_spec::D4S4Spec,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{embedding::EmbeddingError, weight_matrix::WeightMatrix},
    parameters::ParameterTree,
};

pub struct D4S4Table<B: Backend> {
    pub codes: B::GlobalBuffer,
    pub row_scales: B::GlobalBuffer,
    pub ladder_indices: B::GlobalBuffer,
    pub ladder: B::GlobalBuffer,
    pub codebook: B::GlobalBuffer,
}

pub enum EmbeddingStorage<B: Backend> {
    Matrix(WeightMatrix<B>),
    D4S4(D4S4Table<B>),
}

pub struct EmbeddingResource<B: Backend> {
    pub storage: EmbeddingStorage<B>,
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
        if !matches!(data_type, DataType::F32 | DataType::BF16) {
            return Err(EmbeddingError::UnsupportedConfiguration(format!(
                "input embedding lookup does not support {data_type:?}"
            )));
        }
        let load_matrix = |tree: &ParameterTree<B>, spec| {
            WeightMatrix::load(tree, spec, Layout::InputOutput, model_dim, vocab_size, data_type)
        };
        let (storage, output_hadamard_factors) = match tree.metadata::<AnyWeightMatrixSpec>("spec")? {
            AnyWeightMatrixSpec::D4S4Spec(spec) => {
                let (table, factors) = load_d4s4(tree, vocab_size, model_dim, data_type, spec)?;
                (EmbeddingStorage::D4S4(table), Some(factors))
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                adapter_spec: None,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode: IncoherenceProcessingMode::Output,
                ..
            }) if model_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) => {
                let matrix = load_matrix(&tree.subtree("quantized"), *quantization_spec)?;
                if matrix.quantization().is_none() {
                    return Err(EmbeddingError::UnsupportedConfiguration(
                        "output-Hadamard factors require a quantized table".into(),
                    ));
                }
                (EmbeddingStorage::Matrix(matrix), Some(Self::load_output_hadamard_factors(tree, model_dim)?))
            },
            spec @ AnyWeightMatrixSpec::HybridSpec(_) => {
                return Err(EmbeddingError::UnsupportedConfiguration(format!(
                    "{spec:?} with embedding dim {model_dim}"
                )));
            },
            spec => (EmbeddingStorage::Matrix(load_matrix(tree, spec)?), None),
        };

        Ok(Self {
            storage,
            output_hadamard_factors,
            vocab_size,
            model_dim,
            data_type,
        })
    }

    pub fn as_matrix(&self) -> Option<&WeightMatrix<B>> {
        match &self.storage {
            EmbeddingStorage::Matrix(matrix) => Some(matrix),
            EmbeddingStorage::D4S4(_) => None,
        }
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

fn load_d4s4<B: Backend>(
    tree: &ParameterTree<B>,
    vocab_size: u32,
    model_dim: u32,
    data_type: DataType,
    spec: D4S4Spec,
) -> Result<(D4S4Table<B>, B::GlobalBuffer), EmbeddingError<B>> {
    if spec.layout != Layout::InputOutput || !model_dim.is_multiple_of(d4s4::COLUMNS_PER_LADDER_INDEX_BYTE) {
        return Err(EmbeddingError::UnsupportedConfiguration(format!(
            "{spec:?} with {data_type:?} and embedding dim {model_dim}"
        )));
    }
    let read = |name: &str, shape: &[u32], data_type| tree.leaf(name)?.validate(shape, data_type)?.read_buffer();
    let table = D4S4Table {
        codes: read("codes", &[vocab_size, model_dim / d4s4::VALUES_PER_CODE], DataType::U8)?,
        row_scales: read("row_scales", &[vocab_size], data_type)?,
        ladder_indices: read(
            "ladder_indices",
            &[vocab_size, model_dim / d4s4::COLUMNS_PER_LADDER_INDEX_BYTE],
            DataType::U8,
        )?,
        ladder: read("ladder", &[d4s4::LADDER_SIZE], DataType::F16)?,
        codebook: read("table", &[d4s4::CODEBOOK_SIZE, d4s4::VALUES_PER_CODE], DataType::I8)?,
    };
    let factors = read("output_hadamard_factors", &[model_dim], DataType::I32)?;
    Ok((table, factors))
}
