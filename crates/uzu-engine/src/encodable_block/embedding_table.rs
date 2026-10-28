use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::{d4s4, embedding::EmbeddingTableKind, hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE},
        kernel::InputEmbeddingLookupKernel,
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        d4s4_spec::D4S4Spec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::weight_matrix::{WeightMatrix, WeightMatrixError},
    safetensors::{ParameterLoaderError, ParameterTree},
};

type LookupKernel<B> = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel;

#[derive(Debug, Error)]
pub enum EmbeddingTableError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Unsupported embedding table configuration: {0}")]
    UnsupportedConfiguration(String),
}

/// Lookup-only D4S4 table: one codebook entry per 4 columns and a ladder scale per 64 columns.
struct D4S4Table<B: Backend> {
    codes: B::GlobalGpuBuffer,
    row_scales: B::GlobalGpuBuffer,
    ladder_indices: B::GlobalGpuBuffer,
    ladder: B::GlobalGpuBuffer,
    codebook: B::GlobalGpuBuffer,
}

enum Storage<B: Backend> {
    Matrix(WeightMatrix<B>),
    D4S4(D4S4Table<B>),
}

struct LookupBindings<'a, B: Backend> {
    values: &'a B::GlobalGpuBuffer,
    scales: Option<&'a B::GlobalGpuBuffer>,
    zero_points: Option<&'a B::GlobalGpuBuffer>,
    biases: Option<&'a B::GlobalGpuBuffer>,
    ladder_indices: Option<&'a B::GlobalGpuBuffer>,
    ladder: Option<&'a B::GlobalGpuBuffer>,
    codebook: Option<&'a B::GlobalGpuBuffer>,
}

impl<B: Backend> Storage<B> {
    fn lookup_bindings(&self) -> LookupBindings<'_, B> {
        match self {
            Self::Matrix(matrix) => LookupBindings {
                values: matrix.values(),
                scales: matrix.scales(),
                zero_points: matrix.zero_points(),
                biases: matrix.biases(),
                ladder_indices: None,
                ladder: None,
                codebook: None,
            },
            Self::D4S4(table) => LookupBindings {
                values: &table.codes,
                scales: Some(&table.row_scales),
                zero_points: None,
                biases: None,
                ladder_indices: Some(&table.ladder_indices),
                ladder: Some(&table.ladder),
                codebook: Some(&table.codebook),
            },
        }
    }
}

pub struct EmbeddingTable<B: Backend> {
    storage: Storage<B>,
    output_hadamard_factors: Option<B::GlobalGpuBuffer>,
    lookup: LookupKernel<B>,
    vocab_size: u32,
    embedding_dim: u32,
}

impl<B: Backend> EmbeddingTable<B> {
    pub async fn load(
        vocab_size: u32,
        embedding_dim: u32,
        data_type: DataType,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, EmbeddingTableError<B>> {
        if !matches!(data_type, DataType::F32 | DataType::BF16) {
            return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                "input embedding lookup does not support {data_type:?}"
            )));
        }
        let load_matrix = async |tree: &ParameterTree<B>, spec| {
            WeightMatrix::load(embedding_dim, vocab_size, data_type, spec, Layout::InputOutput, tree).await
        };
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let (storage, output_hadamard_factors) = match spec {
            AnyWeightMatrixSpec::D4S4Spec(spec) => {
                let (table, factors) = load_d4s4(tree, vocab_size, embedding_dim, data_type, spec).await?;
                (Storage::D4S4(table), Some(factors))
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode: IncoherenceProcessingMode::Output,
                ..
            }) if embedding_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) => {
                let matrix = load_matrix(&tree.subtree("quantized"), *quantization_spec).await?;
                if matrix.quantization().is_none() {
                    return Err(EmbeddingTableError::UnsupportedConfiguration(
                        "output-Hadamard factors require a quantized table".into(),
                    ));
                }
                (Storage::Matrix(matrix), Some(read_output_signs(tree, embedding_dim).await?))
            },
            spec @ AnyWeightMatrixSpec::HybridSpec(_) => {
                return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                    "{spec:?} with embedding dim {embedding_dim}"
                )));
            },
            spec => (Storage::Matrix(load_matrix(tree, spec).await?), None),
        };

        let (table_kind, quantization) = match &storage {
            Storage::D4S4(_) => (EmbeddingTableKind::D4S4, None),
            Storage::Matrix(matrix) => {
                if let Some(info) = matrix.quantization() {
                    (EmbeddingTableKind::Quantized, Some(info))
                } else {
                    (EmbeddingTableKind::Dense, None)
                }
            },
        };
        let group_size = quantization.map(|info| info.group_size);
        let mode = quantization.map(|info| info.mode);
        let method = quantization.map(|info| info.method);
        let use_hadamard = output_hadamard_factors.is_some();
        let lookup = LookupKernel::<B>::new(context, data_type, table_kind, group_size, mode, method, use_hadamard)
            .map_err(EmbeddingTableError::BackendError)?;

        Ok(Self {
            storage,
            output_hadamard_factors,
            lookup,
            vocab_size,
            embedding_dim,
        })
    }

    /// The matrix used by tied readout; D4S4 tables support lookup only.
    pub fn as_matrix(&self) -> Option<&WeightMatrix<B>> {
        match &self.storage {
            Storage::Matrix(matrix) => Some(matrix),
            Storage::D4S4(_) => None,
        }
    }

    /// Gathers one row per token id into `output`, scaling by `scale`.
    pub fn encode_lookup(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        batch_dim: u32,
        scale: f32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let bindings = self.storage.lookup_bindings();
        self.lookup.encode(
            token_ids,
            bindings.values,
            bindings.scales,
            bindings.zero_points,
            bindings.biases,
            self.output_hadamard_factors.as_ref(),
            bindings.ladder_indices,
            bindings.ladder,
            bindings.codebook,
            output,
            batch_dim,
            self.vocab_size,
            self.embedding_dim,
            scale,
            command_buffer,
        );
    }
}

async fn load_d4s4<B: Backend>(
    tree: &ParameterTree<'_, B>,
    vocab_size: u32,
    embedding_dim: u32,
    data_type: DataType,
    spec: D4S4Spec,
) -> Result<(D4S4Table<B>, B::GlobalGpuBuffer), EmbeddingTableError<B>> {
    if spec.layout != Layout::InputOutput || !embedding_dim.is_multiple_of(d4s4::COLUMNS_PER_LADDER_INDEX_BYTE) {
        return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
            "{spec:?} with {data_type:?} and embedding dim {embedding_dim}"
        )));
    }
    let read = async |name: &str, shape: &[u32], data_type| tree.read_buffer(name, shape, data_type).await;
    let codes = read("codes", &[vocab_size, embedding_dim / d4s4::VALUES_PER_CODE], DataType::U8).await?;
    let row_scales = read("row_scales", &[vocab_size], data_type).await?;
    let ladder_indices =
        read("ladder_indices", &[vocab_size, embedding_dim / d4s4::COLUMNS_PER_LADDER_INDEX_BYTE], DataType::U8)
            .await?;
    let ladder = read("ladder", &[d4s4::LADDER_SIZE], DataType::F16).await?;
    let codebook = read("table", &[d4s4::CODEBOOK_SIZE, d4s4::VALUES_PER_CODE], DataType::I8).await?;
    let table = D4S4Table {
        codes,
        row_scales,
        ladder_indices,
        ladder,
        codebook,
    };
    let factors = read("output_hadamard_factors", &[embedding_dim], DataType::I32).await?;
    Ok((table, factors))
}

pub(super) async fn read_output_signs<B: Backend>(
    tree: &ParameterTree<'_, B>,
    embedding_dim: u32,
) -> Result<B::GlobalGpuBuffer, EmbeddingTableError<B>> {
    let signs = tree.subtree("incoherence_signs");
    Ok(signs.read_buffer("output_signs", &[embedding_dim], DataType::I32).await?)
}
