use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::{
            d4s4, embedding::EmbeddingTableKind, hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            quantization_method::QuantizationMethod,
        },
        kernel::{
            ActivationTransform, InputEmbeddingLookupKernel,
            matmul::{MatmulB, QuantizedCorrection},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode},
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

struct Part<B: Backend> {
    storage: Storage<B>,
    lookup: LookupKernel<B>,
    first_row: u32,
    rows: u32,
}

enum OutputRotation<B: Backend> {
    Fused(B::GlobalGpuBuffer),
    Separate(B::GlobalGpuBuffer, ActivationTransform<B>),
}

pub struct EmbeddingTable<B: Backend> {
    parts: Box<[Part<B>]>,
    output_rotation: Option<OutputRotation<B>>,
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
        let load_matrix = async |tree: &ParameterTree<B>, rows, spec| {
            WeightMatrix::load(embedding_dim, rows, data_type, spec, Layout::InputOutput, tree).await
        };
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let (storages, output_rotation) = match spec {
            AnyWeightMatrixSpec::D4S4Spec(spec) => {
                let table = load_d4s4(tree, vocab_size, embedding_dim, data_type, spec.layout).await?;
                let factors = tree.read_buffer("output_hadamard_factors", &[embedding_dim], DataType::I32).await?;
                (vec![(Storage::D4S4(table), vocab_size)], Some(OutputRotation::Fused(factors)))
            },
            AnyWeightMatrixSpec::LatticeSpec(spec) => {
                let table = load_d4s4(tree, vocab_size, embedding_dim, data_type, spec.layout).await?;
                (vec![(Storage::D4S4(table), vocab_size)], None)
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_spec:
                    Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                        block_size: block_size @ (HADAMARD_TRANSFORM_BLOCK_SIZE | 128),
                        processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    })),
                ..
            }) if embedding_dim.is_multiple_of(block_size) => {
                let storages = match *quantization_spec {
                    AnyWeightMatrixSpec::LatticeSpec(spec) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => {
                        let table =
                            load_d4s4(&tree.subtree("quantized"), vocab_size, embedding_dim, data_type, spec.layout)
                                .await?;
                        vec![(Storage::D4S4(table), vocab_size)]
                    },
                    AnyWeightMatrixSpec::OutputStackSpec(stack) => {
                        let stack_rows = stack.parts.iter().map(|part| part.rows).sum::<u32>();
                        if stack_rows != vocab_size {
                            return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                                "stacked table has {stack_rows} rows, expected {vocab_size}"
                            )));
                        }
                        let mut storages = Vec::with_capacity(stack.parts.len());
                        for (index, part) in stack.parts.into_iter().enumerate() {
                            let part_tree = tree.subtree(&format!("quantized.parts.{index}"));
                            storages.push((
                                Storage::Matrix(load_matrix(&part_tree, part.rows, part.spec).await?),
                                part.rows,
                            ));
                        }
                        storages
                    },
                    spec => {
                        let matrix = load_matrix(&tree.subtree("quantized"), vocab_size, spec).await?;
                        vec![(Storage::Matrix(matrix), vocab_size)]
                    },
                };
                if storages.iter().any(|(storage, _)| {
                    matches!(storage, Storage::Matrix(matrix) if matrix.matmul_b().quantized().is_none())
                }) {
                    return Err(EmbeddingTableError::UnsupportedConfiguration(
                        "output-Hadamard factors require a quantized table".into(),
                    ));
                }
                let factors = read_output_signs(tree, embedding_dim).await?;
                let rotation = if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE {
                    OutputRotation::Fused(factors)
                } else {
                    let transform = ActivationTransform::output_rht(context, data_type, None, block_size, true)
                        .map_err(EmbeddingTableError::BackendError)?;
                    OutputRotation::Separate(factors, transform)
                };
                (storages, Some(rotation))
            },
            spec @ AnyWeightMatrixSpec::HybridSpec(_) => {
                return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                    "{spec:?} with embedding dim {embedding_dim}"
                )));
            },
            spec => (vec![(Storage::Matrix(load_matrix(tree, vocab_size, spec).await?), vocab_size)], None),
        };

        let use_hadamard = matches!(output_rotation, Some(OutputRotation::Fused(_)));
        let mut parts = Vec::with_capacity(storages.len());
        let mut first_row = 0;
        for (storage, rows) in storages {
            let (table_kind, quantization) = match &storage {
                Storage::D4S4(_) => (EmbeddingTableKind::D4S4, None),
                Storage::Matrix(matrix) => match matrix.matmul_b() {
                    MatmulB::Quantized(quantized) => (EmbeddingTableKind::Quantized, Some(quantized)),
                    _ => (EmbeddingTableKind::Dense, None),
                },
            };
            let group_size = quantization.as_ref().map(|quantized| quantized.group_size);
            let mode = quantization.as_ref().map(|quantized| quantized.mode);
            let method = quantization.as_ref().map(|quantized| match quantized.correction {
                QuantizedCorrection::Symmetric => QuantizationMethod::ScaleSymmetric,
                QuantizedCorrection::Biases(_) => QuantizationMethod::ScaleBias,
                QuantizedCorrection::ZeroPoints(_) => QuantizationMethod::ScaleZeroPoint,
            });
            let lookup = LookupKernel::<B>::new(context, data_type, table_kind, group_size, mode, method, use_hadamard)
                .map_err(EmbeddingTableError::BackendError)?;
            parts.push(Part {
                storage,
                lookup,
                first_row,
                rows,
            });
            first_row += rows;
        }

        Ok(Self {
            parts: parts.into_boxed_slice(),
            output_rotation,
            embedding_dim,
        })
    }

    /// The matrix used by tied readout; D4S4 tables support lookup only.
    pub fn as_matrix(&self) -> Option<&WeightMatrix<B>> {
        if let [part] = &*self.parts
            && let Storage::Matrix(matrix) = &part.storage
            && !matches!(self.output_rotation, Some(OutputRotation::Separate(..)))
        {
            Some(matrix)
        } else {
            None
        }
    }

    /// Gathers one row per token id into `output`, scaling by `scale`.
    pub fn encode_lookup(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        mut output: impl BufferMut<Backend = B>,
        batch_dim: u32,
        scale: f32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let fused_hadamard_factors = match &self.output_rotation {
            Some(OutputRotation::Fused(factors)) => Some(factors),
            _ => None,
        };
        for part in &self.parts {
            let bindings = part.storage.lookup_bindings();
            part.lookup.encode(
                token_ids,
                bindings.values,
                bindings.scales,
                bindings.zero_points,
                bindings.biases,
                fused_hadamard_factors,
                bindings.ladder_indices,
                bindings.ladder,
                bindings.codebook,
                output.reborrow(),
                batch_dim,
                part.rows,
                part.first_row,
                self.embedding_dim,
                scale,
                command_buffer,
            );
        }
        if let Some(OutputRotation::Separate(factors, transform)) = &self.output_rotation {
            transform.encode_fp_in_place(output, factors, None, batch_dim, self.embedding_dim, command_buffer);
        }
    }
}

async fn load_d4s4<B: Backend>(
    tree: &ParameterTree<'_, B>,
    vocab_size: u32,
    embedding_dim: u32,
    data_type: DataType,
    layout: Layout,
) -> Result<D4S4Table<B>, EmbeddingTableError<B>> {
    if layout != Layout::InputOutput || !embedding_dim.is_multiple_of(d4s4::COLUMNS_PER_LADDER_INDEX_BYTE) {
        return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
            "{layout:?} with {data_type:?} and embedding dim {embedding_dim}"
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
    Ok(table)
}

pub(super) async fn read_output_signs<B: Backend>(
    tree: &ParameterTree<'_, B>,
    embedding_dim: u32,
) -> Result<B::GlobalGpuBuffer, EmbeddingTableError<B>> {
    let signs = tree.subtree("incoherence_signs");
    Ok(signs.read_buffer("output_signs", &[embedding_dim], DataType::I32).await?)
}
