use std::{ops::Range, sync::Arc};

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
            InputEmbeddingLookupKernel,
            activation_transform::SUPPORTED_HADAMARD_BLOCK_SIZES,
            matmul::{MatmulB, QuantizedCorrection},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode},
        output_stack::OutputStackError,
    },
    data_type::DataType,
    encodable_block::weight_matrix::{WeightMatrix, WeightMatrixError},
    safetensors::{ParameterLoaderError, ParameterTree},
};

type LookupKernel<B> = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel;

#[derive(Debug, Error)]
pub enum EmbeddingTableError<B: Backend> {
    #[error(transparent)]
    OutputStack(#[from] OutputStackError),
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
    Matrix(Arc<WeightMatrix<B>>),
    D4S4(D4S4Table<B>),
}

struct Part<B: Backend> {
    storage: Storage<B>,
    kernel: LookupKernel<B>,
    rows: Range<u32>,
    rotation: Option<(u32, Arc<B::GlobalGpuBuffer>)>,
}

struct PendingPart<B: Backend> {
    rows: Range<u32>,
    spec: AnyWeightMatrixSpec,
    path: String,
    rotation: Option<(u32, Arc<B::GlobalGpuBuffer>)>,
}

struct LookupBindings<'a, B: Backend> {
    values: &'a B::GlobalGpuBuffer,
    scales: Option<&'a B::GlobalGpuBuffer>,
    zero_points: Option<&'a B::GlobalGpuBuffer>,
    biases: Option<&'a B::GlobalGpuBuffer>,
    d4s4: Option<&'a D4S4Table<B>>,
}

pub struct EmbeddingTable<B: Backend> {
    parts: Box<[Part<B>]>,
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
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        Self::load_spec(vocab_size, embedding_dim, data_type, spec, tree, context).await
    }

    pub async fn load_tied(
        vocab_size: u32,
        embedding_dim: u32,
        data_type: DataType,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Arc<WeightMatrix<B>>), EmbeddingTableError<B>> {
        let unsupported = || {
            EmbeddingTableError::UnsupportedConfiguration(
                "tied embeddings need a matrix table with no rotation or Hadamard-32".into(),
            )
        };
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if matches!(spec, AnyWeightMatrixSpec::OutputStackSpec(_)) {
            return Err(unsupported());
        }
        let table = EmbeddingTable::load_spec(vocab_size, embedding_dim, data_type, spec, tree, context).await?;
        let [
            Part {
                storage: Storage::Matrix(matrix),
                rotation,
                ..
            },
        ] = &*table.parts
        else {
            return Err(unsupported());
        };
        if rotation.as_ref().is_some_and(|(size, _)| *size != HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return Err(unsupported());
        }
        let matrix = matrix.clone();
        Ok((table, matrix))
    }

    pub fn readout_input_hadamard_factors(&self) -> Option<Arc<B::GlobalGpuBuffer>> {
        self.parts.first()?.rotation.as_ref().map(|(_, signs)| signs.clone())
    }

    async fn load_spec(
        vocab_size: u32,
        embedding_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, EmbeddingTableError<B>> {
        if !matches!(data_type, DataType::F32 | DataType::BF16) {
            return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                "input embedding lookup does not support {data_type:?}"
            )));
        }
        let mut pending = vec![PendingPart::<B> {
            rows: 0..vocab_size,
            spec,
            path: String::new(),
            rotation: None,
        }];
        let mut parts = Vec::new();
        while let Some(PendingPart {
            rows: range,
            spec,
            path,
            mut rotation,
        }) = pending.pop()
        {
            let rows = range.end - range.start;
            let part_tree = (!path.is_empty()).then(|| tree.subtree(&path));
            let part_tree = part_tree.as_ref().unwrap_or(tree);
            let child_path = |name: &str| {
                if path.is_empty() {
                    name.to_owned()
                } else {
                    format!("{path}.{name}")
                }
            };
            let storage = match spec {
                AnyWeightMatrixSpec::OutputStackSpec(stack) => {
                    let parts = stack.into_parts(rows)?;
                    for (index, (local, spec)) in parts.into_iter().enumerate().rev() {
                        pending.push(PendingPart {
                            rows: range.start + local.start..range.start + local.end,
                            spec,
                            path: child_path(&format!("parts.{index}")),
                            rotation: rotation.clone(),
                        });
                    }
                    continue;
                },
                AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                    quantization_spec,
                    incoherence_spec:
                        Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                            block_size,
                            processing_mode: IncoherenceProcessingMode::Output,
                            ..
                        })),
                    ..
                }) if rotation.is_none()
                    && SUPPORTED_HADAMARD_BLOCK_SIZES.contains(&block_size)
                    && embedding_dim.is_multiple_of(block_size) =>
                {
                    let factors = Arc::new(
                        part_tree
                            .read_buffer("incoherence_signs.output_signs", &[embedding_dim], DataType::I32)
                            .await?,
                    );
                    pending.push(PendingPart {
                        rows: range,
                        spec: *quantization_spec,
                        path: child_path("quantized"),
                        rotation: Some((block_size, factors)),
                    });
                    continue;
                },
                spec @ AnyWeightMatrixSpec::HybridSpec(_) => {
                    return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                        "{spec:?} at {path:?} with embedding dim {embedding_dim}"
                    )));
                },
                AnyWeightMatrixSpec::D4S4Spec(spec) if rotation.is_none() => {
                    let factors =
                        part_tree.read_buffer("output_hadamard_factors", &[embedding_dim], DataType::I32).await?;
                    rotation = Some((HADAMARD_TRANSFORM_BLOCK_SIZE, Arc::new(factors)));
                    Storage::D4S4(load_d4s4(part_tree, rows, embedding_dim, data_type, spec.layout).await?)
                },
                AnyWeightMatrixSpec::LatticeSpec(spec) => {
                    Storage::D4S4(load_d4s4(part_tree, rows, embedding_dim, data_type, spec.layout).await?)
                },
                spec => Storage::Matrix(Arc::new(
                    WeightMatrix::load(embedding_dim, rows, data_type, spec, Layout::InputOutput, part_tree).await?,
                )),
            };
            let (table_kind, quantization) = match &storage {
                Storage::D4S4(_) => (EmbeddingTableKind::D4S4, None),
                Storage::Matrix(matrix) => match matrix.matmul_b() {
                    MatmulB::Quantized(quantized) => (EmbeddingTableKind::Quantized, Some(quantized)),
                    _ if rotation.is_some() => {
                        return Err(EmbeddingTableError::UnsupportedConfiguration(
                            "output-Hadamard factors require a quantized table".into(),
                        ));
                    },
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
            let hadamard_size = rotation.as_ref().map_or(0, |(size, _)| *size);
            let lookup =
                LookupKernel::<B>::new(context, data_type, table_kind, group_size, mode, method, hadamard_size)
                    .map_err(EmbeddingTableError::BackendError)?;
            parts.push(Part {
                storage,
                kernel: lookup,
                rows: range,
                rotation,
            });
        }
        Ok(Self {
            parts: parts.into_boxed_slice(),
            embedding_dim,
        })
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
        for part in &self.parts {
            let bindings = match &part.storage {
                Storage::Matrix(matrix) => LookupBindings {
                    values: matrix.values(),
                    scales: matrix.scales(),
                    zero_points: matrix.zero_points(),
                    biases: matrix.biases(),
                    d4s4: None,
                },
                Storage::D4S4(table) => LookupBindings {
                    values: &table.codes,
                    scales: Some(&table.row_scales),
                    zero_points: None,
                    biases: None,
                    d4s4: Some(table),
                },
            };
            part.kernel.encode(
                token_ids,
                bindings.values,
                bindings.scales,
                bindings.zero_points,
                bindings.biases,
                part.rotation.as_ref().map(|(_, factors)| &**factors),
                bindings.d4s4.map(|table| &table.ladder_indices),
                bindings.d4s4.map(|table| &table.ladder),
                bindings.d4s4.map(|table| &table.codebook),
                output.reborrow(),
                batch_dim,
                part.rows.end - part.rows.start,
                part.rows.start,
                self.embedding_dim,
                scale,
                command_buffer,
            );
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
    Ok(D4S4Table {
        codes,
        row_scales,
        ladder_indices,
        ladder,
        codebook,
    })
}
