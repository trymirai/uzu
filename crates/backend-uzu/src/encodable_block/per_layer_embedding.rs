mod async_stager;
mod prefill_ring;
mod row_ring;

use std::path::PathBuf;

pub(crate) use async_stager::StageTicket;
pub(crate) use prefill_ring::{PrefillRing, PreparedPrefillBatch, prefill_chunk_size};
pub(crate) use row_ring::{PreparedRow, RowRing};
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Encoder,
        kernel::{
            FullPrecisionEmbeddingLookupKernel, GatedActMulKernel, Kernels, TensorAddBiasKernel, TensorAddScaleKernel,
        },
    },
    config::{
        activation::AnyActivation,
        per_layer_embedding::{PLELayerConfig, PLEModelConfig},
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterRowSource, ParameterTree},
};

#[derive(Debug, Error)]
pub enum PerLayerEmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Row staging error: {0}")]
    RowStaging(#[from] std::io::Error),
    #[error("row token IDs are required when PLE row offload is enabled")]
    RowTokenIdsUnavailable,
    #[error("wrong number of PLE row token IDs: expected {expected}, got {actual}")]
    RowTokenIdsLength {
        expected: usize,
        actual: usize,
    },
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
}

pub struct PerLayerEmbedding<B: Backend> {
    token_embedding: Option<Allocation<B>>,
    token_embedding_rows: Option<ParameterRowSource>,
    token_embedding_lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    model_projection: Box<dyn Linear<B>>,
    projection_norm: Normalization<B>,
    add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    ple_dim: usize,
    num_layers: usize,
    ple_vocab_size: usize,
    model_dim: usize,
    fused_token_scale: f32,
    data_type: DataType,
    offload: PleOffloadConfig,
}

pub(crate) enum PleSource<'a, B: Backend> {
    Resident,
    HostRows(&'a [u64]),
    StagedRow(PreparedRow<'a, B>),
    StagedChunk(PreparedPrefillBatch<'a, B>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeMode {
    Resident,
    Pread,
    Ring,
}

struct PleOffloadConfig {
    decode: DecodeMode,
    prefill: bool,
    row_file: Option<PathBuf>,
    disable_page_cache: bool,
}

impl PleOffloadConfig {
    fn from_env() -> Self {
        let row_ring = env_enabled("UZU_PLE_ROW_RING");
        let row_pread = env_enabled("UZU_PLE_ROW_PREAD");
        Self {
            decode: if row_ring {
                DecodeMode::Ring
            } else if row_pread {
                DecodeMode::Pread
            } else {
                DecodeMode::Resident
            },
            prefill: env_enabled("UZU_PLE_PREFILL_RING"),
            row_file: std::env::var_os("UZU_PLE_ROW_FILE").map(PathBuf::from),
            disable_page_cache: env_enabled("UZU_PLE_ROW_NOCACHE"),
        }
    }

    fn loads_rows(&self) -> bool {
        self.prefill || self.decode != DecodeMode::Resident
    }
}

impl<B: Backend> PerLayerEmbedding<B> {
    fn encode_lookup<I, S>(
        &self,
        indices: &I,
        source: &S,
        batch_dim: usize,
        source_rows: usize,
        total_ple_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, PerLayerEmbeddingError<B>>
    where
        I: AsBufferRangeRef<Buffer = B::DenseBuffer>,
        S: AsBufferRangeRef<Buffer = B::DenseBuffer>,
    {
        let mut output = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, total_ple_dim], self.data_type))
            .map_err(PerLayerEmbeddingError::BackendError)?;
        self.token_embedding_lookup.encode(
            indices,
            source,
            &mut output,
            batch_dim as u32,
            source_rows as u32,
            total_ple_dim as u32,
            self.fused_token_scale,
            encoder,
        );
        Ok(output)
    }

    pub fn new(
        context: &B::Context,
        config: &PLEModelConfig,
        model_dim: usize,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let total_ple_dim = config.num_layers * config.ple_dim;
        let offload = PleOffloadConfig::from_env();

        let token_embedding_leaf =
            parameter_tree.leaf("token_embedding")?.validate(&[config.ple_vocab_size, total_ple_dim], data_type)?;
        let row_bytes = total_ple_dim * data_type.size_in_bytes();
        let token_embedding_rows = if offload.loads_rows() {
            Some(token_embedding_leaf.row_source(row_bytes, offload.row_file.as_deref(), offload.disable_page_cache)?)
        } else {
            None
        };
        let token_embedding = if token_embedding_rows.is_none() {
            Some(token_embedding_leaf.read_allocation()?)
        } else {
            None
        };
        let token_embedding_lookup =
            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                .map_err(PerLayerEmbeddingError::BackendError)?;

        let model_projection = <dyn Linear<B>>::new(
            model_dim,
            [total_ple_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("model_projection")?,
        )?;

        let scale_squared = config.model_projection_scale * config.model_projection_scale;
        let projection_norm_config = {
            let mut adjusted = config.norm_config.clone();
            adjusted.epsilon /= scale_squared;
            adjusted
        };
        let projection_norm = Normalization::new(
            config.ple_dim,
            None,
            false,
            false,
            PostLayerScalar::ScaleOutput(config.input_scale),
            data_type,
            &projection_norm_config,
            &parameter_tree.subtree("projection_norm")?,
            context,
        )?;

        let add_scale = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        Ok(Self {
            token_embedding,
            token_embedding_rows,
            token_embedding_lookup,
            model_projection,
            projection_norm,
            add_scale,
            ple_dim: config.ple_dim,
            num_layers: config.num_layers,
            ple_vocab_size: config.ple_vocab_size,
            model_dim,
            fused_token_scale: config.ple_embed_scale * config.input_scale,
            data_type,
            offload,
        })
    }

    pub fn encode(
        &self,
        token_ids: &Allocation<B>,
        source: PleSource<'_, B>,
        inner_features: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, PerLayerEmbeddingError<B>> {
        let total_ple_dim = self.num_layers * self.ple_dim;
        let total_rows = batch_dim * self.num_layers;
        let total_elements = batch_dim * total_ple_dim;

        let mut model_projection_input = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))
            .map_err(PerLayerEmbeddingError::BackendError)?;
        encoder.encode_copy(inner_features, .., &mut model_projection_input, ..);
        let model_projected = self
            .model_projection
            .encode(model_projection_input, batch_dim, encoder)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        let model_normed = self
            .projection_norm
            .encode(&model_projected, 0, total_rows, None, encoder)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        let token_ple = match source {
            PleSource::StagedChunk(prepared) => {
                if batch_dim != prepared.batch_size {
                    return Err(PerLayerEmbeddingError::RowTokenIdsLength {
                        expected: prepared.batch_size,
                        actual: batch_dim,
                    });
                }
                encoder.wait_for_event(prepared.ready_event, prepared.value);
                let token_ple = self.encode_lookup(
                    prepared.indices,
                    prepared.allocation,
                    batch_dim,
                    prepared.batch_size,
                    total_ple_dim,
                    encoder,
                )?;
                encoder.signal_event(prepared.consumed_event, prepared.value);
                token_ple
            },
            PleSource::StagedRow(prepared) => {
                if batch_dim != 1 {
                    return Err(PerLayerEmbeddingError::RowTokenIdsLength {
                        expected: 1,
                        actual: batch_dim,
                    });
                }
                encoder.wait_for_event(prepared.ready_event, prepared.value);
                let mut row_index = encoder
                    .allocate_constant(DataType::U64.size_in_bytes())
                    .map_err(PerLayerEmbeddingError::BackendError)?;
                row_index.copyin(&[0_u64]);
                self.encode_lookup(&row_index, prepared.allocation, 1, 1, total_ple_dim, encoder)?
            },
            PleSource::HostRows(row_token_ids) => {
                let token_embedding_rows =
                    self.token_embedding_rows.as_ref().ok_or(PerLayerEmbeddingError::RowTokenIdsUnavailable)?;
                if row_token_ids.len() != batch_dim {
                    return Err(PerLayerEmbeddingError::RowTokenIdsLength {
                        expected: batch_dim,
                        actual: row_token_ids.len(),
                    });
                }
                let mut raw_rows = encoder
                    .allocate_constant(size_for_shape(&[batch_dim, total_ple_dim], self.data_type))
                    .map_err(PerLayerEmbeddingError::BackendError)?;
                token_embedding_rows.read_rows(row_token_ids, raw_rows.as_slice_mut::<u8>())?;

                let mut row_indices = encoder
                    .allocate_constant(batch_dim * DataType::U64.size_in_bytes())
                    .map_err(PerLayerEmbeddingError::BackendError)?;
                row_indices.copyin(&(0..batch_dim as u64).collect::<Box<[u64]>>());

                self.encode_lookup(&row_indices, &raw_rows, batch_dim, batch_dim, total_ple_dim, encoder)?
            },
            PleSource::Resident => {
                let token_embedding = self.token_embedding.as_ref().expect("resident PLE embedding missing");
                self.encode_lookup(token_ids, token_embedding, batch_dim, self.ple_vocab_size, total_ple_dim, encoder)?
            },
        };

        let mut per_layer_inputs = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.num_layers, self.ple_dim], self.data_type))
            .map_err(PerLayerEmbeddingError::BackendError)?;
        self.add_scale.encode(
            Some(&token_ple),
            &model_normed,
            &mut per_layer_inputs,
            total_elements as u32,
            total_elements as u32,
            1.0,
            encoder,
        );

        Ok(per_layer_inputs)
    }
}

impl<B: Backend> PerLayerEmbedding<B> {
    pub fn rows_offloaded(&self) -> bool {
        self.token_embedding_rows.is_some()
    }

    pub fn create_row_ring(
        &self,
        context: &B::Context,
    ) -> Result<Option<RowRing<B>>, PerLayerEmbeddingError<B>> {
        if self.offload.decode != DecodeMode::Ring {
            return Ok(None);
        }
        let source =
            self.token_embedding_rows.as_ref().ok_or(PerLayerEmbeddingError::RowTokenIdsUnavailable)?.try_clone()?;
        Ok(Some(RowRing::new(context, source)?))
    }

    pub fn create_prefill_ring(
        &self,
        context: &B::Context,
    ) -> Result<Option<PrefillRing<B>>, PerLayerEmbeddingError<B>> {
        if !self.offload.prefill {
            return Ok(None);
        }
        let source =
            self.token_embedding_rows.as_ref().ok_or(PerLayerEmbeddingError::RowTokenIdsUnavailable)?.try_clone()?;
        Ok(Some(PrefillRing::new(context, source)?))
    }
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
}

pub struct PerLayerEmbeddingProjection<B: Backend> {
    gate: Box<dyn Linear<B>>,
    projection: Box<dyn Linear<B>>,
    norm: Normalization<B>,
    gate_act_mul: <B::Kernels as Kernels>::GatedActMulKernel,
    residual_finalize: <B::Kernels as Kernels>::TensorAddBiasKernel,
    residual_combine: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model_dim: usize,
    ple_dim: usize,
    num_layers: usize,
    activation: AnyActivation,
    post_layer_scalar: f32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbeddingProjection<B> {
    pub fn new(
        context: &B::Context,
        config: &PLELayerConfig,
        model_dim: usize,
        num_layers: usize,
        post_layer_scalar: f32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let gate = <dyn Linear<B>>::new(
            model_dim,
            [config.ple_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("gate")?,
        )?;
        let projection = <dyn Linear<B>>::new(
            config.ple_dim,
            [model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("projection")?,
        )?;
        let norm = Normalization::new(
            model_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.norm_config,
            &parameter_tree.subtree("norm")?,
            context,
        )?;

        let gate_act_mul = <B::Kernels as Kernels>::GatedActMulKernel::new(context, data_type, false, false)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_finalize = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, data_type, true)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_combine = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        Ok(Self {
            gate,
            projection,
            norm,
            gate_act_mul,
            residual_finalize,
            residual_combine,
            model_dim,
            ple_dim: config.ple_dim,
            num_layers,
            activation: config.activation.clone(),
            post_layer_scalar,
            data_type,
        })
    }

    pub fn encode(
        &self,
        layer_index: usize,
        per_layer_input: &Allocation<B>,
        outputs: &mut Allocation<B>,
        hidden: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let length = batch_dim * self.model_dim;

        self.residual_finalize.encode(
            None::<&Allocation<B>>,
            hidden,
            &mut *outputs,
            length as u32,
            length as u32,
            encoder,
        );

        let mut gate_input = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        encoder.encode_copy(outputs, .., &mut gate_input, ..);
        let gate_out = self.gate.encode(gate_input, batch_dim, encoder)?;

        let mut activated = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.ple_dim], self.data_type))?;
        self.gate_act_mul.encode(
            &gate_out,
            Some(per_layer_input),
            &mut activated,
            None::<&Allocation<B>>,
            self.ple_dim as u32,
            batch_dim as u32,
            (layer_index * self.ple_dim) as u32,
            (self.num_layers * self.ple_dim) as u32,
            self.activation.act_type(),
            encoder,
        );

        let projected = self.projection.encode(activated, batch_dim, encoder)?;
        let normed = self.norm.encode(&projected, 0, batch_dim, None, encoder)?;

        self.residual_combine.encode(
            None::<&Allocation<B>>,
            &normed,
            &mut *outputs,
            length as u32,
            length as u32,
            self.post_layer_scalar,
            encoder,
        );

        Ok(())
    }
}
