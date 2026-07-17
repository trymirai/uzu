mod decode_stager;
mod prefill_stager;
mod row_source;

use std::path::PathBuf;

pub(crate) use decode_stager::DecodeRowStager;
pub(crate) use prefill_stager::{PREFILL_CHUNK_SIZE, PrefillBatch, PrefillStager};
use row_source::RowSource;
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
    parameters::{ParameterLoaderError, ParameterTree},
    staging::{Reservation, Stage, StageView},
};

#[derive(Clone, Copy, Debug, Default)]
pub enum PageCachePolicy {
    #[default]
    Buffered,
    Bypass,
}

#[derive(Clone, Debug, Default)]
pub enum PleStorage {
    #[default]
    Resident,
    FileBacked {
        path: PathBuf,
        page_cache: PageCachePolicy,
    },
}

#[derive(Debug, Error)]
pub enum PerLayerEmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Row staging error: {0}")]
    RowStaging(#[from] std::io::Error),
    #[error("staged PLE decode requires greedy sampling without grammar or speculation")]
    DecodeStagingUnsupported,
    #[error("PLE source does not match row storage")]
    RowSourceMismatch,
    #[error("wrong number of staged PLE rows: expected {expected}, got {actual}")]
    StagedRowsLength {
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
    rows: PleRows<B>,
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
}

enum PleRows<B: Backend> {
    Resident(Allocation<B>),
    FileBacked(RowSource),
}

pub(crate) struct StagedRows<'a, B: Backend> {
    stage: StageView<'a, B>,
    indices: &'a B::DenseBuffer,
    batch_size: usize,
}

pub(crate) enum PleSource<'a, B: Backend> {
    Resident,
    Staged(StagedRows<'a, B>),
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
        storage: &PleStorage,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let total_ple_dim = config.num_layers * config.ple_dim;

        let token_embedding_leaf =
            parameter_tree.leaf("token_embedding")?.validate(&[config.ple_vocab_size, total_ple_dim], data_type)?;
        let rows = match storage {
            PleStorage::Resident => PleRows::Resident(token_embedding_leaf.read_allocation()?),
            PleStorage::FileBacked {
                path,
                page_cache,
            } => {
                let row_bytes = total_ple_dim * data_type.size_in_bytes();
                let source = RowSource::open_exact(
                    path,
                    size_for_shape(&[config.ple_vocab_size, total_ple_dim], data_type),
                    row_bytes,
                )?;
                if matches!(page_cache, PageCachePolicy::Bypass) {
                    source.disable_page_cache()?;
                }
                PleRows::FileBacked(source)
            },
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
            rows,
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
            PleSource::Staged(rows) => {
                if batch_dim != rows.batch_size {
                    return Err(PerLayerEmbeddingError::StagedRowsLength {
                        expected: rows.batch_size,
                        actual: batch_dim,
                    });
                }
                encoder.wait_for_event(rows.stage.ready_event, rows.stage.value);
                self.encode_lookup(rows.indices, rows.stage.allocation, batch_dim, batch_dim, total_ple_dim, encoder)?
            },
            PleSource::Resident => {
                let PleRows::Resident(token_embedding) = &self.rows else {
                    return Err(PerLayerEmbeddingError::RowSourceMismatch);
                };
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

pub(crate) enum PreparedPleRows<B: Backend> {
    Decode(Stage<B>),
    Prefill(PrefillBatch<B>),
}

impl<B: Backend> PreparedPleRows<B> {
    pub(crate) fn into_stage(self) -> Stage<B> {
        match self {
            Self::Decode(stage) => stage,
            Self::Prefill(batch) => batch.stage,
        }
    }
}

pub(crate) struct PleStaging<B: Backend> {
    decode: Option<DecodeRowStager<B>>,
    prefill: Option<PrefillStager<B>>,
}

impl<B: Backend> Unpin for PleStaging<B> {}

pub(crate) struct PleSample<'a, B: Backend>(Option<(&'a mut DecodeRowStager<B>, Reservation)>);

impl<B: Backend> PleSample<'_, B> {
    pub(crate) fn readback(&mut self) -> Option<&mut B::DenseBuffer> {
        self.0.as_mut().map(|(stager, reservation)| stager.sample_readback(reservation))
    }

    pub(crate) fn submit(
        self,
        encoder: &mut Encoder<B>,
    ) -> std::io::Result<Option<PreparedPleRows<B>>> {
        self.0
            .map(|(stager, reservation)| stager.publish_sample(reservation, encoder).map(PreparedPleRows::Decode))
            .transpose()
    }
}

impl<B: Backend> PleStaging<B> {
    pub(crate) fn new(
        embedding: Option<&PerLayerEmbedding<B>>,
        context: &B::Context,
        decode_staging_supported: bool,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let Some(PleRows::FileBacked(source)) = embedding.map(|embedding| &embedding.rows) else {
            return Ok(Self {
                decode: None,
                prefill: None,
            });
        };
        if !decode_staging_supported {
            return Err(PerLayerEmbeddingError::DecodeStagingUnsupported);
        }
        Ok(Self {
            decode: Some(DecodeRowStager::new(context, source.try_clone()?)?),
            prefill: Some(PrefillStager::new(context, source.try_clone()?)?),
        })
    }

    pub(crate) fn begin_sample(&mut self) -> std::io::Result<PleSample<'_, B>> {
        let sample = self
            .decode
            .as_mut()
            .map(|stager| stager.reserve_sample().map(|reservation| (stager, reservation)))
            .transpose()?;
        Ok(PleSample(sample))
    }

    pub(crate) fn stage_token(
        &mut self,
        token_id: u64,
    ) -> std::io::Result<Option<PreparedPleRows<B>>> {
        self.decode.as_mut().map(|stager| stager.stage_token(token_id).map(PreparedPleRows::Decode)).transpose()
    }

    pub(crate) fn stage_prefill(
        &mut self,
        token_ids: &[u64],
    ) -> std::io::Result<Option<PreparedPleRows<B>>> {
        self.prefill.as_mut().map(|stager| stager.stage(token_ids).map(PreparedPleRows::Prefill)).transpose()
    }

    pub(crate) fn source<'a>(
        &'a self,
        rows: Option<&'a PreparedPleRows<B>>,
    ) -> PleSource<'a, B> {
        match rows {
            Some(PreparedPleRows::Decode(stage)) => {
                PleSource::Staged(self.decode.as_ref().expect("decode stager missing").view(stage))
            },
            Some(PreparedPleRows::Prefill(batch)) => {
                PleSource::Staged(self.prefill.as_ref().expect("prefill stager missing").view(batch))
            },
            None => {
                debug_assert!(self.decode.is_none(), "file-backed PLE requires staged rows");
                PleSource::Resident
            },
        }
    }

    pub(crate) fn is_file_backed(&self) -> bool {
        self.prefill.is_some()
    }
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
