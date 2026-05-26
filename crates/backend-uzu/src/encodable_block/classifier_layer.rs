use std::rc::Rc;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder, Kernels, kernel::TensorAddSwapKernel},
    config::{transformer::TransformerConfig, transformer_layer::TransformerLayerConfig},
    encodable_block::{
        Attention, AttentionError, LayerArguments, Mlp, MlpBlockError, Normalization, NormalizationError, QkUnpack,
        Rope,
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum ClassifierLayerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Attention error: {0}")]
    AttentionError(#[from] AttentionError<B>),
    #[error("MLP error: {0}")]
    MlpBlockError(#[from] MlpBlockError<B>),
    #[error("Normalization error: {0}")]
    NormalizationError(#[from] NormalizationError<B>),
    #[error("classifier layers must use attention")]
    NonAttentionMixer,
    #[error("classifier does not support attention hadamard")]
    AttentionHadamardUnsupported,
    #[error("classifier does not support MLP hadamard")]
    MlpHadamardUnsupported,
}

pub struct ClassifierLayer<B: Backend> {
    pre_attention_norm: Option<Normalization<B>>,
    attention: Attention<B>,
    post_attention_norm: Option<Normalization<B>>,
    mixer_residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    post_mlp_norm: Option<Normalization<B>>,
    mlp_residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    model_dim: usize,
}

impl<B: Backend> ClassifierLayer<B> {
    pub fn new(
        context: &B::Context,
        transformer_config: &TransformerConfig,
        layer_config: &TransformerLayerConfig,
        layer_loader: &ParameterTree<B>,
        rope: Rc<Rope<B>>,
        qk_unpack: Rc<QkUnpack<B>>,
        data_type: DataType,
    ) -> Result<Self, ClassifierLayerError<B>> {
        let attention_config =
            layer_config.mixer_config.as_attention().ok_or(ClassifierLayerError::NonAttentionMixer)?;

        let pre_attention_norm = if let Some(norm_config) = &layer_config.pre_mixer_norm_config {
            Some(Normalization::new(
                context,
                data_type,
                transformer_config.model_dim,
                norm_config.clone(),
                &layer_loader.subtree("pre_mixer_norm")?,
            )?)
        } else {
            None
        };

        let (attention, attention_hadamard_factors) = Attention::new(
            context,
            transformer_config.model_dim,
            data_type,
            attention_config,
            &layer_loader.subtree("mixer")?,
            rope,
            qk_unpack,
            false,
        )?;
        if attention_hadamard_factors.is_some() {
            return Err(ClassifierLayerError::AttentionHadamardUnsupported);
        }

        let post_attention_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(Normalization::new(
                context,
                data_type,
                transformer_config.model_dim,
                norm_config.clone(),
                &layer_loader.subtree("post_mixer_norm")?,
            )?)
        } else {
            None
        };

        let mixer_residual_add = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)
            .map_err(ClassifierLayerError::BackendError)?;

        let pre_mlp_norm = Normalization::new(
            context,
            data_type,
            transformer_config.model_dim,
            layer_config.pre_mlp_norm_config.clone(),
            &layer_loader.subtree("pre_mlp_norm")?,
        )?;

        let (mlp, mlp_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            transformer_config.model_dim,
            layer_config.hidden_dim.unwrap_or(transformer_config.hidden_dim),
            context,
            &layer_loader.subtree("mlp")?,
            data_type,
        )?;

        if mlp_hadamard_factors.is_some() {
            return Err(ClassifierLayerError::MlpHadamardUnsupported);
        }

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(Normalization::new(
                context,
                data_type,
                transformer_config.model_dim,
                norm_config.clone(),
                &layer_loader.subtree("post_mlp_norm")?,
            )?)
        } else {
            None
        };

        let mlp_residual_add = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)
            .map_err(ClassifierLayerError::BackendError)?;

        Ok(Self {
            pre_attention_norm,
            attention,
            post_attention_norm,
            mixer_residual_add,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            mlp_residual_add,
            model_dim: transformer_config.model_dim,
        })
    }

    pub fn encode(
        &self,
        args: LayerArguments<B>,
        main: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let LayerArguments {
            batch_dim,
            token_positions,
            token_subtrie_ranges,
            rope_buffers,
            #[cfg(feature = "tracing")]
            trace,
            ..
        } = args;
        #[cfg(feature = "tracing")]
        let mut layer_traces = trace;
        let layer_len = batch_dim * self.model_dim;

        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.inputs.allocation_mut(), ..);
        }

        debug_assert_eq!(main.as_buffer_range_ref().range().len(), shortcut.as_buffer_range_ref().range().len());
        encoder.encode_copy(&main, .., shortcut, ..);

        let mut main = if let Some(ref pre_attn_norm) = self.pre_attention_norm {
            pre_attn_norm.encode(&main, 0, batch_dim, encoder)?
        } else {
            main
        };
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.pre_attention_norm.allocation_mut(), ..);
        }

        main = self.attention.encode(
            token_positions,
            token_subtrie_ranges,
            rope_buffers,
            None,
            main,
            batch_dim,
            encoder,
        )?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.attention.allocation_mut(), ..);
        }

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            main = post_attn_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&main, .., layer_traces.post_attention_norm.allocation_mut(), ..);
            }
        }

        self.mixer_residual_add.encode(&mut *shortcut, &mut main, layer_len as u32, encoder);
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.mlp_inputs.allocation_mut(), ..);
        }

        debug_assert_eq!(main.as_buffer_range_ref().range().len(), shortcut.as_buffer_range_ref().range().len());
        encoder.encode_copy(&main, .., shortcut, ..);

        main = self.pre_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.pre_mlp_norm.allocation_mut(), ..);
        }

        main = self.mlp.encode(main, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.mlp.allocation_mut(), ..);
        }

        if let Some(ref post_mlp_norm) = self.post_mlp_norm {
            main = post_mlp_norm.encode(&main, 0, batch_dim, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&main, .., layer_traces.post_mlp_norm.allocation_mut(), ..);
            }
        }

        self.mlp_residual_add.encode(shortcut, &mut main, layer_len as u32, encoder);
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&main, .., layer_traces.outputs.allocation_mut(), ..);
        }

        Ok(main)
    }
}
