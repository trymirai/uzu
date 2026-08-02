use derive_more::Debug;
use num_traits::AsPrimitive;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, kernel::ActivationTransform,
    },
    config::transformer_layer::{TransformerLayerConfig, TransformerLayerConvConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        convolution::{ConvolutionNewError, SeparableCausalConv},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerNewError, MixerState, attention::rope::PrecalculatedRoPE},
        mlp::{Mlp, MlpBlockNewError, MlpEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbeddingError, PerLayerEmbeddingProjection},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{dispatch_dtype::dispatch_dtype, maybe_mut::MaybeMut},
};

#[derive(Debug, Error)]
pub enum TransformerLayerError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Mixer error: {0}")]
    Mixer(#[from] MixerNewError<B>),
    #[error("Mixer encoding error: {0}")]
    MixerEncode(#[from] MixerEncodeError<B>),
    #[error("MLP error: {0}")]
    MlpBlock(#[from] MlpBlockNewError<B>),
    #[error("MLP encoding error: {0}")]
    MlpEncode(#[from] MlpEncodeError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Convolution error: {0}")]
    Convolution(#[from] ConvolutionNewError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Layer {layer_index} sets post_layer_scalar but has no post_mlp_norm")]
    PostLayerScalarWithoutPostMlpNorm {
        layer_index: u32,
    },
    #[error("Transformer layers except the first one if it doesn't have rht in mixer require pre_mixer_norm_config")]
    MissingPreMixerNormConfig,
}

struct TransformerLayerConv<B: Backend> {
    model_dim: u32,
    pre_conv: SeparableCausalConv<B>,
    kernel_projection: Box<dyn Linear<B>>,
    post_conv: SeparableCausalConv<B>,
    input_rht: Option<(ActivationTransform<B>, B::GlobalGpuBuffer)>,
    coefficient_count: u32,
}

// TODO: saner shortcut

pub struct TransformerLayer<B: Backend> {
    pub layer_index: u32,
    pub kv_source_layer_index: Option<u32>,
    pub pre_mixer_norm: Option<Normalization<B>>,
    mixer_conv: Option<TransformerLayerConv<B>>,
    pub mixer: Box<dyn Mixer<B>>,
    pub post_mixer_norm: Option<Normalization<B>>,
    pub pre_mlp_norm: Normalization<B>,
    mlp_conv: Option<TransformerLayerConv<B>>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<Normalization<B>>,
    pub ple_projection: Option<PerLayerEmbeddingProjection<B>>,
}

impl<B: Backend> TransformerLayerConv<B> {
    async fn new(
        model_dim: u32,
        data_type: DataType,
        input_hadamard_factors: Option<B::GlobalGpuBuffer>,
        config: &TransformerLayerConvConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let pre_conv = SeparableCausalConv::new(
            model_dim,
            config.conv_kernel_size,
            config.conv_group_size,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("pre_conv"),
            context,
        )
        .await?;
        let post_conv = SeparableCausalConv::new(
            model_dim,
            config.conv_kernel_size,
            config.conv_group_size,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("post_conv"),
            context,
        )
        .await?;

        let coefficient_count = config.conv_kernel_size * (model_dim / config.conv_group_size);
        let projection_dim = coefficient_count * 2;
        let kernel_projection = <dyn Linear<B>>::new(
            model_dim,
            projection_dim,
            data_type,
            false,
            &parameter_tree.subtree("kernel_projection"),
            context,
        )
        .await?;

        let input_rht = input_hadamard_factors
            .map(|factors| ActivationTransform::input_rht(context, data_type, true).map(|kernel| (kernel, factors)))
            .transpose()
            .map_err(TransformerLayerError::Backend)?;

        Ok(Self {
            model_dim,
            pre_conv,
            kernel_projection,
            post_conv,
            input_rht,
            coefficient_count,
        })
    }

    fn encode_pre_convolution(
        &self,
        input: impl BufferRef<Backend = B>,
        sequence_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), TransformerLayerError<B>> {
        let mut projection_input =
            command_buffer.allocate_scratch(input.size()).map_err(TransformerLayerError::Backend)?;
        command_buffer.encode_copy(input, &mut projection_input);
        let coefficients = self.kernel_projection.encode(projection_input, sequence_length, command_buffer)?;
        let mut output = self
            .pre_conv
            .encode(input, &coefficients, 2 * self.coefficient_count, 0, sequence_length, command_buffer)
            .map_err(TransformerLayerError::Backend)?;
        if let Some((transform, factors)) = &self.input_rht {
            transform.encode_fp_in_place(&mut output, factors, None, sequence_length, self.model_dim, command_buffer);
        }
        Ok((output, coefficients))
    }

    fn encode_post_convolution(
        &self,
        input: impl BufferRef<Backend = B>,
        coefficients: impl BufferRef<Backend = B>,
        sequence_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.post_conv.encode(
            input,
            coefficients,
            2 * self.coefficient_count,
            self.coefficient_count,
            sequence_length,
            command_buffer,
        )
    }
}

impl<B: Backend> TransformerLayer<B> {
    pub async fn new(
        model_dim: u32,
        hidden_dim: u32,
        num_layers: u32,
        layer_index: u32,
        data_type: DataType,
        layer_config: &TransformerLayerConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerLayerError<B>> {
        let post_layer_scalar = if layer_config.has_post_layer_scalar {
            if layer_config.post_mlp_norm_config.is_none() {
                return Err(TransformerLayerError::PostLayerScalarWithoutPostMlpNorm {
                    layer_index,
                });
            }
            let scalar = dispatch_dtype!(|(T: data_type)| parameter_tree.read_slice::<T>("post_layer_scalar", &[1]).await?[0].as_());
            Some(scalar)
        } else {
            None
        };

        // When a PLE projection is present it owns the post-layer scalar (applied
        // to the combined residual), so the norms must not also apply it.
        let (residual_sum_scalar, output_scalar) = match (post_layer_scalar, layer_config.ple_config.is_none()) {
            (Some(scalar), true) => (PostLayerScalar::ScaleResidualSum(scalar), PostLayerScalar::ScaleOutput(scalar)),
            _ => (PostLayerScalar::None, PostLayerScalar::None),
        };

        let (mixer, mixer_hadamard_factors) = <dyn Mixer<B>>::new(
            model_dim,
            data_type,
            layer_config.rope_config.as_ref(),
            &layer_config.mixer_config,
            &parameter_tree.subtree("mixer"),
            context,
        )
        .await?;

        let (mixer_conv, mixer_hadamard_factors) = match &layer_config.mixer_conv_config {
            Some(config) => (
                Some(
                    TransformerLayerConv::new(
                        model_dim,
                        data_type,
                        mixer_hadamard_factors,
                        config,
                        &parameter_tree.subtree("mixer_conv"),
                        context,
                    )
                    .await?,
                ),
                None,
            ),
            None => (None, mixer_hadamard_factors),
        };

        let pre_mixer_norm = if let Some(pre_mixer_norm_config) = &layer_config.pre_mixer_norm_config {
            Some(
                Normalization::new(
                    model_dim,
                    data_type,
                    mixer_hadamard_factors,
                    if layer_index > 0 {
                        ShortcutMode::Add
                    } else {
                        ShortcutMode::Copy
                    },
                    PostLayerScalar::None,
                    pre_mixer_norm_config,
                    &parameter_tree.subtree("pre_mixer_norm"),
                    context,
                )
                .await?,
            )
        } else {
            if layer_index != 0 || mixer_hadamard_factors.is_some() {
                return Err(TransformerLayerError::MissingPreMixerNormConfig);
            }
            None
        };

        let post_mixer_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(
                Normalization::new(
                    model_dim,
                    data_type,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    norm_config,
                    &parameter_tree.subtree("post_mixer_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            model_dim,
            layer_config.hidden_dim.unwrap_or(hidden_dim),
            data_type,
            &layer_config.mlp_config,
            &parameter_tree.subtree("mlp"),
            context,
        )
        .await?;

        let (mlp_conv, mlp_input_hadamard_factors) = match &layer_config.mlp_conv_config {
            Some(config) => (
                Some(
                    TransformerLayerConv::new(
                        model_dim,
                        data_type,
                        mlp_input_hadamard_factors,
                        config,
                        &parameter_tree.subtree("mlp_conv"),
                        context,
                    )
                    .await?,
                ),
                None,
            ),
            None => (None, mlp_input_hadamard_factors),
        };

        let pre_mlp_norm = Normalization::new(
            model_dim,
            data_type,
            mlp_input_hadamard_factors,
            ShortcutMode::Add,
            residual_sum_scalar,
            &layer_config.pre_mlp_norm_config,
            &parameter_tree.subtree("pre_mlp_norm"),
            context,
        )
        .await?;

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(
                Normalization::new(
                    model_dim,
                    data_type,
                    None,
                    ShortcutMode::None,
                    output_scalar,
                    norm_config,
                    &parameter_tree.subtree("post_mlp_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let ple_projection = if let Some(ple_config) = &layer_config.ple_config {
            let ple_loader = parameter_tree.subtree("ple");
            Some(
                PerLayerEmbeddingProjection::new(
                    model_dim,
                    num_layers,
                    data_type,
                    post_layer_scalar.unwrap_or(1.0),
                    ple_config,
                    &ple_loader,
                    context,
                )
                .await
                .expect("Failed to create per-layer embedding projection"),
            )
        } else {
            None
        };

        Ok(Self {
            layer_index,
            kv_source_layer_index: layer_config.kv_source_layer_index,
            pre_mixer_norm,
            mixer_conv,
            mixer,
            post_mixer_norm,
            pre_mlp_norm,
            mlp_conv,
            mlp,
            post_mlp_norm,
            ple_projection,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        mut shortcut: impl BufferMut<Backend = B>,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("transformer layer {}", self.layer_index));

        let mut hidden = if let Some(pre_mixer_norm) = &self.pre_mixer_norm {
            pre_mixer_norm
                .encode(&input, 0, batch_dim.size(), Some(shortcut.reborrow()), command_buffer)
                .map_err(TransformerLayerError::Backend)?
        } else {
            assert!(self.layer_index == 0);
            command_buffer.encode_copy(&input, shortcut.reborrow());
            input
        };

        let mixer_coefficients = if let Some(convolution) = &self.mixer_conv {
            let (output, coefficients) =
                convolution.encode_pre_convolution(&hidden, batch_dim.size(), command_buffer)?;
            hidden = output;
            Some(coefficients)
        } else {
            None
        };

        // TODO: In prefill outside of sampling suffix in last layer part of mixer (ie out projection) and everything after is dead code
        hidden = self.mixer.encode(hidden, precalculated_rope, batch_dim, state, command_buffer)?;

        if let Some(coefficients) = mixer_coefficients {
            let convolution = self.mixer_conv.as_ref().expect("mixer convolution required");
            hidden = convolution
                .encode_post_convolution(&hidden, &coefficients, batch_dim.size(), command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }

        if let Some(post_mixer_norm) = &self.post_mixer_norm {
            hidden = post_mixer_norm
                .encode(&hidden, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }

        hidden = self
            .pre_mlp_norm
            .encode(&hidden, 0, batch_dim.size(), Some(shortcut.reborrow()), command_buffer)
            .map_err(TransformerLayerError::Backend)?;

        let mlp_coefficients = if let Some(convolution) = &self.mlp_conv {
            let (output, coefficients) =
                convolution.encode_pre_convolution(&hidden, batch_dim.size(), command_buffer)?;
            hidden = output;
            Some(coefficients)
        } else {
            None
        };

        hidden = self.mlp.encode(hidden, batch_dim.size(), command_buffer)?;

        if let Some(coefficients) = mlp_coefficients {
            let convolution = self.mlp_conv.as_ref().expect("mlp convolution required");
            hidden = convolution
                .encode_post_convolution(&hidden, &coefficients, batch_dim.size(), command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            hidden = post_mlp_norm
                .encode(&hidden, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerLayerError::Backend)?;
        }

        if let Some(ple_projection) = &self.ple_projection {
            let per_layer_inputs = per_layer_inputs.expect("per-layer inputs required for PLE layer");
            ple_projection.encode(
                per_layer_inputs,
                shortcut,
                &hidden,
                self.layer_index,
                batch_dim.size(),
                command_buffer,
            )?;
            command_buffer.encode_fill(&mut hidden, 0);
        }

        Ok(hidden)
    }
}
