use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::transformer_layer::{TransformerLayerConfig, TransformerLayerConvConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        convolution::{ConvolutionNewError, SeparableCausalConv},
        linear::{Linear, LinearBlockError},
        mixer::{Mixer, MixerNewError, MixerState, attention::rope::PrecalculatedRoPE},
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::PerLayerEmbeddingProjection,
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

#[derive(Debug, Error)]
pub enum TransformerLayerError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Mixer error: {0}")]
    Mixer(#[from] MixerNewError<B>),
    #[error("MLP error: {0}")]
    MlpBlock(#[from] MlpBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Convolution error: {0}")]
    Convolution(#[from] ConvolutionNewError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Layer {layer_index} sets post_layer_scalar but has no post_mlp_norm")]
    PostLayerScalarWithoutPostMlpNorm {
        layer_index: u32,
    },
    #[error("Transformer layers except the first one if it doesn't have rht in mixer require pre_mixer_norm_config")]
    MissingPreMixerNormConfig,
}

struct TransformerLayerConv<B: Backend> {
    pre_conv: SeparableCausalConv<B>,
    kernel_projection: Box<dyn Linear<B>>,
    post_conv: SeparableCausalConv<B>,
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
    fn new(
        context: &B::Context,
        model_dim: u32,
        config: &TransformerLayerConvConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, TransformerLayerError<B>> {
        let pre_conv = SeparableCausalConv::new(
            model_dim,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("pre_conv"),
            context,
        )?;
        let post_conv = SeparableCausalConv::new(
            model_dim,
            data_type,
            &config.conv_config,
            &parameter_tree.subtree("post_conv"),
            context,
        )?;

        let group_size = config.conv_config.coefficient_group_size.expect("coefficient_group_size is required");
        let projection_dim = 2 * config.conv_config.kernel_size * (model_dim / group_size);
        let kernel_projection = <dyn Linear<B>>::new(
            model_dim,
            [projection_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("kernel_projection"),
        )?;

        Ok(Self {
            pre_conv,
            kernel_projection,
            post_conv,
            coefficient_count: config.conv_config.kernel_size * (model_dim / group_size),
        })
    }

    fn encode_pre_convolution(
        &self,
        input: &Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let mut projection_input = encoder.allocate_scratch(input.size())?;
        encoder.encode_copy(input, .., &mut projection_input, ..);
        let coefficients = self.kernel_projection.encode(projection_input, sequence_length, encoder)?;
        let output =
            self.pre_conv.encode(input, &coefficients, 2 * self.coefficient_count, 0, sequence_length, encoder)?;
        Ok((output, coefficients))
    }

    fn encode_post_convolution(
        &self,
        input: &Allocation<B>,
        coefficients: &Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.post_conv.encode(
            input,
            coefficients,
            2 * self.coefficient_count,
            self.coefficient_count,
            sequence_length,
            encoder,
        )
    }
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(
        context: &B::Context,
        model_dim: u32,
        hidden_dim: u32,
        num_layers: u32,
        layer_config: &TransformerLayerConfig,
        layer_index: u32,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, TransformerLayerError<B>> {
        let post_layer_scalar = if layer_config.has_post_layer_scalar {
            if layer_config.post_mlp_norm_config.is_none() {
                return Err(TransformerLayerError::PostLayerScalarWithoutPostMlpNorm {
                    layer_index,
                });
            }
            let leaf = parameter_tree.leaf("post_layer_scalar")?;
            let scalar = match data_type {
                DataType::F32 => leaf.validate(&[1], data_type)?.read_slice::<f32>()?[0],
                DataType::F16 => leaf.validate(&[1], data_type)?.read_slice::<half::f16>()?[0].to_f32(),
                DataType::BF16 => leaf.validate(&[1], data_type)?.read_slice::<half::bf16>()?[0].to_f32(),
                _ => unreachable!(),
            };
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
        )?;

        let pre_mixer_norm = if let Some(pre_mixer_norm_config) = &layer_config.pre_mixer_norm_config {
            Some(Normalization::new(
                model_dim,
                mixer_hadamard_factors,
                if layer_index > 0 {
                    ShortcutMode::Add
                } else {
                    ShortcutMode::Copy
                },
                PostLayerScalar::None,
                data_type,
                pre_mixer_norm_config,
                &parameter_tree.subtree("pre_mixer_norm"),
                context,
            )?)
        } else {
            if layer_index != 0 || mixer_hadamard_factors.is_some() {
                return Err(TransformerLayerError::MissingPreMixerNormConfig);
            }
            None
        };

        let post_mixer_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(Normalization::new(
                model_dim,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                data_type,
                norm_config,
                &parameter_tree.subtree("post_mixer_norm"),
                context,
            )?)
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            layer_config.hidden_dim.unwrap_or(hidden_dim),
            context,
            &parameter_tree.subtree("mlp"),
            data_type,
        )?;

        let pre_mlp_norm = Normalization::new(
            model_dim,
            mlp_input_hadamard_factors,
            ShortcutMode::Add,
            residual_sum_scalar,
            data_type,
            &layer_config.pre_mlp_norm_config,
            &parameter_tree.subtree("pre_mlp_norm"),
            context,
        )?;

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(Normalization::new(
                model_dim,
                None,
                ShortcutMode::None,
                output_scalar,
                data_type,
                norm_config,
                &parameter_tree.subtree("post_mlp_norm"),
                context,
            )?)
        } else {
            None
        };

        let ple_projection = layer_config.ple_config.as_ref().map(|ple_config| {
            let ple_loader = parameter_tree.subtree("ple");
            PerLayerEmbeddingProjection::new(
                context,
                ple_config,
                model_dim,
                num_layers,
                post_layer_scalar.unwrap_or(1.0),
                data_type,
                &ple_loader,
            )
            .expect("Failed to create per-layer embedding projection")
        });

        let mixer_conv = layer_config
            .mixer_conv_config
            .as_ref()
            .map(|config| {
                TransformerLayerConv::new(context, model_dim, config, &parameter_tree.subtree("mixer_conv"), data_type)
            })
            .transpose()?;
        let mlp_conv = layer_config
            .mlp_conv_config
            .as_ref()
            .map(|config| {
                TransformerLayerConv::new(context, model_dim, config, &parameter_tree.subtree("mlp_conv"), data_type)
            })
            .transpose()?;

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
        input: Allocation<B>,
        shortcut: &mut Allocation<B>,
        per_layer_inputs: Option<&Allocation<B>>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group(&format!("transformer layer {}", self.layer_index));

        let mut hidden = if let Some(pre_mixer_norm) = &self.pre_mixer_norm {
            pre_mixer_norm.encode(&input, 0, batch_dim.size(), Some(shortcut), encoder)?
        } else {
            assert!(self.layer_index == 0);
            encoder.encode_copy(&input, .., shortcut, ..);
            input
        };

        let mixer_coefficients = if let Some(convolution) = &self.mixer_conv {
            let (output, coefficients) = convolution.encode_pre_convolution(&hidden, batch_dim.size(), encoder)?;
            hidden = output;
            Some(coefficients)
        } else {
            None
        };

        // TODO: In prefill outside of sampling suffix in last layer part of mixer (ie out projection) and everything after is dead code
        hidden = self.mixer.encode(hidden, precalculated_rope, batch_dim, state, encoder)?;

        if let Some(coefficients) = mixer_coefficients {
            let convolution = self.mixer_conv.as_ref().expect("mixer convolution required");
            hidden = convolution.encode_post_convolution(&hidden, &coefficients, batch_dim.size(), encoder)?;
        }

        if let Some(post_mixer_norm) = &self.post_mixer_norm {
            hidden = post_mixer_norm.encode(&hidden, 0, batch_dim.size(), None, encoder)?;
        }

        hidden = self.pre_mlp_norm.encode(&hidden, 0, batch_dim.size(), Some(shortcut), encoder)?;

        let mlp_coefficients = if let Some(convolution) = &self.mlp_conv {
            let (output, coefficients) = convolution.encode_pre_convolution(&hidden, batch_dim.size(), encoder)?;
            hidden = output;
            Some(coefficients)
        } else {
            None
        };

        hidden = self.mlp.encode(hidden, batch_dim.size(), encoder)?;

        if let Some(coefficients) = mlp_coefficients {
            let convolution = self.mlp_conv.as_ref().expect("mlp convolution required");
            hidden = convolution.encode_post_convolution(&hidden, &coefficients, batch_dim.size(), encoder)?;
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            hidden = post_mlp_norm.encode(&hidden, 0, batch_dim.size(), None, encoder)?;
        }

        if let Some(ple_projection) = &self.ple_projection {
            let per_layer_inputs = per_layer_inputs.expect("per-layer inputs required for PLE layer");
            ple_projection.encode(self.layer_index, per_layer_inputs, shortcut, &hidden, batch_dim.size(), encoder)?;
            encoder.encode_fill(&mut hidden, 0);
        }

        encoder.pop_debug_group();

        Ok(hidden)
    }
}
