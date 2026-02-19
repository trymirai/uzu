//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use crate::{
    DataType, DecoderConfig,
    backends::common::{Backend, kernel::matmul::MatmulKernels},
    config::{DecoderLayerType, MixerConfig},
    encodable_block::{
        EncodableBlock, EncodingParameters, LayerExecutables, RMSNorm, Rope,
        transformer_layer::{embed_block, readout_block},
    },
    forward_pass::{
        model_shape::ModelShape,
        state::{ArrayId, ForwardPassState, RopeType},
    },
    parameters::ParameterTree,
};

/// Full decoder executable with all layers and components.
pub struct Decoder<B: Backend> {
    pub embed: Box<dyn EncodableBlock<B>>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: Box<dyn EncodableBlock<B>>,
    pub readout: Box<dyn EncodableBlock<B>>,
    pub global_rope: Option<Rc<Box<dyn EncodableBlock<B>>>>,
    pub local_rope: Option<Rc<Box<dyn EncodableBlock<B>>>>,
}

impl<B: Backend + 'static> Decoder<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new(
        context: Rc<B::Context>,
        decoder_config: Rc<DecoderConfig>,
        root_weight_loader: &ParameterTree<B::Context>,
    ) -> Self {
        let decoder_weight_loader = root_weight_loader.subtree("transformer").expect("transformer subtree not found");

        let embed = embed_block(&decoder_config, context.as_ref(), root_weight_loader);

        let readout = readout_block(&decoder_config, context.as_ref(), root_weight_loader);

        let attention_data_type = Self::attention_data_type(&decoder_config);
        let norm_reference_layer =
            decoder_config.layer_configs.as_ref().map(|configs| &configs[0]).unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config {
            MixerConfig::Attention(attention_config) => {
                attention_config.qkv_projection_config.activation_precision().into()
            },
            MixerConfig::Mamba(mamba_config) => mamba_config.in_projection_config.activation_precision().into(),
            MixerConfig::ShortConv(short_conv_config) => {
                short_conv_config.in_projection_config.activation_precision().into()
            },
        };

        let global_rope = if decoder_config.global_rope_config.is_some() {
            attention_data_type
                .as_ref()
                .map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Global))
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Local))
        } else {
            None
        };

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes = model_shape.sliding_window_length_per_layer.clone();

        let layers = (0..decoder_config.num_layers)
            .map(|layer_index| {
                let layer_config = decoder_config
                    .layer_configs
                    .as_ref()
                    .map(|configs| &configs[layer_index])
                    .unwrap_or(&decoder_config.layer_config);
                let layer_type = model_shape.layer_type(layer_index);
                let rope_for_layer = match layer_type {
                    DecoderLayerType::Transformer => {
                        let mut rope_block = global_rope.clone().expect("Global rope missing for transformer layer");
                        if let (Some(_), Some(local_rope_block)) =
                            (sliding_window_sizes[layer_index], local_rope.clone())
                        {
                            rope_block = local_rope_block;
                        }
                        Some(rope_block)
                    },
                    DecoderLayerType::StateSpace {
                        ..
                    } => None,
                    DecoderLayerType::ShortConv {
                        ..
                    } => None,
                };

                let layer_loader = decoder_weight_loader.subtree(&format!("layers.{}", layer_index)).unwrap();

                LayerExecutables::new(
                    context.clone(),
                    layer_config,
                    layer_type,
                    layer_index,
                    decoder_config.model_dim,
                    decoder_config.hidden_dim,
                    decoder_config.num_heads,
                    decoder_config.head_dim,
                    decoder_config.num_groups,
                    decoder_config.attention_scale,
                    &layer_loader,
                    rope_for_layer,
                )
            })
            .collect::<Vec<_>>();

        let norm_block: Box<dyn EncodableBlock<B>> = Box::new(
            RMSNorm::new(
                context.as_ref(),
                norm_data_type,
                decoder_config.output_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_weight_loader.subtree("output_norm").unwrap(),
            )
            .map(RMSNorm::with_sampling_range)
            .expect("Failed to create output RMS norm kernel"),
        );

        Self {
            embed,
            layers: layers.into_boxed_slice(),
            norm: norm_block,
            readout,
            global_rope,
            local_rope,
        }
    }

    fn create_rope_block(
        context: &B::Context,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableBlock<B>>> {
        let rotation: Box<dyn EncodableBlock<B>> =
            Box::new(Rope::<B>::new(context, data_type, rope_type).expect("Failed to create Rope"));
        Rc::new(rotation)
    }

    fn attention_data_type(decoder_config: &DecoderConfig) -> Option<DataType> {
        (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config
                .attention_config()
                .map(|attention_config| attention_config.qkv_projection_config.activation_precision().into())
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Decoder<B> {
    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        _encoder: &B::ComputeEncoder,
    ) {
        unimplemented!("Decoder does not support shared encoder")
    }

    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        self.embed.encode(state, parameters, command_buffer);

        for layer in self.layers.iter() {
            layer.encode(state, parameters, command_buffer);
        }

        if state.is_prefilling() {
            return;
        }

        self.norm.encode(state, parameters, command_buffer);
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(command_buffer, ArrayId::Main, traces.borrow().output_norm.clone());
        }

        self.readout.encode(state, parameters, command_buffer);
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(command_buffer, ArrayId::Logits, traces.borrow().logits.clone());
        }
    }
}
