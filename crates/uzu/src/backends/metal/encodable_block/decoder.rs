//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use super::{
    EncodableBlock, EncodingParameters, LayerExecutables, RMSNorm, Rope,
};
use crate::{
    DataType, DecoderConfig,
    backends::metal::{
        KernelDataType, MTLCommandBuffer, MTLContext, ModelShape,
        ProtocolObject,
        compilation_parameters::CompilationConfig,
        encodable_block::transformer_layer::{embed_block, readout_block},
        forward_pass::{ArrayId, ForwardPassState, RopeType},
    },
    config::{DecoderLayerType, MixerConfig},
    parameters::ParameterTree,
};

/// Full decoder executable with all layers and components.
pub struct Decoder {
    pub embed: Box<dyn EncodableBlock>,
    pub layers: Box<[LayerExecutables]>,
    pub norm: Box<dyn EncodableBlock>,
    pub readout: Box<dyn EncodableBlock>,
    pub global_rope: Option<Rc<Box<dyn EncodableBlock>>>,
    pub local_rope: Option<Rc<Box<dyn EncodableBlock>>>,
}

impl Decoder {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        decoder_config: Rc<DecoderConfig>,
        root_weight_loader: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
    ) -> Self {
        let decoder_weight_loader = root_weight_loader
            .subtree("transformer")
            .expect("transformer subtree not found");

        let embed =
            embed_block(&decoder_config, &mtl_context, root_weight_loader);

        let readout =
            readout_block(&decoder_config, &mtl_context, root_weight_loader);

        let attention_data_type = Self::attention_data_type(&decoder_config);
        let norm_reference_layer = decoder_config
            .layer_configs
            .as_ref()
            .map(|configs| &configs[0])
            .unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config
        {
            MixerConfig::Attention(attention_config) => attention_config
                .qkv_projection_config
                .activation_precision()
                .into(),
            MixerConfig::Mamba(mamba_config) => {
                mamba_config.in_projection_config.activation_precision().into()
            },
            MixerConfig::ShortConv(short_conv_config) => short_conv_config
                .in_projection_config
                .activation_precision()
                .into(),
        };

        let global_rope = if decoder_config.global_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| {
                Self::create_rope_block(
                    &mtl_context,
                    (*data_type).into(),
                    RopeType::Global,
                )
            })
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| {
                Self::create_rope_block(
                    &mtl_context,
                    (*data_type).into(),
                    RopeType::Local,
                )
            })
        } else {
            None
        };

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes =
            model_shape.sliding_window_length_per_layer.clone();

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
                        let mut rope_block = global_rope.clone().expect(
                            "Global rope missing for transformer layer",
                        );
                        if let (Some(_), Some(local_rope_block)) = (
                            sliding_window_sizes[layer_index],
                            local_rope.clone(),
                        ) {
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

                let layer_loader = decoder_weight_loader
                    .subtree(&format!("layers.{}", layer_index))
                    .unwrap();

                LayerExecutables::new(
                    mtl_context.clone(),
                    layer_config,
                    layer_type,
                    compilation_config.clone(),
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

        let norm_block: Box<dyn EncodableBlock> = Box::new(
            RMSNorm::new(
                &mtl_context,
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
        mtl_context: &MTLContext,
        kernel_data_type: KernelDataType,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableBlock>> {
        let rotation: Box<dyn EncodableBlock> = Box::new(
            Rope::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create Rope"),
        );
        Rc::new(rotation)
    }

    fn attention_data_type(decoder_config: &DecoderConfig) -> Option<DataType> {
        (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config.attention_config().map(|attention_config| {
                attention_config
                    .qkv_projection_config
                    .activation_precision()
                    .into()
            })
        })
    }
}

impl EncodableBlock for Decoder {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        self.embed.encode(state, command_buffer, parameters);

        for layer in self.layers.iter() {
            layer.encode(state, command_buffer, parameters);
        }

        if state.is_prefilling() {
            return;
        }

        self.norm.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                traces.borrow().output_norm.clone(),
            );
        }

        self.readout.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(
                command_buffer,
                ArrayId::Logits,
                traces.borrow().logits.clone(),
            );
        }
    }
}
