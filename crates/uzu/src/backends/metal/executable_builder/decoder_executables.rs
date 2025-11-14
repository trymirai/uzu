use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::layer_executables::LayerExecutables;
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState, RopeType,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            transformer_layer::{embed_block, readout_block},
        },
        kernel::{RMSNormKernelEncodable, RopeKernelEncodable},
    },
    config::{DecoderConfig, DecoderLayerType, decoder_layer::MixerConfig},
    parameters::ParameterTree,
};

pub struct DecoderExecutables {
    pub embed: Box<dyn EncodableWithState>,
    pub layers: Box<[LayerExecutables]>,
    pub norm: Box<dyn EncodableWithState>,
    pub readout: Box<dyn EncodableWithState>,
    pub global_rope: Option<Rc<Box<dyn EncodableWithState>>>,
    pub local_rope: Option<Rc<Box<dyn EncodableWithState>>>,
}

impl DecoderExecutables {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        decoder_config: Rc<DecoderConfig>,
        decoder_weight_loader: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
    ) -> Self {
        let embed = embed_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
            decoder_weight_loader,
        );

        let readout = readout_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
            decoder_weight_loader,
        );

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
                };

                let layer_loader = decoder_weight_loader
                    .subtree(&format!("layers.{}", layer_index))
                    .unwrap();

                LayerExecutables::new(
                    &mtl_context,
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

        let norm_block: Box<dyn EncodableWithState> = Box::new(
            RMSNormKernelEncodable::new(
                &mtl_context,
                norm_data_type,
                decoder_config.output_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_weight_loader.subtree("output_norm").unwrap(),
            )
            .expect("Failed to create output RMS norm kernel"),
        );

        Self {
            embed: embed,
            layers: layers.into_boxed_slice(),
            norm: norm_block,
            readout: readout,
            global_rope,
            local_rope,
        }
    }

    fn create_rope_block(
        mtl_context: &MTLContext,
        kernel_data_type: KernelDataType,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableWithState>> {
        let rotation: Box<dyn EncodableWithState> = Box::new(
            RopeKernelEncodable::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create RopeKernelEncodable"),
        );
        return Rc::new(rotation);
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

impl EncodableWithState for DecoderExecutables {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        self.embed.encode(state, command_buffer, parameters);

        for layer in self.layers.iter() {
            layer.encode(state, command_buffer, parameters);
        }

        self.norm.encode(state, command_buffer, parameters);
        if let Some(traces) = state.traces.clone() {
            state
                .copy_array(ArrayId::Main, traces.borrow().output_norm.clone());
        }

        self.readout.encode(state, command_buffer, parameters);
        if let Some(traces) = state.traces.clone() {
            state.copy_array(ArrayId::Logits, traces.borrow().logits.clone());
        }
    }
}
