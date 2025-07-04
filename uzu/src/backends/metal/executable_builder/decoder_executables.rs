use std::{cell::RefCell, rc::Rc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    attention_executable_provider::{
        AttentionExecutableProvider, AttentionExecutableProviderConfig,
    },
    layer_executables::LayerExecutables,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState, MPSGraphBlock, RopeType,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            transformer_layer::{
                embedding_with_placeholder_weights_blocks, rms_norm_block,
                rotation_block, rotation_executable,
            },
        },
        kernel::{RMSNormKernelEncodable, RopeKernelEncodable},
    },
    config::{RoPEConfig, decoder::DecoderConfig},
    parameters::ParameterTree,
};

#[derive(Debug, Clone)]
pub struct KernelsConfig {
    pub use_rope: bool,
    pub use_attention: bool,
    pub use_rms_norm: bool,
}

impl KernelsConfig {
    pub fn default() -> Self {
        Self {
            use_rope: true,
            use_attention: true,
            use_rms_norm: true,
        }
    }
}
pub struct DecoderExecutables {
    pub embedding: Option<MPSGraphBlock>,
    pub embed: MPSGraphBlock,
    pub layers: Box<[LayerExecutables]>,
    pub norm: Box<dyn EncodableWithState>,
    pub readout: MPSGraphBlock,
    pub global_rope: Rc<Box<dyn EncodableWithState>>,
    pub local_rope: Option<Rc<Box<dyn EncodableWithState>>>,
}

impl DecoderExecutables {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        decoder_config: Rc<DecoderConfig>,
        decoder_weight_loader: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
        attention_executable_provider_config: AttentionExecutableProviderConfig,
        kernels_config: KernelsConfig,
    ) -> Self {
        let embedding_blocks = embedding_with_placeholder_weights_blocks(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
        );

        let global_rope = Self::create_rope_block(
            &mtl_context,
            &decoder_config,
            &decoder_weight_loader,
            compilation_config.clone(),
            &kernels_config,
            String::from("global_rope"),
            &decoder_config.global_rope_config,
            RopeType::Global,
        );

        let local_rope: Option<Rc<Box<dyn EncodableWithState>>>;
        if let Some(local_rope_config) = &decoder_config.local_rope_config {
            local_rope = Some(Self::create_rope_block(
                &mtl_context,
                &decoder_config,
                &decoder_weight_loader,
                compilation_config.clone(),
                &kernels_config,
                String::from("local_rope"),
                local_rope_config,
                RopeType::Local,
            ));
        } else {
            local_rope = None;
        }

        let attention_executable_provider: Option<
            Rc<RefCell<AttentionExecutableProvider>>,
        >;
        if kernels_config.use_attention {
            attention_executable_provider = None;
        } else {
            attention_executable_provider =
                Some(Rc::new(RefCell::new(AttentionExecutableProvider::new(
                    mtl_context.clone(),
                    decoder_config.clone(),
                    compilation_config.clone(),
                    attention_executable_provider_config,
                ))));
        }

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes = model_shape.sliding_window_length_per_layer;

        let intermediate_data_type: DataType = decoder_config
            .layer_config
            .attention_config
            .qkv_projection_config
            .activation_precision()
            .into();

        let layers = (0..decoder_config.num_layers)
            .map(|layer_index| {
                let mut rope = global_rope.clone();
                if let (Some(_), Some(local_rope_block)) =
                    (sliding_window_sizes[layer_index], local_rope.clone())
                {
                    rope = local_rope_block;
                }

                LayerExecutables::new(
                    &mtl_context,
                    &decoder_config.layer_config,
                    compilation_config.clone(),
                    layer_index,
                    decoder_config.model_dim,
                    decoder_config.hidden_dim,
                    decoder_config.num_heads,
                    decoder_config.head_dim,
                    decoder_config.num_groups,
                    decoder_config.attention_scale,
                    &decoder_weight_loader
                        .subtree(&format!("layers.{}", layer_index))
                        .unwrap(),
                    attention_executable_provider.clone(),
                    rope,
                    kernels_config.clone(),
                )
            })
            .collect::<Vec<_>>();

        let norm_block: Box<dyn EncodableWithState> =
            if kernels_config.use_rms_norm {
                Box::new(
                    RMSNormKernelEncodable::new(
                        &mtl_context,
                        intermediate_data_type,
                        decoder_config.output_norm_config.clone(),
                        ArrayId::Main,
                        ArrayId::Main,
                        &decoder_weight_loader.subtree("output_norm").unwrap(),
                    )
                    .expect("Failed to create output RMS norm kernel"),
                )
            } else {
                Box::new(rms_norm_block(
                    &decoder_config.output_norm_config,
                    decoder_config.model_dim,
                    &mtl_context,
                    &decoder_weight_loader.subtree("output_norm").unwrap(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &compilation_config.descriptor_general,
                ))
            };

        Self {
            embed: embedding_blocks.embed,
            embedding: embedding_blocks.embedding,
            layers: layers.into_boxed_slice(),
            norm: norm_block,
            readout: embedding_blocks.readout,
            global_rope,
            local_rope,
        }
    }

    fn create_rope_block(
        mtl_context: &MTLContext,
        decoder_config: &DecoderConfig,
        decoder_weight_loader: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
        kernels_config: &KernelsConfig,
        rope_name: String,
        rope_config: &RoPEConfig,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableWithState>> {
        let intermediate_data_type: DataType = decoder_config
            .layer_config
            .attention_config
            .qkv_projection_config
            .activation_precision()
            .into();
        let kernel_data_type: KernelDataType = intermediate_data_type.into();

        let rotation: Box<dyn EncodableWithState>;
        if kernels_config.use_rope {
            rotation = Box::new(
                RopeKernelEncodable::new(
                    mtl_context,
                    kernel_data_type,
                    rope_type,
                )
                .expect("Failed to create RopeKernelEncodable"),
            );
        } else {
            let rotation_executable = rotation_executable(
                rope_name,
                rope_config,
                decoder_config.head_dim,
                decoder_config.num_groups,
                decoder_config.num_heads,
                decoder_config.context_length,
                &mtl_context,
                decoder_weight_loader,
                &compilation_config.descriptor_general,
            );
            rotation = Box::new(rotation_block(rotation_executable));
        }
        return Rc::new(rotation);
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
