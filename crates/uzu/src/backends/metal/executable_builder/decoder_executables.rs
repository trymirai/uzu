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
    config::decoder::DecoderConfig,
    parameters::ParameterTree,
};

pub struct DecoderExecutables {
    pub embed: Box<dyn EncodableWithState>,
    pub layers: Box<[LayerExecutables]>,
    pub norm: Box<dyn EncodableWithState>,
    pub readout: Box<dyn EncodableWithState>,
    pub global_rope: Rc<Box<dyn EncodableWithState>>,
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
        );

        let readout = readout_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
        );

        let global_rope = Self::create_rope_block(
            &mtl_context,
            &decoder_config,
            RopeType::Global,
        );

        let local_rope: Option<Rc<Box<dyn EncodableWithState>>>;
        if let Some(_) = &decoder_config.local_rope_config {
            local_rope = Some(Self::create_rope_block(
                &mtl_context,
                &decoder_config,
                RopeType::Local,
            ));
        } else {
            local_rope = None;
        }

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes = model_shape.sliding_window_length_per_layer;

        let intermediate_data_type: DataType = decoder_config
            .layer_config
            .attention_config
            .qkv_projection_config
            .activation_precision()
            .into();

        // STEP 1: Transform MOE expert weights once (initialization only)
        eprintln!("[DEBUG] Step 1: MOE initialization ENABLED, execution DISABLED");
        let shared_moe_weights = if let crate::config::MLPConfig::MixtureOfExperts(moe_config) = &decoder_config.layer_config.mlp_config {
            eprintln!("[Decoder] Detected MOE config, transforming expert weights once for all {} layers", decoder_config.num_layers);
            let first_layer_mlp_tree = decoder_weight_loader
                .subtree("layers.0")
                .unwrap()
                .subtree("mlp")
                .unwrap();
            
            Some(
                crate::backends::metal::kernel::moe::MoeBlockEncodable::transform_expert_weights_once(
                    &mtl_context,
                    moe_config,
                    decoder_config.model_dim,
                    decoder_config.hidden_dim,
                    &first_layer_mlp_tree,
                )
                .expect("Failed to transform MOE weights")
            )
        } else {
            None
        };

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
                    rope,
                    shared_moe_weights.clone(),
                )
            })
            .collect::<Vec<_>>();

        let norm_block: Box<dyn EncodableWithState> = Box::new(
            RMSNormKernelEncodable::new(
                &mtl_context,
                intermediate_data_type,
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
        decoder_config: &DecoderConfig,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableWithState>> {
        let intermediate_data_type: DataType = decoder_config
            .layer_config
            .attention_config
            .qkv_projection_config
            .activation_precision()
            .into();
        let kernel_data_type: KernelDataType = intermediate_data_type.into();

        let rotation: Box<dyn EncodableWithState> = Box::new(
            RopeKernelEncodable::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create RopeKernelEncodable"),
        );
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
        let t_total = std::time::Instant::now();
        
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
        
        eprintln!("[Decoder] Encoding complete: {:.3}ms", t_total.elapsed().as_secs_f64() * 1000.0);
    }
}
