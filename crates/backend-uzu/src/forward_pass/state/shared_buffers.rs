use half::{bf16, f16};

use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    config::DecoderConfig,
    parameters::ParameterTree,
};

pub struct SharedBuffers<B: Backend> {
    pub attention_sinks: Option<Vec<Array<B>>>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
    ) -> Self {
        let attention_sinks = decoder_config.layer_config.attention_config().is_some_and(|c| c.has_sinks).then(|| {
            let num_heads = decoder_config.num_heads;
            (0..decoder_config.num_layers)
                .map(|_| {
                    context.create_array_uninitialized(&[num_heads], DataType::F32, "shared_buffers_attention_sinks")
                })
                .collect()
        });

        Self {
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
    ) {
        let transformer_tree = parameter_tree.subtree("transformer").expect("transformer subtree not found");
        self.update_data_from_transformer_tree(&transformer_tree);
    }

    pub fn update_data_from_transformer_tree(
        &mut self,
        transformer_tree: &ParameterTree<B::Context>,
    ) {
        if let Some(sinks_vec) = &mut self.attention_sinks {
            for (layer_idx, sink_cell) in sinks_vec.iter_mut().enumerate() {
                let layer_tree = transformer_tree.subtree(&format!("layers.{}", layer_idx)).unwrap();
                let attn_tree = layer_tree.subtree("mixer").unwrap();
                let sinks_arr = attn_tree.leaf_array("sinks").unwrap();
                let dst_slice = sink_cell.as_slice_mut::<f32>();

                match sinks_arr.data_type() {
                    DataType::F32 => {
                        let src = sinks_arr.as_slice::<f32>();
                        dst_slice.copy_from_slice(src);
                    },
                    DataType::BF16 => {
                        let src = sinks_arr.as_slice::<bf16>();
                        for (dst_val, src_val) in dst_slice.iter_mut().zip(src.iter()) {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    DataType::F16 => {
                        let src = sinks_arr.as_slice::<f16>();
                        for (dst_val, src_val) in dst_slice.iter_mut().zip(src.iter()) {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    other => {
                        panic!("Unsupported attention sink data type: {:?}", other);
                    },
                }
            }
        }
    }
}
