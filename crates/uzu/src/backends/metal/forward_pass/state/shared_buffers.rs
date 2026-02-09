use std::cell::RefCell;

use half::{bf16, f16};

use super::{super::ModelShape, RopeBuffers};
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::metal::{MTLContext, MetalArray},
    parameters::ParameterTree,
};

type ArrayCell = RefCell<MetalArray>;

pub struct MoeExpertWeights {
    pub w1: ArrayCell,
    pub w2: ArrayCell,
    pub w3: ArrayCell,
}

pub struct SharedBuffers {
    pub global_rope: Option<RopeBuffers>,
    pub local_rope: Option<RopeBuffers>,
    pub moe_expert_weights: Option<Vec<MoeExpertWeights>>,
    pub attention_sinks: Option<Vec<ArrayCell>>,
}

impl SharedBuffers {
    pub fn new(
        context: &MTLContext,
        decoder_config: &crate::config::DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let global_rope = if decoder_config.global_rope_config.is_some() {
            Some(RopeBuffers::new(context, model_shape))
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            Some(RopeBuffers::new(context, model_shape))
        } else {
            None
        };

        let moe_expert_weights = if matches!(
            decoder_config.layer_config.mlp_config,
            crate::config::MLPConfig::MixtureOfExperts(_)
        ) {
            Some(Vec::new())
        } else {
            None
        };

        let attention_sinks = if let Some(attention_config) =
            decoder_config.layer_config.attention_config()
        {
            if attention_config.has_sinks {
                let num_heads = decoder_config.num_heads;
                Some(
                    (0..decoder_config.num_layers)
                        .map(|_| {
                            RefCell::new(context.create_array_uninitialized(
                                &[num_heads],
                                DataType::F32,
                                "shared_buffers_attention_sinks",
                            ))
                        })
                        .collect(),
                )
            } else {
                None
            }
        } else {
            None
        };

        Self {
            global_rope,
            local_rope,
            moe_expert_weights,
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<MTLContext>,
    ) {
        let transformer_tree = parameter_tree
            .subtree("transformer")
            .expect("transformer subtree not found");

        if let Some(global_rope) = &mut self.global_rope {
            global_rope.update_data(&transformer_tree, "global_rope");
        }
        if let Some(local_rope) = &mut self.local_rope {
            local_rope.update_data(&transformer_tree, "local_rope");
        }

        if let Some(sinks_vec) = &mut self.attention_sinks {
            for (layer_idx, sink_cell) in sinks_vec.iter_mut().enumerate() {
                let layer_tree = transformer_tree
                    .subtree(&format!("layers.{}", layer_idx))
                    .unwrap();
                let attn_tree = layer_tree.subtree("mixer").unwrap();
                let sinks_arr = attn_tree.leaf("sinks").unwrap();
                let mut dst = sink_cell.borrow_mut();
                let dst_slice = dst.as_slice_mut::<f32>();

                match sinks_arr.data_type() {
                    DataType::F32 => {
                        let src = sinks_arr.as_slice::<f32>();
                        dst_slice.copy_from_slice(src);
                    },
                    DataType::BF16 => {
                        let src = sinks_arr.as_slice::<bf16>();
                        for (dst_val, src_val) in
                            dst_slice.iter_mut().zip(src.iter())
                        {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    DataType::F16 => {
                        let src = sinks_arr.as_slice::<f16>();
                        for (dst_val, src_val) in
                            dst_slice.iter_mut().zip(src.iter())
                        {
                            *dst_val = f32::from(*src_val);
                        }
                    },
                    other => {
                        panic!(
                            "Unsupported attention sink data type: {:?}",
                            other
                        );
                    },
                }
            }
        }
    }
}
