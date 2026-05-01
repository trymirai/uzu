use half::{bf16, f16};

use super::{RopeBuffers, RopeType};
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, Backend, Buffer},
    config::DecoderConfig,
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
};

pub struct SharedBuffers<B: Backend> {
    pub global_rope: Option<RopeBuffers<B>>,
    pub local_rope: Option<RopeBuffers<B>>,
    pub attention_sinks: Option<Vec<Allocation<B>>>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let global_rope = decoder_config.global_rope_config.is_some().then(|| RopeBuffers::new(context, model_shape));

        let local_rope = decoder_config.local_rope_config.is_some().then(|| RopeBuffers::new(context, model_shape));

        let attention_sinks = decoder_config.layer_config.attention_config().is_some_and(|c| c.has_sinks).then(|| {
            let num_heads = decoder_config.num_heads;
            (0..decoder_config.num_layers)
                .map(|_| {
                    context.create_array_uninitialized(&[num_heads], DataType::F32, "attention_sinks").into_allocation()
                })
                .collect()
        });

        Self {
            global_rope,
            local_rope,
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
    ) {
        let transformer_tree = parameter_tree.subtree("transformer").expect("transformer subtree not found");

        if let Some(global_rope) = &mut self.global_rope {
            global_rope.update_data(&transformer_tree, "global_rope");
        }
        if let Some(local_rope) = &mut self.local_rope {
            local_rope.update_data(&transformer_tree, "local_rope");
        }

        if let Some(sinks_vec) = &mut self.attention_sinks {
            for (layer_idx, sink_cell) in sinks_vec.iter_mut().enumerate() {
                let layer_tree = transformer_tree.subtree(&format!("layers.{}", layer_idx)).unwrap();
                let attn_tree = layer_tree.subtree("mixer").unwrap();
                let sinks_arr = attn_tree.leaf_array("sinks").unwrap();
                let dst_slice = unsafe {
                    let (buffer, range) = sink_cell.as_buffer_range();
                    std::slice::from_raw_parts_mut(
                        (buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start) as *mut f32,
                        range.len() / std::mem::size_of::<f32>(),
                    )
                };

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

    pub fn rope_cosines(
        &self,
        rope_type: RopeType,
    ) -> Option<&Allocation<B>> {
        match rope_type {
            RopeType::Global => self.global_rope.as_ref().map(|rope| &rope.cosines),
            RopeType::Local => self.local_rope.as_ref().map(|rope| &rope.cosines),
        }
    }

    pub fn rope_sines(
        &self,
        rope_type: RopeType,
    ) -> Option<&Allocation<B>> {
        match rope_type {
            RopeType::Global => self.global_rope.as_ref().map(|rope| &rope.sines),
            RopeType::Local => self.local_rope.as_ref().map(|rope| &rope.sines),
        }
    }

    pub fn attention_sinks(
        &self,
        layer_index: usize,
    ) -> Option<&Allocation<B>> {
        self.attention_sinks.as_ref().map(|sinks| &sinks[layer_index])
    }
}
