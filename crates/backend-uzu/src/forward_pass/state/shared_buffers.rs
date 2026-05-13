use half::{bf16, f16};

use super::{RopeBuffers, RopeType};
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, AsBufferRangeMut, Backend, DenseBuffer},
    config::DecoderConfig,
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
};

pub struct SharedBuffers<B: Backend> {
    pub global_rope: Option<RopeBuffers<B>>,
    pub local_rope: Option<RopeBuffers<B>>,
    pub attention_sinks: Box<[Option<Allocation<B>>]>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let tf = &decoder_config.transformer_config;
        let global_rope = tf.global_rope_config.is_some().then(|| RopeBuffers::new(context, model_shape));
        let local_rope = tf.local_rope_config.is_some().then(|| RopeBuffers::new(context, model_shape));

        let attention_sinks = tf
            .layer_configs
            .iter()
            .map(|layer| {
                let attn = layer.mixer_config.as_attention()?;
                attn.has_sinks
                    .then(|| context.create_array_uninitialized(&[attn.num_heads], DataType::F32).into_allocation())
            })
            .collect();

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

        for (layer_idx, sink_cell) in self.attention_sinks.iter_mut().enumerate() {
            let Some(sink_cell) = sink_cell.as_mut() else {
                continue;
            };
            let layer_tree = transformer_tree.subtree(&format!("layers.{}", layer_idx)).unwrap();
            let attn_tree = layer_tree.subtree("mixer").unwrap();
            let sinks_arr = attn_tree.leaf_array("sinks").unwrap();
            let dst_slice = unsafe {
                let buffer_range = sink_cell.as_buffer_range_mut();
                let range = buffer_range.range();
                std::slice::from_raw_parts_mut(
                    (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start) as *mut f32,
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
        self.attention_sinks.get(layer_index)?.as_ref()
    }
}
