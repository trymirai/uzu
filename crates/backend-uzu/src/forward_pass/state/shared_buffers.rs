use half::{bf16, f16};

use super::RopeBuffers;
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, AsBufferRangeMut, Backend, DenseBuffer},
    config::{DecoderConfig, RoPEConfig},
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
    session::types::Error,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerRopeKind {
    NoKernel,
    // TODO: split rope kernel into QKV-unpack + rotate so this variant can be removed.
    PassThrough,
    Indexed(usize),
}

pub struct SharedBuffers<B: Backend> {
    pub rope_buffers: Box<[RopeBuffers<B>]>,
    passthrough_rope: RopeBuffers<B>,
    layer_rope_kinds: Box<[LayerRopeKind]>,
    pub attention_sinks: Box<[Option<Allocation<B>>]>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let tf = &decoder_config.transformer_config;

        let mut configs = Vec::<RoPEConfig>::new();
        let layer_rope_kinds: Box<[LayerRopeKind]> = tf
            .layer_configs
            .iter()
            .map(|layer_config| {
                if layer_config.mixer_config.as_attention().is_none() {
                    return LayerRopeKind::NoKernel;
                }
                let Some(rope_config) = &layer_config.rope_config else {
                    return LayerRopeKind::PassThrough;
                };
                let index = configs.iter().position(|existing| existing == rope_config).unwrap_or_else(|| {
                    configs.push(rope_config.clone());
                    configs.len() - 1
                });
                LayerRopeKind::Indexed(index)
            })
            .collect();

        let rope_buffers: Box<[RopeBuffers<B>]> = configs
            .iter()
            .map(|config| {
                let common = config.common();
                RopeBuffers::new(context, common.max_sequence_length, common.head_dim, common.precision.into())
            })
            .collect();

        let passthrough_rope = RopeBuffers::passthrough(context, model_shape.activation_data_type());

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
            rope_buffers,
            passthrough_rope,
            layer_rope_kinds,
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<(), Error> {
        let transformer_tree = parameter_tree.subtree("transformer").map_err(|_| Error::UnableToLoadWeights)?;
        self.update_data_from_transformer_tree(&transformer_tree)?;
        Ok(())
    }

    pub fn update_data_from_transformer_tree(
        &mut self,
        transformer_tree: &ParameterTree<B::Context>,
    ) -> Result<(), Error> {
        for (rope_index, rope_buffers) in self.rope_buffers.iter_mut().enumerate() {
            rope_buffers.update_data(transformer_tree, rope_index)?;
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
        Ok(())
    }

    pub fn rope_buffers_for_layer(
        &self,
        layer_index: usize,
    ) -> Option<&RopeBuffers<B>> {
        match self.layer_rope_kinds[layer_index] {
            LayerRopeKind::NoKernel => None,
            LayerRopeKind::PassThrough => Some(&self.passthrough_rope),
            LayerRopeKind::Indexed(index) => Some(&self.rope_buffers[index]),
        }
    }

    pub fn attention_sinks(
        &self,
        layer_index: usize,
    ) -> Option<&Allocation<B>> {
        self.attention_sinks.get(layer_index)?.as_ref()
    }
}
