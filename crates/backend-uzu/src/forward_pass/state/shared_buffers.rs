use super::RopeBuffers;
use crate::{
    backends::common::Backend,
    config::{decoder::DecoderConfig, rope::AnyRoPEConfig},
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
    session::types::Error,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerRopeKind {
    NoKernel,
    Indexed(usize),
}

pub struct SharedBuffers<B: Backend> {
    pub rope_buffers: Box<[RopeBuffers<B>]>,
    layer_rope_kinds: Box<[LayerRopeKind]>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let tf = &decoder_config.transformer_config;

        let mut configs = Vec::<(AnyRoPEConfig, usize)>::new();
        let layer_rope_kinds: Box<[LayerRopeKind]> = tf
            .layer_configs
            .iter()
            .map(|layer_config| {
                let Some(attention_config) = layer_config.mixer_config.as_attention() else {
                    return LayerRopeKind::NoKernel;
                };
                let Some(rope_config) = &layer_config.rope_config else {
                    return LayerRopeKind::NoKernel;
                };
                let head_dim = rope_config.head_dim().unwrap_or(attention_config.head_dim);
                let index = configs
                    .iter()
                    .position(|(existing_config, existing_head_dim)| {
                        existing_config == rope_config && *existing_head_dim == head_dim
                    })
                    .unwrap_or_else(|| {
                        configs.push((rope_config.clone(), head_dim));
                        configs.len() - 1
                    });
                LayerRopeKind::Indexed(index)
            })
            .collect();

        let rope_buffers: Box<[RopeBuffers<B>]> = configs
            .iter()
            .map(|(config, head_dim)| {
                RopeBuffers::new(context, *config.max_sequence_length(), *head_dim, model_shape.rope_data_type)
            })
            .collect();

        Self {
            rope_buffers,
            layer_rope_kinds,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(), Error> {
        let transformer_tree =
            parameter_tree.subtree("transformer").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        self.update_data_from_transformer_tree(&transformer_tree)?;
        Ok(())
    }

    pub fn update_data_from_transformer_tree(
        &mut self,
        transformer_tree: &ParameterTree<B>,
    ) -> Result<(), Error> {
        for (rope_index, rope_buffers) in self.rope_buffers.iter_mut().enumerate() {
            rope_buffers.update_data(transformer_tree, rope_index)?;
        }
        Ok(())
    }

    pub fn rope_buffers_for_layer(
        &self,
        layer_index: usize,
    ) -> Option<&RopeBuffers<B>> {
        match self.layer_rope_kinds[layer_index] {
            LayerRopeKind::NoKernel => None,
            LayerRopeKind::Indexed(index) => Some(&self.rope_buffers[index]),
        }
    }
}
