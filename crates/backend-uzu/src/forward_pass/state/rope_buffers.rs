use crate::{
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    config::RoPEConfig,
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
};

pub struct RopeBuffers<B: Backend> {
    /// [rope_max_sequence_length, rope_dim]
    pub cosines: Array<B>,
    /// [rope_max_sequence_length, rope_dim]
    pub sines: Array<B>,
}

impl<B: Backend> RopeBuffers<B> {
    pub fn new(
        context: &B::Context,
        rope_config: &RoPEConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let common = rope_config.common();
        let rope_dim = common.partial_rotary_dim.or(common.head_dim).unwrap_or_else(|| model_shape.rope_dim());
        let rope_max_sequence_length = common.max_sequence_length;

        Self {
            cosines: context.create_array_uninitialized(
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
                "rope_buffers_cosines",
            ),
            sines: context.create_array_uninitialized(
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
                "rope_buffers_sines",
            ),
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
        rope_name: &str,
    ) {
        let Ok(rope_tree) = parameter_tree.subtree(rope_name) else {
            return;
        };

        let cosines_view = rope_tree.leaf_array("cosines").unwrap();
        self.cosines.copy_from_array(&cosines_view);

        let sines_view = rope_tree.leaf_array("sines").unwrap();
        self.sines.copy_from_array(&sines_view);
    }
}
