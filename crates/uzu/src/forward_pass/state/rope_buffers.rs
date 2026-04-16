use crate::{
    backends::common::{Allocation, Backend},
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
};

pub struct RopeBuffers<B: Backend> {
    /// [rope_max_sequence_length, rope_dim]
    pub cosines: Allocation<B>,
    /// [rope_max_sequence_length, rope_dim]
    pub sines: Allocation<B>,
}

impl<B: Backend> RopeBuffers<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
    ) -> Self {
        let rope_dim = model_shape.rope_dim();
        let rope_max_sequence_length = model_shape.context_length();

        Self {
            cosines: crate::backends::common::allocation_helpers::create_allocation(
                context,
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
            ),
            sines: crate::backends::common::allocation_helpers::create_allocation(
                context,
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
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

        self.cosines = rope_tree.leaf_allocation("cosines").unwrap();
        self.sines = rope_tree.leaf_allocation("sines").unwrap();
    }
}
