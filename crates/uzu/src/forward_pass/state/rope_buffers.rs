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
            cosines: super::allocation_helpers::create_allocation(
                context,
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
            ),
            sines: super::allocation_helpers::create_allocation(
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

        let cosines_view = rope_tree.leaf_array("cosines").unwrap();
        super::allocation_helpers::copy_array_to_allocation(&mut self.cosines, &cosines_view);

        let sines_view = rope_tree.leaf_array("sines").unwrap();
        super::allocation_helpers::copy_array_to_allocation(&mut self.sines, &sines_view);
    }
}
