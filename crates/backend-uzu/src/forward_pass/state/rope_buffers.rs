use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, Backend},
    parameters::ParameterTree,
    session::types::Error,
};

pub struct RopeBuffers<B: Backend> {
    /// [rope_max_sequence_length, rope_dim]
    pub cosines: Allocation<B>,
    /// [rope_max_sequence_length, rope_dim]
    pub sines: Allocation<B>,
    max_sequence_length: usize,
    dim: usize,
}

impl<B: Backend> RopeBuffers<B> {
    pub fn new(
        context: &B::Context,
        max_sequence_length: usize,
        head_dim: usize,
        data_type: DataType,
    ) -> Self {
        let shape = [max_sequence_length, head_dim];
        Self {
            cosines: context.create_array_uninitialized(&shape, data_type).into_allocation(),
            sines: context.create_array_uninitialized(&shape, data_type).into_allocation(),
            max_sequence_length,
            dim: head_dim,
        }
    }

    pub fn passthrough(
        context: &B::Context,
        data_type: DataType,
    ) -> Self {
        Self {
            cosines: context.create_array_zeros(&[1, 1], data_type).into_allocation(),
            sines: context.create_array_zeros(&[1, 1], data_type).into_allocation(),
            max_sequence_length: 1,
            dim: 0,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
        rope_index: usize,
    ) -> Result<(), Error> {
        let rope_tree =
            parameter_tree.subtree(&format!("ropes.{}", rope_index)).map_err(|_| Error::UnableToLoadWeights)?;
        self.cosines = rope_tree.leaf_allocation("cosines").map_err(|_| Error::UnableToLoadWeights)?;
        self.sines = rope_tree.leaf_allocation("sines").map_err(|_| Error::UnableToLoadWeights)?;
        Ok(())
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}
