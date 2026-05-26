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
    data_type: DataType,
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
            data_type,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B>,
        rope_index: usize,
    ) -> Result<(), Error> {
        let rope_tree = parameter_tree
            .subtree(&format!("ropes.{}", rope_index))
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let cosines_leaf = rope_tree.leaf("cosines").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let cosines_leaf = cosines_leaf
            .validate(&[self.max_sequence_length, self.dim], self.data_type)
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        self.cosines = cosines_leaf.read_allocation().map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        let sines_leaf = rope_tree.leaf("sines").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let sines_leaf = sines_leaf
            .validate(&[self.max_sequence_length, self.dim], self.data_type)
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        self.sines = sines_leaf.read_allocation().map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        Ok(())
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}
