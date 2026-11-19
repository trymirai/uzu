use crate::{
    backends::Backend,
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub struct InterpolatedPositionEmbedding<B: Backend> {
    table: B::GlobalGpuBuffer,
    side: usize,
    model_dim: usize,
    data_type: DataType,
}

impl<B: Backend> InterpolatedPositionEmbedding<B> {
    pub async fn new(
        data_type: DataType,
        num_positions: u32,
        model_dim: u32,
        tree: &ParameterTree<'_, B>,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let side = num_positions.isqrt();
        assert_eq!(side * side, num_positions, "position table size {num_positions} isn't a square");
        Ok(Self {
            table: tree.read_buffer("table", &[num_positions, model_dim], data_type).await?,
            side: side as usize,
            model_dim: model_dim as usize,
            data_type,
        })
    }
}
