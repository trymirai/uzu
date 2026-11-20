use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{BufferMut, CommandBuffer, CommandBufferEncoding, Kernels, kernel::VisionPositionEmbeddingKernel},
    },
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum VisionPositionEmbeddingNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Position table size {0} isn't a square")]
    NonSquareTable(u32),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Model dim {0} isn't a multiple of 4")]
    UnsupportedModelDim(u32),
}

pub struct VisionPositionEmbedding<B: Backend> {
    table: B::GlobalGpuBuffer,
    side: u32,
    model_dim: u32,
    kernel: <B::Kernels as Kernels>::VisionPositionEmbeddingKernel,
}

impl<B: Backend> VisionPositionEmbedding<B> {
    pub async fn new(
        data_type: DataType,
        num_positions: u32,
        model_dim: u32,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, VisionPositionEmbeddingNewError<B>> {
        let side = num_positions.isqrt();
        if side * side != num_positions {
            return Err(VisionPositionEmbeddingNewError::NonSquareTable(num_positions));
        }

        // The GPU kernels add 4 consecutive channels per thread
        if !model_dim.is_multiple_of(4) {
            return Err(VisionPositionEmbeddingNewError::UnsupportedModelDim(model_dim));
        }

        let kernel = <B::Kernels as Kernels>::VisionPositionEmbeddingKernel::new(context, data_type)
            .map_err(VisionPositionEmbeddingNewError::Backend)?;
        Ok(Self {
            table: tree.read_buffer("table", &[num_positions, model_dim], data_type).await?,
            side,
            model_dim,
            kernel,
        })
    }

    /// Adds the position embedding, resized to the `grid_width` x `grid_height` patch grid, to `stream` in place.
    pub fn encode(
        &self,
        stream: impl BufferMut<Backend = B>,
        grid_width: u32,
        grid_height: u32,
        merge_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let command_buffer = &mut *command_buffer.debug_group("vision embedding encode");
        self.kernel.encode(
            &self.table,
            stream,
            self.side,
            grid_width,
            grid_height,
            merge_size,
            self.model_dim,
            grid_width * grid_height * self.model_dim,
            command_buffer,
        );
    }
}
