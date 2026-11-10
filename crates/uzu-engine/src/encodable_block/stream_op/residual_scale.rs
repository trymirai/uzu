use num_traits::AsPrimitive;

use crate::{
    backends::common::{
        Backend, CommandBuffer,
        kernel::{Kernels, TensorAddScaleKernel},
    },
    data_type::DataType,
    encodable_block::stream_op::StreamOpNewError,
    safetensors::ParameterTree,
    utils::dispatch_dtype::dispatch_dtype,
};

pub struct ResidualScale<B: Backend> {
    scale_kernel: <B::Kernels as Kernels>::TensorAddScaleKernel,
    scale: f32,
    model_dim: u32,
}

impl<B: Backend> ResidualScale<B> {
    pub async fn new(
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, StreamOpNewError<B>> {
        let scale: f32 =
            dispatch_dtype!(|(T: data_type)| parameter_tree.read_slice::<T>("scale", &[1]).await?[0].as_());
        let scale_kernel = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, false)
            .map_err(StreamOpNewError::Backend)?;

        Ok(Self {
            scale_kernel,
            scale,
            model_dim,
        })
    }

    /// Takes and returns an additive stream, `[tokens, model_dim]`.
    pub fn encode(
        &self,
        mut stream: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> B::ScratchBuffer {
        let elements = row_count * self.model_dim;
        self.scale_kernel.encode(
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            &mut stream,
            self.model_dim,
            elements,
            self.scale,
            command_buffer,
        );
        stream
    }
}
