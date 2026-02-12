//! Pooling encodable for sequence-level aggregation.

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer,
        kernel::{Kernels, PoolingClsKernel, PoolingMeanKernel},
    },
    config::PoolingType,
    forward_pass::state::{ArrayId, ForwardPassState},
};

enum PoolingKernel<B: Backend> {
    Cls(<B::Kernels as Kernels>::PoolingClsKernel),
    Mean(<B::Kernels as Kernels>::PoolingMeanKernel),
}

impl<B: Backend> PoolingKernel<B> {
    fn encode(
        &self,
        input: &B::NativeBuffer,
        output: &B::NativeBuffer,
        seq_len: u32,
        hidden_dim: u32,
        batch_size: u32,
        encoder: &B::ComputeEncoder,
    ) {
        match self {
            Self::Cls(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_size, encoder),
            Self::Mean(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_size, encoder),
        }
    }
}

pub struct Pooling<B: Backend> {
    pooling_kernel: PoolingKernel<B>,
    model_dim: usize,
}

impl<B: Backend> Pooling<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        pooling_type: PoolingType,
        model_dim: usize,
    ) -> Result<Self, B::Error> {
        let pooling_kernel = match pooling_type {
            PoolingType::Cls => PoolingKernel::Cls(<B::Kernels as Kernels>::PoolingClsKernel::new(context, data_type)?),
            PoolingType::Mean => {
                PoolingKernel::Mean(<B::Kernels as Kernels>::PoolingMeanKernel::new(context, data_type)?)
            },
        };
        Ok(Self {
            pooling_kernel,
            model_dim,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Pooling<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        command_buffer.with_compute_encoder(|encoder| self.encode_with_shared_encoder(state, parameters, encoder));

        #[cfg(feature = "tracing")]
        {
            let output_pooling_trace = state.traces().borrow().output_pooling().clone();
            state.encode_copy_array(command_buffer, ArrayId::ClassifierPooling, output_pooling_trace);
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();

        let arrays = state.arrays(&[ArrayId::Main, ArrayId::ClassifierPooling]);
        let main_array = arrays[0].borrow_mut();
        let pooling_array = arrays[1].borrow_mut();

        self.pooling_kernel.encode(
            main_array.buffer(),
            pooling_array.buffer(),
            seq_len as u32,
            self.model_dim as u32,
            batch_size,
            encoder,
        );
    }
}
