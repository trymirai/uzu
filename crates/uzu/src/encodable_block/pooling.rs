//! Pooling encodable for sequence-level aggregation.

use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
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
        input: &B::Buffer,
        output: &mut B::Buffer,
        seq_len: u32,
        hidden_dim: u32,
        batch_size: u32,
        encoder: &mut Encoder<B>,
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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();

        let main_array = state.array(ArrayId::Main);
        let pooling_array = state.array(ArrayId::ClassifierPooling);

        self.pooling_kernel.encode(
            main_array.buffer().borrow().deref(),
            pooling_array.buffer().borrow_mut().deref_mut(),
            seq_len as u32,
            self.model_dim as u32,
            batch_size,
            encoder,
        );

        #[cfg(feature = "tracing")]
        {
            let output_pooling_trace = state.traces().borrow().output_pooling().clone();
            state.encode_copy_array(encoder, ArrayId::ClassifierPooling, output_pooling_trace);
        }
        Ok(())
    }
}
