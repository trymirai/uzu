//! Pooling encodable for sequence-level aggregation.

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, PoolingClsKernel, PoolingMeanKernel},
    },
    config::PoolingType,
};

enum PoolingKernel<B: Backend> {
    Cls(<B::Kernels as Kernels>::PoolingClsKernel),
    Mean(<B::Kernels as Kernels>::PoolingMeanKernel),
}

impl<B: Backend> PoolingKernel<B> {
    fn encode(
        &self,
        input: &Allocation<B>,
        output: &mut Allocation<B>,
        seq_len: u32,
        hidden_dim: u32,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) {
        match self {
            Self::Cls(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_dim, encoder),
            Self::Mean(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_dim, encoder),
        }
    }
}

pub struct Pooling<B: Backend> {
    pooling_kernel: PoolingKernel<B>,
    model_dim: usize,
    data_type: DataType,
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
            data_type,
        })
    }

    pub fn encode(
        &self,
        seq_len: usize,
        main: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = 1;
        let mut pooling = encoder.allocate_scratch(size_for_shape(&[1, self.model_dim], self.data_type))?;

        self.pooling_kernel.encode(main, &mut pooling, seq_len as u32, self.model_dim as u32, batch_dim, encoder);
        Ok(pooling)
    }
}
