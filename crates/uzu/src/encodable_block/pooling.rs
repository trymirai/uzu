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
    /// No-op passthrough: the prediction head runs on every token.
    None,
}

impl<B: Backend> PoolingKernel<B> {
    fn encode(
        &self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        seq_len: u32,
        hidden_dim: u32,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) {
        match self {
            Self::Cls(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_dim, encoder),
            Self::Mean(kernel) => kernel.encode(input, output, seq_len, hidden_dim, batch_dim, encoder),
            Self::None => {
                // Per-token classifier: nothing to do. Consumers read directly
                // from ArrayId::Main (wired at ClassifierContext::new time).
                let _ = (input, output, seq_len, hidden_dim, batch_dim, encoder);
            },
        }
    }
}

pub struct Pooling<B: Backend> {
    pooling_kernel: PoolingKernel<B>,
    model_dim: usize,
    pooling_type: PoolingType,
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
            PoolingType::None => PoolingKernel::None,
        };
        Ok(Self {
            pooling_kernel,
            model_dim,
            pooling_type,
        })
    }

    /// Whether this pooling layer collapses the sequence down to a single row.
    /// `None` keeps the full sequence (per-token classification).
    #[allow(dead_code)]
    pub fn collapses_sequence(&self) -> bool {
        !matches!(self.pooling_type, PoolingType::None)
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if matches!(self.pooling_type, PoolingType::None) {
            // Per-token path: leave Main alone, leave active_row_count at
            // suffix_length so downstream Linear kernels process every token.
            return Ok(());
        }

        let batch_dim = 1;
        let seq_len = state.aux_buffers_suffix_length();

        let main_array = state.array(ArrayId::Main);
        let pooling_array = state.array(ArrayId::ClassifierPooling);

        self.pooling_kernel.encode(
            main_array.buffer().borrow().deref(),
            pooling_array.buffer().borrow_mut().deref_mut(),
            seq_len as u32,
            self.model_dim as u32,
            batch_dim,
            encoder,
        );

        #[cfg(feature = "tracing")]
        {
            let output_pooling_trace = state.traces().borrow().output_pooling().clone();
            state.encode_copy_array(encoder, ArrayId::ClassifierPooling, output_pooling_trace);
        }

        state.set_active_row_count(pooling_array.shape()[0]);
        Ok(())
    }
}
