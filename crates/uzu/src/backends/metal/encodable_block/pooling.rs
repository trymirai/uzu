//! Pooling encodable for sequence-level aggregation.

use super::{EncodableBlock, Metal};
#[cfg(feature = "tracing")]
use crate::backends::metal::MTLBlitCommandEncoder;
use crate::{
    backends::{
        common::kernel::{PoolingClsKernel, PoolingMeanKernel},
        metal::{
            KernelDataType, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
            MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
            Retained,
            kernel::dsl::{PoolingClsMetalKernel, PoolingMeanMetalKernel},
        },
    },
    config::PoolingType,
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
};

enum PoolingKernel {
    Cls(PoolingClsMetalKernel),
    Mean(PoolingMeanMetalKernel),
}

impl PoolingKernel {
    fn encode(
        &self,
        input: &Retained<ProtocolObject<dyn MTLBuffer>>,
        output: &Retained<ProtocolObject<dyn MTLBuffer>>,
        seq_len: u32,
        hidden_dim: u32,
        batch_size: u32,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        match self {
            Self::Cls(k) => k.encode(
                input, output, seq_len, hidden_dim, batch_size, encoder,
            ),
            Self::Mean(k) => k.encode(
                input, output, seq_len, hidden_dim, batch_size, encoder,
            ),
        }
    }
}

pub struct Pooling {
    pooling_kernel: PoolingKernel,
    model_dim: usize,
}

impl Pooling {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        pooling_type: PoolingType,
        model_dim: usize,
    ) -> Result<Self, MTLError> {
        let pooling_kernel = match pooling_type {
            PoolingType::Cls => PoolingKernel::Cls(PoolingClsMetalKernel::new(
                context,
                data_type.into(),
            )?),
            PoolingType::Mean => PoolingKernel::Mean(
                PoolingMeanMetalKernel::new(context, data_type.into())?,
            ),
        };
        Ok(Self {
            pooling_kernel,
            model_dim,
        })
    }
}

impl EncodableBlock<Metal> for Pooling {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        #[cfg(feature = "tracing")]
        {
            let batch_size = 1;
            let data_type =
                { state.arrays(&[ArrayId::Main])[0].borrow().data_type() };

            let arrays = state.arrays(&[ArrayId::ClassifierPooling]);
            let pooling_array = arrays[0].borrow();
            let output_buffer = pooling_array.buffer();

            let traces_rc = state.traces().clone();
            let traces_ref = traces_rc.borrow();
            let trace_arr = traces_ref.output_pooling().borrow();
            let dst_buf = trace_arr.buffer();

            let blit = command_buffer
                .new_blit_command_encoder()
                .expect("Failed to create blit command encoder");
            blit.copy_buffer_to_buffer(
                output_buffer,
                0,
                dst_buf,
                0,
                batch_size * self.model_dim * data_type.size_in_bytes(),
            );
            blit.end_encoding();
        }

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters<Metal>,
    ) {
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();

        let arrays = state.arrays(&[ArrayId::Main, ArrayId::ClassifierPooling]);
        let main_array = arrays[0].borrow_mut();
        let input_buffer = main_array.buffer();

        let pooling_array = arrays[1].borrow_mut();
        let output_buffer = pooling_array.buffer().to_owned();

        self.pooling_kernel.encode(
            input_buffer,
            &output_buffer,
            seq_len as u32,
            self.model_dim as u32,
            batch_size,
            encoder,
        );
    }
}
