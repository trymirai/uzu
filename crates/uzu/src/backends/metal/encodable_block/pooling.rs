//! Pooling encodable for sequence-level aggregation.

use crate::backends::metal::{
    CommandBufferRef, ComputeCommandEncoderRef, MTLBlitCommandEncoder,
    MTLCommandBuffer, MTLCommandEncoder,
};

use super::{EncodableBlock, EncodingParameters};
#[cfg(feature = "tracing")]
use crate::Array;
use crate::{
    backends::metal::{
        forward_pass::{ArrayId, ForwardPassState},
        kernel::PoolingKernel,
    },
    config::PoolingType,
};

pub struct Pooling {
    pooling_kernel: PoolingKernel,
    pooling_type: PoolingType,
    model_dim: usize,
}

impl Pooling {
    pub fn new(
        pooling_kernel: PoolingKernel,
        pooling_type: PoolingType,
        model_dim: usize,
    ) -> Self {
        Self {
            pooling_kernel,
            pooling_type,
            model_dim,
        }
    }
}

impl EncodableBlock for Pooling {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: CommandBufferRef<'_>,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        #[cfg(feature = "tracing")]
        {
            let batch_size = 1;
            let data_type = {
                Array::data_type(&*state.arrays(&[ArrayId::Main])[0].borrow())
            };

            let arrays = state.arrays(&[ArrayId::ClassifierPooling]);
            let mut pooling_array = arrays[0].borrow_mut();
            let output_buffer =
                unsafe { pooling_array.mtl_buffer().to_owned() };
            drop(pooling_array);

            let traces_rc = state.traces().clone();
            let traces_ref = traces_rc.borrow();
            let mut trace_arr = traces_ref.output_pooling().borrow_mut();
            let dst_buf = unsafe { trace_arr.mtl_buffer().to_owned() };
            drop(trace_arr);
            drop(traces_ref);

            let blit = command_buffer
                .new_blit_command_encoder()
                .expect("Failed to create blit command encoder");
            blit.copy_from_buffer(
                &output_buffer,
                0,
                &dst_buf,
                0,
                (batch_size * self.model_dim * data_type.size_in_bytes())
                    as u64,
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
        state: &mut ForwardPassState,
        encoder: ComputeCommandEncoderRef<'_>,
        _parameters: &EncodingParameters,
    ) {
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();

        let arrays = state.arrays(&[ArrayId::Main, ArrayId::ClassifierPooling]);
        let mut main_array = arrays[0].borrow_mut();
        let input_buffer = unsafe { main_array.mtl_buffer() };

        let mut pooling_array = arrays[1].borrow_mut();
        let output_buffer = unsafe { pooling_array.mtl_buffer().to_owned() };

        let result = match self.pooling_type {
            PoolingType::Cls => self.pooling_kernel.encode_cls(
                encoder,
                input_buffer,
                &output_buffer,
                batch_size as i32,
                seq_len as i32,
                self.model_dim as i32,
            ),
            PoolingType::Mean => self.pooling_kernel.encode_mean(
                encoder,
                input_buffer,
                &output_buffer,
                batch_size as i32,
                seq_len as i32,
                self.model_dim as i32,
            ),
        };

        if result.is_err() {
            panic!("Failed to encode pooling kernel");
        }
    }
}
