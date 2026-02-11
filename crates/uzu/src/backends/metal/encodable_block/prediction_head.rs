//! Prediction head encodable for classification output.

use super::{EncodableBlock, Metal};
#[cfg(feature = "tracing")]
use crate::backends::metal::MTLBlitCommandEncoder;
#[cfg(feature = "tracing")]
use crate::forward_pass::state::ArrayId;
use crate::{
    backends::metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, ProtocolObject, Retained},
    encodable_block::EncodingParameters,
    forward_pass::state::ForwardPassState,
};

pub struct ClassifierPredictionHead {
    dense: Box<dyn EncodableBlock<Metal>>,
    activation: Box<dyn EncodableBlock<Metal>>,
    norm: Box<dyn EncodableBlock<Metal>>,
    readout: Box<dyn EncodableBlock<Metal>>,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    num_labels: usize,
}

impl ClassifierPredictionHead {
    pub fn new(
        dense: Box<dyn EncodableBlock<Metal>>,
        activation: Box<dyn EncodableBlock<Metal>>,
        norm: Box<dyn EncodableBlock<Metal>>,
        readout: Box<dyn EncodableBlock<Metal>>,
        num_labels: usize,
    ) -> Self {
        Self {
            dense,
            activation,
            norm,
            readout,
            num_labels,
        }
    }
}

impl EncodableBlock<Metal> for ClassifierPredictionHead {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        if self.supports_shared_encoder() {
            let encoder =
                command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
            self.encode_with_shared_encoder(state, parameters, &encoder);
            encoder.end_encoding();
        } else {
            self.dense.encode(state, parameters, command_buffer);
            self.activation.encode(state, parameters, command_buffer);
            self.norm.encode(state, parameters, command_buffer);
            self.readout.encode(state, parameters, command_buffer);
        }

        #[cfg(feature = "tracing")]
        {
            let traces_rc = state.traces().clone();
            let logits_arrays = state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
            let logits_array_ref = logits_arrays[0].borrow();
            let linear_output_buffer = logits_array_ref.buffer();
            let data_type = logits_array_ref.data_type();
            let batch_size = logits_array_ref.shape()[0];

            let traces_ref = traces_rc.borrow();
            let trace_logits = traces_ref.logits.borrow();
            let dst_trace_buf = trace_logits.buffer();

            let copy_size_bytes = batch_size * self.num_labels * data_type.size_in_bytes();

            let blit = command_buffer.new_blit_command_encoder().expect("Failed to create blit command encoder");
            blit.copy_buffer_to_buffer(linear_output_buffer, 0, dst_trace_buf, 0, copy_size_bytes);
            blit.end_encoding();
        }

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.dense.supports_shared_encoder()
            && self.activation.supports_shared_encoder()
            && self.norm.supports_shared_encoder()
            && self.readout.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        self.dense.encode_with_shared_encoder(state, parameters, encoder);
        self.activation.encode_with_shared_encoder(state, parameters, encoder);
        self.norm.encode_with_shared_encoder(state, parameters, encoder);
        self.readout.encode_with_shared_encoder(state, parameters, encoder);
    }
}
