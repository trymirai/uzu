//! Prediction head encodable for classification output.

#[cfg(feature = "tracing")]
use crate::forward_pass::state::ArrayId;
use crate::{
    backends::common::{Backend, CommandBuffer},
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::ForwardPassState,
};

pub struct ClassifierPredictionHead<B: Backend> {
    dense: Box<dyn EncodableBlock<B>>,
    activation: Box<dyn EncodableBlock<B>>,
    norm: Box<dyn EncodableBlock<B>>,
    readout: Box<dyn EncodableBlock<B>>,
    #[allow(dead_code)]
    num_labels: usize,
}

impl<B: Backend> ClassifierPredictionHead<B> {
    pub fn new(
        dense: Box<dyn EncodableBlock<B>>,
        activation: Box<dyn EncodableBlock<B>>,
        norm: Box<dyn EncodableBlock<B>>,
        readout: Box<dyn EncodableBlock<B>>,
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

impl<B: Backend> EncodableBlock<B> for ClassifierPredictionHead<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        if self.supports_shared_encoder() {
            command_buffer.with_compute_encoder(|encoder| self.encode_with_shared_encoder(state, parameters, encoder));
        } else {
            self.dense.encode(state, parameters, command_buffer);
            self.activation.encode(state, parameters, command_buffer);
            self.norm.encode(state, parameters, command_buffer);
            self.readout.encode(state, parameters, command_buffer);
        }

        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(
                command_buffer,
                ArrayId::ClassifierPredictionHeadLogits,
                traces.borrow().logits.clone(),
            );
        }

        if parameters.wait_until_completed {
            command_buffer.submit();
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
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        self.dense.encode_with_shared_encoder(state, parameters, encoder);
        self.activation.encode_with_shared_encoder(state, parameters, encoder);
        self.norm.encode_with_shared_encoder(state, parameters, encoder);
        self.readout.encode_with_shared_encoder(state, parameters, encoder);
    }
}
