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
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.dense.encode(state, parameters, command_buffer)?;
        self.activation.encode(state, parameters, command_buffer)?;
        self.norm.encode(state, parameters, command_buffer)?;
        self.readout.encode(state, parameters, command_buffer)?;

        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(
                command_buffer,
                ArrayId::ClassifierPredictionHeadLogits,
                traces.borrow().logits.clone(),
            );
        }

        Ok(())
    }
}
