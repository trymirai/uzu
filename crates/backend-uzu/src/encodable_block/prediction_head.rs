//! Prediction head encodable for classification output.

#[cfg(feature = "tracing")]
use crate::forward_pass::state::ArrayId;
use crate::{
    backends::common::{Backend, Encoder},
    encodable_block::{Activation, Normalization, linear::Linear},
    forward_pass::state::ForwardPassState,
};

pub struct ClassifierPredictionHead<B: Backend> {
    dense: Box<dyn Linear<B>>,
    activation: Activation<B>,
    norm: Normalization<B>,
    readout: Box<dyn Linear<B>>,
    #[allow(dead_code)]
    num_labels: usize,
}

impl<B: Backend> ClassifierPredictionHead<B> {
    pub fn new(
        dense: Box<dyn Linear<B>>,
        activation: Activation<B>,
        norm: Normalization<B>,
        readout: Box<dyn Linear<B>>,
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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.dense.encode(state, encoder)?;
        self.activation.encode(state, encoder)?;
        self.norm.encode(state, encoder)?;
        self.readout.encode(state, encoder)?;

        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::ClassifierPredictionHeadLogits, traces.borrow().logits.clone());
        }

        Ok(())
    }
}
