//! Prediction head encodable for classification output.
//!
//! The head is a chain `maybe(dense) → maybe(activation) → maybe(norm) → readout`.
//! Pooled classifiers (BERT-style) wire all four steps; bare linear per-token
//! taggers (e.g. openai/privacy-filter) only set `readout`.

#[cfg(feature = "tracing")]
use crate::forward_pass::state::ArrayId;
use crate::{
    backends::common::{Backend, Encoder},
    encodable_block::{Activation, Normalization, linear::Linear},
    forward_pass::state::ForwardPassState,
};

pub struct ClassifierPredictionHead<B: Backend> {
    dense: Option<Box<dyn Linear<B>>>,
    activation: Option<Activation<B>>,
    norm: Option<Normalization<B>>,
    readout: Box<dyn Linear<B>>,
    #[allow(dead_code)]
    num_labels: usize,
}

impl<B: Backend> ClassifierPredictionHead<B> {
    pub fn new(
        dense: Option<Box<dyn Linear<B>>>,
        activation: Option<Activation<B>>,
        norm: Option<Normalization<B>>,
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
        if let Some(dense) = self.dense.as_ref() {
            dense.encode(state, encoder)?;
        }
        if let Some(activation) = self.activation.as_ref() {
            activation.encode(state, encoder)?;
        }
        if let Some(norm) = self.norm.as_ref() {
            norm.encode(state, encoder)?;
        }
        self.readout.encode(state, encoder)?;

        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::ClassifierPredictionHeadLogits, traces.borrow().logits.clone());
        }

        Ok(())
    }
}
