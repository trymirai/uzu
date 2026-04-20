//! Prediction head encodable for classification output.

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    encodable_block::{Activation, Normalization, linear::Linear},
};

pub struct ClassifierPredictionHead<B: Backend> {
    dense: Box<dyn Linear<B>>,
    activation: Activation<B>,
    norm: Normalization<B>,
    readout: Box<dyn Linear<B>>,
    hidden_dim: usize,
}

impl<B: Backend> ClassifierPredictionHead<B> {
    pub fn new(
        dense: Box<dyn Linear<B>>,
        activation: Activation<B>,
        norm: Normalization<B>,
        readout: Box<dyn Linear<B>>,
        hidden_dim: usize,
    ) -> Self {
        Self {
            dense,
            activation,
            norm,
            readout,
            hidden_dim,
        }
    }

    pub fn encode(
        &self,
        context: &B::Context,
        mut input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = 1;
        let dense = self.dense.encode(context, &mut input, batch_dim, encoder)?;
        let activated = self.activation.encode(&dense, self.hidden_dim, encoder)?;
        let mut normalized = self.norm.encode(&activated, 0, batch_dim, encoder)?;
        self.readout.encode(context, &mut normalized, batch_dim, encoder)
    }
}
