//! Prediction head encodable for classification output.

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        ActivationConfig, Allocation, Backend, Encoder,
        gpu_types::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
    encodable_block::{Normalization, linear::Linear},
};

pub struct ClassifierPredictionHead<B: Backend> {
    dense: Box<dyn Linear<B>>,
    activation_kernel: <B::Kernels as Kernels>::ActivationKernel,
    activation: ActivationConfig,
    activation_data_type: DataType,
    norm: Normalization<B>,
    readout: Box<dyn Linear<B>>,
    hidden_dim: usize,
}

impl<B: Backend> ClassifierPredictionHead<B> {
    pub fn new(
        context: &B::Context,
        dense: Box<dyn Linear<B>>,
        activation: ActivationConfig,
        activation_data_type: DataType,
        norm: Normalization<B>,
        readout: Box<dyn Linear<B>>,
        hidden_dim: usize,
    ) -> Result<Self, B::Error> {
        let activation_kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, activation_data_type, false)?;
        Ok(Self {
            dense,
            activation_kernel,
            activation,
            activation_data_type,
            norm,
            readout,
            hidden_dim,
        })
    }

    pub fn encode(
        &self,
        input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = 1;
        let dense = self.dense.encode(input, batch_dim, encoder)?;
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel");
        }
        let mut activated = encoder.allocate_scratch(size_for_shape(&[self.hidden_dim], self.activation_data_type))?;
        self.activation_kernel.encode(
            Some(&dense),
            &mut activated,
            self.hidden_dim as u32,
            self.activation.act_type(),
            encoder,
        );
        let normalized = self.norm.encode(&activated, 0, batch_dim, encoder)?;
        self.readout.encode(normalized, batch_dim, encoder)
    }
}
