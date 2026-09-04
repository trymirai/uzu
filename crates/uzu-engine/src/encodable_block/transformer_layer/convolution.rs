use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::GroupedConvolutionKernel},
    config::transformer_layer::GroupedConvolutionConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum ConvolutionNewError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("invalid convolution configuration: {0}")]
    InvalidConfiguration(&'static str),
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum ConvolutionStage {
    Input = 0,
    Output = 1,
}

pub struct GroupedConvolution<B: Backend> {
    base_kernel: Allocation<B>,
    projection: Box<dyn Linear<B>>,
    kernel: <B::Kernels as Kernels>::GroupedConvolutionKernel,
    model_dim: u32,
    groups: u32,
    kernel_size: u32,
    data_type: DataType,
}

pub struct GroupedConvolutions<B: Backend> {
    pub attention: GroupedConvolution<B>,
    pub mlp: GroupedConvolution<B>,
}

impl<B: Backend> GroupedConvolutions<B> {
    pub fn new(
        context: &B::Context,
        config: &GroupedConvolutionConfig,
        model_dim: u32,
        parameters: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, ConvolutionNewError<B>> {
        Ok(Self {
            attention: GroupedConvolution::new(
                context,
                config,
                model_dim,
                &parameters.subtree("attention_convolution"),
                data_type,
            )?,
            mlp: GroupedConvolution::new(
                context,
                config,
                model_dim,
                &parameters.subtree("mlp_convolution"),
                data_type,
            )?,
        })
    }
}

impl<B: Backend> GroupedConvolution<B> {
    pub fn new(
        context: &B::Context,
        config: &GroupedConvolutionConfig,
        model_dim: u32,
        parameters: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, ConvolutionNewError<B>> {
        if [model_dim, config.kernel_size, config.group_size].contains(&0)
            || !model_dim.is_multiple_of(config.group_size)
        {
            return Err(ConvolutionNewError::InvalidConfiguration("invalid grouped convolution dimensions"));
        }
        let groups = model_dim / config.group_size;
        let projection_dim = 2u32
            .checked_mul(config.kernel_size)
            .and_then(|value| value.checked_mul(groups))
            .ok_or(ConvolutionNewError::InvalidConfiguration("projection dimension overflow"))?;
        let base_kernel = parameters
            .leaf("base_kernel")?
            .validate(&[2, config.kernel_size, model_dim], data_type)?
            .read_allocation()?;
        let projection = <dyn Linear<B>>::new(
            model_dim,
            [projection_dim],
            false,
            context,
            data_type,
            &parameters.subtree("kernel_projection"),
        )?;
        let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(
            context,
            data_type,
            model_dim,
            config.group_size,
            config.kernel_size,
        )
        .map_err(ConvolutionNewError::Backend)?;
        Ok(Self {
            base_kernel,
            projection,
            kernel,
            model_dim,
            groups,
            kernel_size: config.kernel_size,
            data_type,
        })
    }

    pub fn encode_around<F>(
        &self,
        input: Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
        encode_sublayer: F,
    ) -> Result<Allocation<B>, B::Error>
    where
        F: FnOnce(Allocation<B>, &mut Encoder<B>) -> Result<Allocation<B>, B::Error>,
    {
        let coefficients = self.project(&input, sequence_length, encoder)?;
        let input =
            self.encode_convolution(&input, &coefficients, sequence_length, ConvolutionStage::Input, encoder)?;
        let output = encode_sublayer(input, encoder)?;
        self.encode_convolution(&output, &coefficients, sequence_length, ConvolutionStage::Output, encoder)
    }

    fn project(
        &self,
        input: &Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut projection_input = encoder.allocate_scratch(input.size())?;
        encoder.encode_copy(input, .., &mut projection_input, ..);
        self.projection.encode(projection_input, sequence_length, encoder)
    }

    fn encode_convolution(
        &self,
        input: &Allocation<B>,
        coefficients: &Allocation<B>,
        sequence_length: u32,
        stage: ConvolutionStage,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch_for_shape(&[sequence_length, self.model_dim], self.data_type)?;
        let stage = stage as usize;
        let element_size = self.data_type.size_in_bytes();
        let coefficient_offset = stage * self.kernel_size as usize * self.groups as usize * element_size;
        let base_kernel_offset = stage * self.kernel_size as usize * self.model_dim as usize * element_size;
        self.kernel.encode(
            input,
            (coefficients, coefficient_offset),
            (&self.base_kernel, base_kernel_offset),
            &mut output,
            sequence_length,
            encoder,
        );
        Ok(output)
    }
}
