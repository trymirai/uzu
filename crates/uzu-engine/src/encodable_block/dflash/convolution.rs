use super::DraftNewError;
use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::GroupedConvolutionKernel},
    config::dflash::GroupedConvolutionConfig,
    data_type::DataType,
    encodable_block::linear::Linear,
    parameters::ParameterTree,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ConvolutionStage {
    Input = 0,
    Output = 1,
}

impl ConvolutionStage {
    pub fn index(self) -> u32 {
        self as u32
    }
}

pub struct GroupedConvolution<B: Backend> {
    base_kernel: Allocation<B>,
    projection: Box<dyn Linear<B>>,
    kernel: <B::Kernels as Kernels>::GroupedConvolutionKernel,
    model_dim: u32,
    groups: u32,
    group_size: u32,
    kernel_size: u32,
    data_type: DataType,
}

impl<B: Backend> GroupedConvolution<B> {
    pub fn new(
        context: &B::Context,
        config: &GroupedConvolutionConfig,
        model_dim: u32,
        block_size: u32,
        parameters: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DraftNewError<B>> {
        if [model_dim, block_size, config.kernel_size, config.group_size].contains(&0)
            || config.kernel_size > block_size
            || !model_dim.is_multiple_of(config.group_size)
        {
            return Err(DraftNewError::InvalidConfiguration("invalid grouped convolution dimensions"));
        }
        let groups = model_dim / config.group_size;
        let projection_dim = 2u32
            .checked_mul(config.kernel_size)
            .and_then(|value| value.checked_mul(groups))
            .ok_or_else(|| DraftNewError::InvalidConfiguration("projection dimension overflow"))?;
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
        let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(context, data_type)
            .map_err(DraftNewError::Backend)?;
        Ok(Self {
            base_kernel,
            projection,
            kernel,
            model_dim,
            groups,
            group_size: config.group_size,
            kernel_size: config.kernel_size,
            data_type,
        })
    }

    pub fn prepare(
        &self,
        input: Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let mut projection_input = encoder.allocate_scratch(input.size())?;
        encoder.encode_copy(&input, .., &mut projection_input, ..);
        let coefficients = self.projection.encode(projection_input, sequence_length, encoder)?;
        Ok((self.encode_stage(&input, &coefficients, sequence_length, ConvolutionStage::Input, encoder)?, coefficients))
    }

    pub fn finish(
        &self,
        input: Allocation<B>,
        coefficients: Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.encode_stage(&input, &coefficients, sequence_length, ConvolutionStage::Output, encoder)
    }

    fn encode_stage(
        &self,
        input: &Allocation<B>,
        coefficients: &Allocation<B>,
        sequence_length: u32,
        stage: ConvolutionStage,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch_for_shape(&[sequence_length, self.model_dim], self.data_type)?;
        self.kernel.encode(
            input,
            coefficients,
            &self.base_kernel,
            &mut output,
            sequence_length,
            self.model_dim,
            self.groups,
            self.group_size,
            self.kernel_size,
            stage.index(),
            encoder,
        );
        Ok(output)
    }
}
