use std::any::Any;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::GroupedConvolutionKernel},
    config::dflash::GroupedConvolutionConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{Attention, rope::PrecalculatedRoPE},
        },
        mlp::Mlp,
        transformer_layer::TransformerLayer,
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
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
enum Stage {
    Input = 0,
    Output = 1,
}

struct GroupedConvolution<B: Backend> {
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
    fn new(
        context: &B::Context,
        config: &GroupedConvolutionConfig,
        model_dim: u32,
        block_size: u32,
        parameters: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, ConvolutionNewError<B>> {
        if [model_dim, block_size, config.kernel_size, config.group_size].contains(&0)
            || config.kernel_size > block_size
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
        let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(context, data_type)
            .map_err(ConvolutionNewError::Backend)?;
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

    fn encode_around(
        &self,
        input: Allocation<B>,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
        encode: impl FnOnce(Allocation<B>, &mut Encoder<B>) -> Result<Allocation<B>, B::Error>,
    ) -> Result<Allocation<B>, B::Error> {
        let coefficients = self.project(&input, sequence_length, encoder)?;
        let input = self.encode_stage(&input, &coefficients, sequence_length, Stage::Input, encoder)?;
        let output = encode(input, encoder)?;
        self.encode_stage(&output, &coefficients, sequence_length, Stage::Output, encoder)
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

    fn encode_stage(
        &self,
        input: &Allocation<B>,
        coefficients: &Allocation<B>,
        sequence_length: u32,
        stage: Stage,
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
            stage as u32,
            encoder,
        );
        Ok(output)
    }
}

pub fn wrap<B: Backend>(
    mut layer: TransformerLayer<B>,
    context: &B::Context,
    config: &GroupedConvolutionConfig,
    model_dim: u32,
    block_size: u32,
    parameters: &ParameterTree<B>,
    data_type: DataType,
) -> Result<TransformerLayer<B>, ConvolutionNewError<B>> {
    if (layer.mixer.as_ref() as &dyn Any).downcast_ref::<Attention<B>>().is_none() {
        return Err(ConvolutionNewError::InvalidConfiguration("DFlash convolution requires a direct Attention mixer"));
    }
    let convolution = |name: &str| {
        GroupedConvolution::new(context, config, model_dim, block_size, &parameters.subtree(name), data_type)
    };
    layer.mixer = Box::new(ConvolvedAttention {
        inner: layer.mixer,
        convolution: convolution("attention")?,
    });
    layer.mlp = Box::new(ConvolvedMlp {
        inner: layer.mlp,
        convolution: convolution("mlp")?,
    });
    Ok(layer)
}

struct ConvolvedAttention<B: Backend> {
    inner: Box<dyn Mixer<B>>,
    convolution: GroupedConvolution<B>,
}

impl<B: Backend> Mixer<B> for ConvolvedAttention<B> {
    fn speculation_supported(&self) -> bool {
        self.inner.speculation_supported()
    }

    fn max_context_length(&self) -> Option<u32> {
        self.inner.max_context_length()
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        self.inner.create_empty_state(max_context_length, context)
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.convolution.encode_around(hidden, batch_dim.size(), encoder, |hidden, encoder| {
            self.inner.encode(hidden, precalculated_rope, batch_dim, state, encoder)
        })
    }
}

struct ConvolvedMlp<B: Backend> {
    inner: Box<dyn Mlp<B>>,
    convolution: GroupedConvolution<B>,
}

impl<B: Backend> Mlp<B> for ConvolvedMlp<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.convolution
            .encode_around(input, batch_dim, encoder, |input, encoder| self.inner.encode(input, batch_dim, encoder))
    }
}

pub fn attention_of<B: Backend>(mixer: &dyn Mixer<B>) -> Option<&Attention<B>> {
    let any = mixer as &dyn Any;
    any.downcast_ref::<Attention<B>>().or_else(|| {
        any.downcast_ref::<ConvolvedAttention<B>>()
            .and_then(|mixer| (mixer.inner.as_ref() as &dyn Any).downcast_ref::<Attention<B>>())
    })
}
