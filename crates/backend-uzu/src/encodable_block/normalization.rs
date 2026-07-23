use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, NormalizationKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

// TODO: clean up

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PostLayerScalar {
    None,
    ScaleResidualSum(f32),
    ScaleOutput(f32),
}

#[derive(Debug, Error)]
pub enum NormalizationNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
}

pub struct Normalization<B: Backend> {
    epsilon: f32,
    scale_offset: Option<f32>,
    scales: Allocation<B>,
    biases: Option<Allocation<B>>,
    element_count: usize,
    hadamard_factors: Option<Allocation<B>>,
    post_layer_scalar_value: f32,
    data_type: DataType,
    kernel: <B::Kernels as Kernels>::NormalizationKernel,
}

impl<B: Backend> Normalization<B> {
    pub fn new(
        element_count: usize,
        hadamard_factors: Option<Allocation<B>>,
        copy_to_shortcut: bool,
        residual_add: bool,
        post_layer_scalar: PostLayerScalar,
        data_type: DataType,
        config: &NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, NormalizationNewError<B>> {
        assert!(copy_to_shortcut || !residual_add, "residual_add requires shortcut");

        let scales = parameter_tree.leaf("scales")?.validate(&[element_count], DataType::F32)?.read_allocation()?;
        let biases = config
            .has_biases
            .then(|| parameter_tree.leaf("biases")?.validate(&[element_count], DataType::F32)?.read_allocation())
            .transpose()?;

        let (scale_residual_sum, scale_output, post_layer_scalar_value) = match post_layer_scalar {
            PostLayerScalar::None => (false, false, 1.0),
            PostLayerScalar::ScaleResidualSum(value) => (true, false, value),
            PostLayerScalar::ScaleOutput(value) => (false, true, value),
        };

        let kernel = <B::Kernels as Kernels>::NormalizationKernel::new(
            context,
            data_type,
            DataType::F32,
            data_type,
            DataType::F32,
            false,
            config.subtract_mean,
            config.upcast_mode == UpcastMode::FullLayer,
            copy_to_shortcut,
            residual_add,
            hadamard_factors.is_some(),
            scale_residual_sum,
            scale_output,
            biases.is_some(),
        )
        .map_err(NormalizationNewError::Backend)?;

        Ok(Self {
            epsilon: config.epsilon,
            scale_offset: config.scale_offset,
            scales,
            biases,
            element_count,
            hadamard_factors,
            post_layer_scalar_value,
            data_type,
            kernel,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        row_offset: usize,
        row_count: usize,
        shortcut: Option<&mut Allocation<B>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let row_size = self.element_count * self.data_type.size_in_bytes();
        let row_offset_bytes = row_offset * row_size;
        let shortcut = shortcut.map(|shortcut| (shortcut, row_offset_bytes));
        let mut output = encoder.allocate_scratch(size_for_shape(&[row_count, self.element_count], self.data_type))?;
        self.kernel.encode(
            Some((input, row_offset_bytes)),
            &self.scales,
            self.biases.as_ref(),
            &mut output,
            shortcut,
            self.hadamard_factors.as_ref(),
            row_count as u32,
            self.element_count as u32,
            self.epsilon,
            self.scale_offset.unwrap_or(0.0),
            self.post_layer_scalar_value,
            encoder,
        );
        Ok(output)
    }
}
