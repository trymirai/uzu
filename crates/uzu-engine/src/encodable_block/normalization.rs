use std::sync::Arc;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{Kernels, NormalizationKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    encodable_block::linear::InputRhtParameters,
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::size_for_shape::size_for_shape,
};

// This was used a lot previously, but is now only used in weaver and confidence heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortcutMode {
    None,
    Copy,
    Add,
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
    scales: Option<B::GlobalGpuBuffer>,
    biases: Option<B::GlobalGpuBuffer>,
    element_count: u32,
    mixture_size: u32,
    hadamard_factors: Option<Arc<B::GlobalGpuBuffer>>,
    input_data_type: DataType,
    output_data_type: DataType,
    kernel: <B::Kernels as Kernels>::NormalizationKernel,
}

impl<B: Backend> Normalization<B> {
    pub async fn new(
        element_count: u32,
        input_data_type: DataType,
        output_data_type: DataType,
        hadamard_factors: Option<InputRhtParameters<B>>,
        shortcut_mode: ShortcutMode,
        config: &NormalizationConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, NormalizationNewError<B>> {
        Self::new_mixture(
            None,
            element_count,
            input_data_type,
            output_data_type,
            hadamard_factors,
            shortcut_mode,
            config,
            parameter_tree,
            context,
        )
        .await
    }

    pub async fn new_mixture(
        mixture_size: Option<u32>,
        element_count: u32,
        input_data_type: DataType,
        output_data_type: DataType,
        hadamard_factors: Option<InputRhtParameters<B>>,
        shortcut_mode: ShortcutMode,
        config: &NormalizationConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, NormalizationNewError<B>> {
        let (hadamard_size, hadamard_factors) = hadamard_factors.map(|rht| (rht.hadamard_size, rht.signs)).unzip();
        assert!(element_count.is_multiple_of(4), "normalization width ({element_count}) must be a multiple of 4");
        let affine_shape: &[u32] = if let Some(mixture_size) = mixture_size {
            &[mixture_size, element_count]
        } else {
            &[element_count]
        };
        let (copy_to_shortcut, residual_add) = match shortcut_mode {
            ShortcutMode::None => (false, false),
            ShortcutMode::Copy => (true, false),
            ShortcutMode::Add => (true, true),
        };

        let scales = if config.has_scale {
            Some(parameter_tree.read_buffer("scales", affine_shape, DataType::F32).await?)
        } else {
            None
        };
        let biases = if config.has_biases {
            Some(parameter_tree.read_buffer("biases", affine_shape, DataType::F32).await?)
        } else {
            None
        };

        let kernel = <B::Kernels as Kernels>::NormalizationKernel::new(
            context,
            input_data_type,
            DataType::F32,
            output_data_type,
            DataType::F32,
            false,
            config.subtract_mean,
            config.upcast_mode == UpcastMode::FullLayer,
            copy_to_shortcut,
            residual_add,
            hadamard_factors.is_some(),
            hadamard_size.unwrap_or(HADAMARD_TRANSFORM_BLOCK_SIZE),
            biases.is_some(),
            scales.is_some(),
        )
        .map_err(NormalizationNewError::Backend)?;

        Ok(Self {
            epsilon: config.epsilon,
            scale_offset: config.scale_offset,
            scales,
            biases,
            element_count,
            mixture_size: mixture_size.unwrap_or(1),
            hadamard_factors,
            input_data_type,
            output_data_type,
            kernel,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        row_offset: u32,
        row_count: u32,
        shortcut: Option<impl BufferMut<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        debug_assert!(row_offset.is_multiple_of(self.mixture_size));
        let row_size = size_for_shape(&[self.element_count], self.input_data_type);
        let row_offset_bytes = row_offset as usize * row_size;
        let shortcut = shortcut.map(|shortcut| shortcut.subrange_mut(row_offset_bytes..));
        let mut output =
            command_buffer.allocate_scratch_for_shape(&[row_count, self.element_count], self.output_data_type)?;
        self.kernel.encode(
            Some(input.subrange(row_offset_bytes..)),
            self.scales.as_ref(),
            self.biases.as_ref(),
            &mut output,
            shortcut,
            self.hadamard_factors.as_deref(),
            row_count,
            self.element_count,
            self.epsilon,
            self.scale_offset.unwrap_or(0.0),
            self.mixture_size,
            command_buffer,
        );

        Ok(output)
    }

    /// Normalizes `row_count` rows of `input` with the scales and biases of mixture component `component`, into
    /// `output`.
    pub fn encode_component(
        &self,
        input: impl BufferRef<Backend = B>,
        component: u32,
        row_count: u32,
        output: impl BufferMut<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        assert!(component < self.mixture_size, "mixture component {component} out of {}", self.mixture_size);
        let affine_size = size_for_shape(&[self.element_count], DataType::F32);
        let affine = component as usize * affine_size..(component as usize + 1) * affine_size;
        self.kernel.encode(
            Some(input),
            self.scales.as_ref().map(|scales| scales.subrange(affine.clone())),
            self.biases.as_ref().map(|biases| biases.subrange(affine.clone())),
            output,
            None::<&mut B::ScratchBuffer>,
            self.hadamard_factors.as_deref(),
            row_count,
            self.element_count,
            self.epsilon,
            self.scale_offset.unwrap_or(0.0),
            1,
            command_buffer,
        );
    }
}
