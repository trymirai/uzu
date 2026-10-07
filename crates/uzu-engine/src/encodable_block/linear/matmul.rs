use derive_more::Debug;
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::QuantizationMode,
        kernel::{
            ActivationQuantization, Kernels,
            matmul::{ActivationFormat, MatmulA, MatmulArguments, MatmulDOps, MatmulKernel, MatmulOutput, MatmulShape},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::{Gather, Linear, LinearInput},
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported linear matmul configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct LinearMatmul<B: Backend> {
    kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    matrix: WeightMatrix<B>,
    biases: Option<B::GlobalBuffer>,
    output_hadamard_factors: Option<B::GlobalBuffer>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
}

fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: u32,
    parameter_tree: Option<&ParameterTree<B>>,
) -> Result<Option<B::GlobalBuffer>, LinearMatmulError<B>> {
    if parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    Ok(parameter_tree
        .map(|tree| tree.leaf("biases")?.validate(&[output_dim], weights_data_type)?.read_buffer())
        .transpose()?)
}

impl<B: Backend> LinearMatmul<B> {
    pub fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<B::GlobalBuffer>,
    ) -> Result<Self, LinearMatmulError<B>> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }
        let matrix =
            WeightMatrix::load(weights_tree, spec, Layout::OutputInput, output_dim, input_dim, weights_data_type)?;
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(LinearMatmulError::UnsupportedConfiguration(
                "fused output-hadamard factors require quantized weights".into(),
            ));
        }

        let biases = load_biases(weights_data_type, output_data_type, output_dim, bias_tree)?;

        let kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            kernel: Mutex::new(kernel),
            matrix,
            biases,
            output_hadamard_factors,
            input_dim,
            output_dim,
            output_data_type,
        })
    }

    pub(super) fn prepare_a8(
        &mut self,
        context: &B::Context,
    ) -> Option<ActivationQuantization> {
        let signed_codes = self.matrix.quantization()?.mode != QuantizationMode::U4;
        let mut candidate = self.single_matmul_shape(1, false);
        candidate.signed_codes = signed_codes;
        let quantization = self.kernel.lock().select_activation_quantization(&candidate, context)?;
        self.matrix.try_prepare_a8_storage().then_some(quantization)
    }

    pub(super) fn encode_with_a(
        &self,
        a: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        gather: Option<Gather<impl BufferRef<Backend = B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let (output_dim, mut gather_indices) =
            gather.map_or((self.output_dim, None), |gather| (gather.output_dim, Some(gather.indices)));
        let mut output = command_buffer.allocate_scratch_for_shape(&[batch_dim, output_dim], self.output_data_type)?;
        let blocks = self.matrix.blocks();
        let row_stride = (blocks.len() > 1).then_some(output_dim);
        let mut row_offset = 0;
        for (rows, b) in blocks {
            let gather_indices = gather_indices.take();
            self.kernel.lock().encode(
                MatmulArguments {
                    a,
                    b,
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput {
                        values: (&mut output).subrange_mut(row_offset * self.output_data_type.size_in_bytes()..),
                        row_stride,
                        ops: self.d_ops(),
                    },
                    n: if gather_indices.is_some() {
                        output_dim
                    } else {
                        rows
                    },
                    gather_indices,
                    m: batch_dim,
                    k: self.input_dim,
                },
                command_buffer,
            )?;
            row_offset += rows as usize;
        }
        Ok(output)
    }

    fn single_matmul_shape(
        &self,
        batch_dim: u32,
        a_full_precision: bool,
    ) -> MatmulShape {
        let b = self.matrix.single_matmul_b();
        MatmulShape {
            m: batch_dim,
            n: self.output_dim,
            k: self.input_dim,
            b_transpose: true,
            b_leading_dimension: None,
            b_prologue: b.b_prologue(),
            b_is_trellis: b.is_trellis(),
            b_bits: b.bits_per_b(),
            b_group_size: b.group_size(),
            signed_codes: b.signed_codes(),
            a_full_precision,
            gathered: false,
            params_layout: b.quant_params_layout(),
            d_transform: self.d_ops().mask(),
        }
    }

    fn d_ops(&self) -> MatmulDOps<'_, B> {
        MatmulDOps {
            bias: self.biases.as_ref(),
            rht_factors: self.output_hadamard_factors.as_ref(),
            ..MatmulDOps::none()
        }
    }
}

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("matmul");

        let output = self.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            batch_dim,
            None::<Gather<&B::ScratchBuffer>>,
            command_buffer,
        )?;

        command_buffer.pop_debug_group();

        Ok(output)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_with_a(input.as_matmul_a(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer)
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &B::Context,
    ) -> ActivationFormat {
        self.kernel.lock().select_activation_format(&self.single_matmul_shape(batch_dim, true), context)
    }
}
