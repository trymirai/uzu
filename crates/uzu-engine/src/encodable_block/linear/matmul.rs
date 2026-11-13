use derive_more::Debug;
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{
            Kernels,
            matmul::{
                ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                MatmulShape,
            },
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearEncodeError, LinearInput},
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Matmul error: {0}")]
    Matmul(#[source] <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::Error),
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
    biases: Option<B::GlobalGpuBuffer>,
    output_hadamard_factors: Option<B::GlobalGpuBuffer>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
    prepared_activation_format: ActivationFormat,
}

async fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: u32,
    parameter_tree: Option<&ParameterTree<'_, B>>,
) -> Result<Option<B::GlobalGpuBuffer>, LinearMatmulError<B>> {
    if parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    Ok(if let Some(tree) = parameter_tree {
        Some(tree.read_buffer("biases", &[output_dim], weights_data_type).await?)
    } else {
        None
    })
}

impl<B: Backend> LinearMatmul<B> {
    pub(super) fn input_dim(&self) -> u32 {
        self.input_dim
    }

    pub(super) fn output_dim(&self) -> u32 {
        self.output_dim
    }

    pub(super) fn output_data_type(&self) -> DataType {
        self.output_data_type
    }

    pub async fn load(
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        spec: AnyWeightMatrixSpec,
        output_hadamard_factors: Option<B::GlobalGpuBuffer>,
        weights_tree: &ParameterTree<'_, B>,
        bias_tree: Option<&ParameterTree<'_, B>>,
        context: &B::Context,
    ) -> Result<Self, LinearMatmulError<B>> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }

        let matrix = WeightMatrix::load(
            output_dim,
            input_dim,
            weights_data_type,
            spec,
            Layout::OutputInput,
            weights_tree,
            context,
        )
        .await?;
        if output_hadamard_factors.is_some() && matrix.matmul_b().quantized().is_none() {
            return Err(LinearMatmulError::UnsupportedConfiguration(
                "fused output-hadamard factors require quantized weights".into(),
            ));
        }

        let biases = load_biases(weights_data_type, output_data_type, output_dim, bias_tree).await?;

        let kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::Matmul)?;

        Ok(Self {
            kernel: Mutex::new(kernel),
            matrix,
            biases,
            output_hadamard_factors,
            input_dim,
            output_dim,
            output_data_type,
            prepared_activation_format: ActivationFormat::Bf16,
        })
    }

    pub(super) fn prepare_activation(
        &mut self,
        context: &B::Context,
    ) -> Result<ActivationFormat, B::Error> {
        {
            let shape = self.matmul_shape(1, ActivationFormat::Bf16);
            let kernel = self.kernel.lock();
            self.matrix.optimize_weights_layout(|weights| kernel.optimize_weights_layout(&shape, weights, context))?;
        }
        let shape = self.matmul_shape(1, ActivationFormat::Bf16);
        self.prepared_activation_format =
            self.kernel.lock().activation_preparation(&shape, self.matmul_b().weights_layout(), context);
        Ok(self.prepared_activation_format)
    }

    pub(super) fn encode_with_a(
        &self,
        a: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        output_dim: u32,
        gather_indices: Option<impl BufferRef<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, output_dim], self.output_data_type)
            .map_err(LinearEncodeError::Backend)?;
        self.encode_output(
            a,
            batch_dim,
            output_dim,
            gather_indices,
            MatmulOutput::new(&mut output, self.d_ops()),
            command_buffer,
        )?;

        Ok(output)
    }

    pub(super) fn encode_columns(
        &self,
        a: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        output: &mut B::ScratchBuffer,
        column: u32,
        row_stride: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), LinearEncodeError<B>> {
        let element = self.output_data_type.size_in_bytes();
        if !self.matmul_b().is_trellis() {
            unreachable!("TODO: support strided output in ordinary GEMM/GEMV");
        }
        let output = MatmulOutput {
            values: output.subrange_mut(column as usize * element..),
            row_stride: Some(row_stride),
            ops: self.d_ops(),
        };
        self.encode_output(a, batch_dim, self.output_dim, None::<&B::ScratchBuffer>, output, command_buffer)
    }

    fn encode_output(
        &self,
        a: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        output_dim: u32,
        gather_indices: Option<impl BufferRef<Backend = B>>,
        output: MatmulOutput<'_, B, impl BufferMut<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), LinearEncodeError<B>> {
        self.kernel
            .lock()
            .encode(
                MatmulArguments {
                    a,
                    b: self.matmul_b(),
                    b_leading_dimension: None,
                    b_transpose: true,
                    output,
                    gather_indices,
                    m: batch_dim,
                    n: output_dim,
                    k: self.input_dim,
                },
                command_buffer,
            )
            .map_err(LinearEncodeError::Matmul)
    }

    fn matmul_shape(
        &self,
        batch_dim: u32,
        a: ActivationFormat,
    ) -> MatmulShape {
        let b = self.matmul_b();
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
            a,
            gathered: false,
            params_layout: b.quant_params_layout(),
            d_transform: self.d_ops().mask(),
        }
    }

    fn matmul_b(&self) -> MatmulB<&B::GlobalGpuBuffer> {
        self.matrix.matmul_b()
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
    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        self.encode_with_a(input.as_ref(), batch_dim, self.output_dim, None::<&B::ScratchBuffer>, command_buffer)
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        output_dim: Option<u32>,
        context: &B::Context,
    ) -> ActivationFormat {
        let mut shape = self.matmul_shape(batch_dim, ActivationFormat::Bf16);
        shape.n = output_dim.unwrap_or(self.output_dim);
        self.kernel.lock().select_activation_format(
            &shape,
            self.matmul_b().weights_layout(),
            self.prepared_activation_format,
            context,
        )
    }
}
