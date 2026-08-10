use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulPath, MatmulShape},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::Linear,
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
    biases: Option<Allocation<B>>,
    output_hadamard_factors: Option<Allocation<B>>,
    input_dim: usize,
    output_dim: usize,
    output_data_type: DataType,
}

fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: usize,
    parameter_tree: Option<&ParameterTree<B>>,
) -> Result<Option<Allocation<B>>, LinearMatmulError<B>> {
    if parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    Ok(parameter_tree
        .map(|tree| tree.leaf("biases")?.validate(&[output_dim], weights_data_type)?.read_allocation())
        .transpose()?)
}

impl<B: Backend> LinearMatmul<B> {
    /// Loads a linear over any parsed spec — full precision, MLX or Int. Hybrid
    /// compositions live in the wrappers, not here.
    pub fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: usize,
        output_dim: usize,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<Allocation<B>>,
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

    pub(super) fn make_codes_signed(&mut self) {
        self.matrix.make_codes_signed();
    }

    pub(super) fn encode_with_a(
        &self,
        a: MatmulA<'_, B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.output_data_type))?;

        self.kernel.lock().encode(
            MatmulArguments {
                a,
                b: self.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: self.d_ops(),
                gather_indices: None,
                m: batch_dim as u32,
                n: self.output_dim as u32,
                k: self.input_dim as u32,
            },
            encoder,
        )?;

        Ok(output)
    }

    fn matmul_b(&self) -> MatmulB<'_, B> {
        self.matrix.matmul_b()
    }

    fn d_ops(&self) -> MatmulDOps<'_, B> {
        MatmulDOps {
            bias: self.biases.as_ref(),
            rht_factors: self.output_hadamard_factors.as_ref(),
            ..MatmulDOps::none()
        }
    }

    pub(super) fn select_path(
        &self,
        batch_dim: usize,
        context: &B::Context,
    ) -> MatmulPath {
        let b = self.matmul_b();
        let shape = MatmulShape {
            m: batch_dim as u32,
            n: self.output_dim as u32,
            k: self.input_dim as u32,
            b_transpose: true,
            b_leading_dimension: None,
            b_prologue: b.b_prologue(),
            b_bits: b.bits_per_b(),
            b_group_size: b.group_size(),
            signed_codes: b.signed_codes(),
            a_full_precision: true,
            gathered: false,
            d_transform: self.d_ops().mask(),
        };
        self.kernel.lock().select_path(&shape, context)
    }
}

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("matmul");

        let output = self.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            batch_dim,
            encoder,
        )?;

        encoder.pop_debug_group();

        Ok(output)
    }
}
