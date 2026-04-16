use std::{cell::RefCell, ops::Deref, rc::Rc};

use thiserror::Error;

use super::Linear;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            ManualKernels,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum FullPrecisionLinearError<B: Backend> {
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
    #[error("Unsupported data type for full precision linear kernel: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unexpected weights shape: got {got:?}, expected [{expected_output_dim}, {expected_input_dim}]")]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_output_dim: usize,
        expected_input_dim: usize,
    },
    #[error("Weights dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidWeightsDataType {
        expected: DataType,
        got: DataType,
    },
    #[error("Bias shape mismatch: got {got:?}, expected [{expected_output_dim}]")]
    InvalidBiasShape {
        got: Box<[usize]>,
        expected_output_dim: usize,
    },
    #[error("Bias dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidBiasDataType {
        expected: DataType,
        got: DataType,
    },
}

pub struct FullPrecisionLinear<B: Backend> {
    kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    bias_buffer: Option<Rc<RefCell<B::Buffer>>>,
    weights_buffer: Rc<RefCell<B::Buffer>>,
    input_dim: usize,
    output_dim: usize,
    precision: DataType,
}

impl<B: Backend> FullPrecisionLinear<B> {
    pub fn new(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, FullPrecisionLinearError<B>> {
        if !matches!(precision, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(FullPrecisionLinearError::UnsupportedDataType(precision));
        }

        let weights = parameter_tree.leaf_array("weights").map_err(FullPrecisionLinearError::ParameterError)?;
        let weights_shape = weights.shape().to_vec();
        if weights_shape != [output_dim, input_dim] {
            return Err(FullPrecisionLinearError::InvalidWeightsShape {
                got: weights_shape.into_boxed_slice(),
                expected_output_dim: output_dim,
                expected_input_dim: input_dim,
            });
        }

        if weights.data_type() != precision {
            return Err(FullPrecisionLinearError::InvalidWeightsDataType {
                expected: precision,
                got: weights.data_type(),
            });
        }

        let bias_buffer = match parameter_tree.leaf_array("biases") {
            Ok(biases) => {
                let bias_shape = biases.shape().to_vec();
                if bias_shape != [output_dim] {
                    return Err(FullPrecisionLinearError::InvalidBiasShape {
                        got: bias_shape.into_boxed_slice(),
                        expected_output_dim: output_dim,
                    });
                }

                if biases.data_type() != precision {
                    return Err(FullPrecisionLinearError::InvalidBiasDataType {
                        expected: precision,
                        got: biases.data_type(),
                    });
                }

                Some(biases.buffer())
            },
            Err(_) => None,
        };

        let kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(context, precision)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer: weights.buffer(),
            input_dim,
            output_dim,
            precision,
        })
    }
}

impl<B: Backend> Linear<B> for FullPrecisionLinear<B> {
    fn encode(
        &self,
        context: &B::Context,
        input: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.precision))?;
        let bias_borrow = self.bias_buffer.as_ref().map(|b| b.borrow());
        self.kernel.borrow_mut().encode(
            context,
            MatmulArguments {
                a: input,
                b: self.weights_buffer.borrow().deref(),
                ab_scale: 1.0,
                c: match bias_borrow.as_deref() {
                    Some(b) => MatmulArgumentC::Bias(b),
                    None => MatmulArgumentC::None,
                },
                d: &mut output,
                batch_dim: batch_dim as u32,
                input_dim: self.input_dim as u32,
                output_dim: self.output_dim as u32,
            },
            encoder,
        );
        Ok(output)
    }
}
