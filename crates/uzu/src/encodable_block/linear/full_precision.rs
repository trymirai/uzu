use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use half::{bf16, f16};
use thiserror::Error;

use super::Linear;
use crate::{
    ArrayElement, DataType,
    array::ArrayContextExt,
    backends::common::{
        Backend, CommandBuffer,
        kernel::matmul::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel, MatmulError, MatmulKernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
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
    kernel: RefCell<<B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel>,
    bias_buffer: Option<Rc<RefCell<B::Buffer>>>,
    weights_buffer: Rc<RefCell<B::Buffer>>,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl<B: Backend> FullPrecisionLinear<B> {
    pub fn new(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
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

        let kernel = <B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel::new(context, precision)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer: weights.buffer(),
            input_dim,
            output_dim,
            input_array_id,
            output_array_id,
        })
    }

    pub fn new_selected_rows(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        selected_rows: &[usize],
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, FullPrecisionLinearError<B>> {
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

        let weights_buffer = compact_rows_buffer(context, &weights, selected_rows, input_dim, "selected_rows_weights");
        let bias_buffer = match parameter_tree.leaf_array("biases") {
            Ok(biases) => Some(compact_rows_buffer(context, &biases, selected_rows, 1, "selected_rows_biases")),
            Err(_) => None,
        };
        let kernel = <B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel::new(context, precision)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer,
            input_dim,
            output_dim: selected_rows.len(),
            input_array_id,
            output_array_id,
        })
    }

    pub fn new_selected_columns(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        selected_columns: &[usize],
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, FullPrecisionLinearError<B>> {
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

        let weights_buffer = compact_columns_buffer(
            context,
            &weights,
            selected_columns,
            output_dim,
            input_dim,
            "selected_columns_weights",
        );
        let bias_buffer = match parameter_tree.leaf_array("biases") {
            Ok(biases) => Some(biases.buffer()),
            Err(_) => None,
        };
        let kernel = <B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel::new(context, precision)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer,
            input_dim: selected_columns.len(),
            output_dim,
            input_array_id,
            output_array_id,
        })
    }
}

fn compact_rows_buffer<B: Backend>(
    context: &B::Context,
    source: &crate::array::Array<B>,
    selected_rows: &[usize],
    input_dim: usize,
    label: &str,
) -> Rc<RefCell<B::Buffer>> {
    match source.data_type() {
        DataType::BF16 => compact_rows_buffer_t::<B, bf16>(context, source, selected_rows, input_dim, label),
        DataType::F16 => compact_rows_buffer_t::<B, f16>(context, source, selected_rows, input_dim, label),
        DataType::F32 => compact_rows_buffer_t::<B, f32>(context, source, selected_rows, input_dim, label),
        dtype => panic!("Unsupported dtype for selected full-precision rows: {dtype:?}"),
    }
}

fn compact_rows_buffer_t<B: Backend, T: ArrayElement + Copy>(
    context: &B::Context,
    source: &crate::array::Array<B>,
    selected_rows: &[usize],
    input_dim: usize,
    label: &str,
) -> Rc<RefCell<B::Buffer>> {
    let values = source.as_slice::<T>();
    let mut compact = Vec::with_capacity(selected_rows.len() * input_dim);
    for &row in selected_rows {
        let start = row * input_dim;
        compact.extend_from_slice(&values[start..start + input_dim]);
    }
    context.create_array_from(&[selected_rows.len(), input_dim], &compact, label).buffer()
}

fn compact_columns_buffer<B: Backend>(
    context: &B::Context,
    source: &crate::array::Array<B>,
    selected_columns: &[usize],
    output_dim: usize,
    input_dim: usize,
    label: &str,
) -> Rc<RefCell<B::Buffer>> {
    match source.data_type() {
        DataType::BF16 => {
            compact_columns_buffer_t::<B, bf16>(context, source, selected_columns, output_dim, input_dim, label)
        },
        DataType::F16 => {
            compact_columns_buffer_t::<B, f16>(context, source, selected_columns, output_dim, input_dim, label)
        },
        DataType::F32 => {
            compact_columns_buffer_t::<B, f32>(context, source, selected_columns, output_dim, input_dim, label)
        },
        dtype => panic!("Unsupported dtype for selected full-precision columns: {dtype:?}"),
    }
}

fn compact_columns_buffer_t<B: Backend, T: ArrayElement + Copy>(
    context: &B::Context,
    source: &crate::array::Array<B>,
    selected_columns: &[usize],
    output_dim: usize,
    input_dim: usize,
    label: &str,
) -> Rc<RefCell<B::Buffer>> {
    let values = source.as_slice::<T>();
    let mut compact = Vec::with_capacity(output_dim * selected_columns.len());
    for row in 0..output_dim {
        let row_offset = row * input_dim;
        for &column in selected_columns {
            compact.push(values[row_offset + column]);
        }
    }
    context.create_array_from(&[output_dim, selected_columns.len()], &compact, label).buffer()
}

impl<B: Backend> Linear<B> for FullPrecisionLinear<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();
        let input_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();

        let bias_borrow = self.bias_buffer.as_ref().map(|b| b.borrow());
        self.kernel.borrow_mut().encode(
            state.context(),
            command_buffer,
            FullPrecisionMatmulArguments {
                a: input_array.buffer().borrow().deref(),
                a_offset: 0,
                b: self.weights_buffer.borrow().deref(),
                output: output_array.buffer().borrow_mut().deref_mut(),
                bias: bias_borrow.as_deref(),
                batch: batch_size,
                input_dim: self.input_dim,
                output_dim: self.output_dim,
            },
        );
        Ok(())
    }
}
