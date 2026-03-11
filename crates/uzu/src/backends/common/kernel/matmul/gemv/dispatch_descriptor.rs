use super::{super::matmul_arguments::MatmulArguments, specialization::Specialization};
use crate::{
    DataType,
    backends::common::{Backend, kernel::matmul::MatmulError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSource {
    None,
    Bias,
}

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub specialization: Specialization,
    pub matrix_is_rhs: bool,
    pub output_source: OutputSource,
    pub input_dimension: i32,
    pub output_dimension: i32,
    pub matrix_leading_dim: i32,
    pub alpha: f32,
    pub beta: f32,
    pub batch_shape: [i32; 1],
    pub vector_batch_stride: [i64; 1],
    pub matrix_batch_stride: [i64; 1],
    pub bias_batch_stride: [i64; 1],
    pub bias_stride: i32,
    pub batch_rows: i32,
}

impl DispatchDescriptor {
    pub fn try_new<B: Backend>(
        data_type: DataType,
        arguments: &MatmulArguments<B>,
        gemv_max_batch: i32,
    ) -> Result<Option<Self>, MatmulError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        if !arguments.transpose_b {
            return Ok(None);
        }

        let m = arguments.batch;
        let n = arguments.output_dim;

        if n == 1 {
            if m != 1 {
                return Ok(None);
            }
        } else if m > gemv_max_batch {
            return Ok(None);
        }

        let matrix_is_rhs = n != 1;
        let transpose_matrix = if matrix_is_rhs {
            !arguments.transpose_b
        } else {
            false
        };

        let output_source = if arguments.bias.is_some() {
            OutputSource::Bias
        } else {
            OutputSource::None
        };

        let (apply_output_scale_and_accumulate, alpha, beta, bias_stride) = match output_source {
            OutputSource::None => (false, 1.0f32, 0.0f32, 0),
            OutputSource::Bias => (true, 1.0f32, 1.0f32, 1),
        };

        let output_dimension = if matrix_is_rhs {
            arguments.output_dim
        } else {
            arguments.batch
        };

        let specialization = Specialization::select(
            transpose_matrix,
            arguments.input_dim,
            output_dimension,
            apply_output_scale_and_accumulate,
        );

        let input_dimension = arguments.input_dim;
        let matrix_leading_dim = if matrix_is_rhs {
            arguments.leading_dim_b
        } else {
            arguments.leading_dim_a
        };

        let batch_shape = [if arguments.batch_count > 1 {
            arguments.batch_count
        } else {
            1
        }];

        let elements_per_matrix_a = (arguments.batch as i64) * (arguments.leading_dim_a as i64);
        let elements_per_matrix_b = if arguments.transpose_b {
            (arguments.output_dim as i64) * (arguments.leading_dim_b as i64)
        } else {
            (arguments.input_dim as i64) * (arguments.leading_dim_b as i64)
        };

        let vector_batch_stride = [if matrix_is_rhs {
            elements_per_matrix_a
        } else {
            elements_per_matrix_b
        }];

        let matrix_batch_stride = [if matrix_is_rhs {
            elements_per_matrix_b
        } else {
            elements_per_matrix_a
        }];

        let bias_batch_stride = [if arguments.batch_count > 1 {
            (output_dimension as i64) * (arguments.leading_dim_d as i64)
        } else {
            0
        }];

        let batch_rows = arguments.batch;

        Ok(Some(Self {
            specialization,
            matrix_is_rhs,
            output_source,
            input_dimension,
            output_dimension,
            matrix_leading_dim,
            alpha,
            beta,
            batch_shape,
            vector_batch_stride,
            matrix_batch_stride,
            bias_batch_stride,
            bias_stride,
            batch_rows,
        }))
    }

    pub fn bias_is_fused(&self) -> bool {
        matches!(self.output_source, OutputSource::Bias)
    }
}
