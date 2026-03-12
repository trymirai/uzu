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

        let output_source = if arguments.bias.is_some() {
            OutputSource::Bias
        } else {
            OutputSource::None
        };

        let apply_output_scale_and_accumulate = matches!(output_source, OutputSource::Bias);
        let output_dimension = if matrix_is_rhs { n } else { m };

        let specialization = Specialization::select(
            false,
            arguments.input_dim,
            output_dimension,
            apply_output_scale_and_accumulate,
        );

        Ok(Some(Self {
            specialization,
            matrix_is_rhs,
            output_source,
            input_dimension: arguments.input_dim,
            output_dimension,
            batch_rows: m,
        }))
    }

    pub fn bias_is_fused(&self) -> bool {
        matches!(self.output_source, OutputSource::Bias)
    }
}
