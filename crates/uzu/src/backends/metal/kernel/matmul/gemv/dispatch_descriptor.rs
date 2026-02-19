use std::sync::OnceLock;

use super::pipeline_configuration::PipelineConfiguration;
use crate::{
    DataType,
    backends::metal::{context::MetalContext, error::MetalError, kernel::matmul::common::MatmulArguments},
};

const DEFAULT_GEMV_MAX_BATCH: i32 = 8;

static GEMV_MAX_BATCH: OnceLock<i32> = OnceLock::new();

fn max_gemv_batch_threshold() -> i32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputSource {
    None,
    Bias,
    C,
}

#[derive(Debug, Clone)]
pub(crate) struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) matrix_is_rhs: bool,
    pub(crate) output_source: OutputSource,
    pub(crate) input_dimension: i32,
    pub(crate) output_dimension: i32,
    pub(crate) matrix_leading_dim: i32,
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
    pub(crate) batch_shape: [i32; 1],
    pub(crate) vector_batch_stride: [i64; 1],
    pub(crate) matrix_batch_stride: [i64; 1],
    pub(crate) bias_batch_stride: [i64; 1],
    pub(crate) bias_stride: i32,
    pub(crate) batch_rows: i32,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MetalContext,
        data_type: DataType,
        arguments: &MatmulArguments,
    ) -> Result<Option<Self>, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported data type for GEMV: {data_type:?}")));
        }

        if arguments.transpose_a || !arguments.transpose_b {
            return Ok(None);
        }

        let m = arguments.batch;
        let n = arguments.output_dim;

        let max_gemv_batch = max_gemv_batch_threshold();

        if n == 1 {
            if m != 1 {
                return Ok(None);
            }
        } else if m > max_gemv_batch {
            return Ok(None);
        }

        let matrix_is_rhs = n != 1;
        let transpose_matrix = if matrix_is_rhs {
            !arguments.transpose_b
        } else {
            arguments.transpose_a
        };

        let output_source = if arguments.c.is_some() {
            OutputSource::C
        } else if arguments.bias.is_some() {
            OutputSource::Bias
        } else {
            OutputSource::None
        };

        let (apply_output_scale_and_accumulate, alpha, beta, bias_stride) = match output_source {
            OutputSource::None => (false, 1.0f32, 0.0f32, 0),
            OutputSource::Bias => (true, 1.0f32, 1.0f32, 1),
            OutputSource::C => (true, arguments.alpha, arguments.beta, arguments.ldd),
        };

        let output_dimension = if matrix_is_rhs {
            arguments.output_dim
        } else {
            arguments.batch
        };

        let pipeline_configuration = PipelineConfiguration::select(
            transpose_matrix,
            arguments.input_dim,
            output_dimension,
            apply_output_scale_and_accumulate,
        );

        let input_dimension = arguments.input_dim;
        let matrix_leading_dim = if matrix_is_rhs {
            arguments.ldb
        } else {
            arguments.lda
        };

        let batch_shape = [if arguments.batch_count > 1 {
            arguments.batch_count
        } else {
            1
        }];

        let elements_per_matrix_a = (arguments.batch as i64) * (arguments.lda as i64);
        let elements_per_matrix_b = if arguments.transpose_b {
            (arguments.output_dim as i64) * (arguments.ldb as i64)
        } else {
            (arguments.input_dim as i64) * (arguments.ldb as i64)
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
            (output_dimension as i64) * (arguments.ldd as i64)
        } else {
            0
        }];

        let batch_rows = arguments.batch;

        Ok(Some(Self {
            pipeline_configuration,
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

    pub(crate) fn bias_is_fused(&self) -> bool {
        matches!(self.output_source, OutputSource::Bias)
    }
}
