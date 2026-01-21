use crate::backends::metal::{MTLSize, mtl_size};

use super::pipeline_configuration::{
    PipelineConfiguration, select_configuration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, kernel::matmul::common::MatmulArguments,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxpbySource {
    None,
    Bias,
    C,
}

#[derive(Debug, Clone)]
pub(crate) struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) matrix_is_rhs: bool,
    pub(crate) axpby_source: AxpbySource,
    pub(crate) input_dimension: i32,
    pub(crate) output_dimension: i32,
    pub(crate) matrix_leading_dim: i32,
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
    pub(crate) batch_ndim: i32,
    pub(crate) batch_shape: [i32; 1],
    pub(crate) vector_batch_stride: [i64; 1],
    pub(crate) matrix_batch_stride: [i64; 1],
    pub(crate) bias_batch_stride: [i64; 1],
    pub(crate) bias_stride: i32,
    pub(crate) threadgroups: MTLSize,
    pub(crate) threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MTLContext,
        data_type: DataType,
        arguments: &MatmulArguments,
    ) -> Result<Option<Self>, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for GEMV: {data_type:?}"
            )));
        }

        if arguments.transpose_a || !arguments.transpose_b {
            return Ok(None);
        }

        let m = arguments.batch;
        let n = arguments.output_dim;
        if m != 1 && n != 1 {
            return Ok(None);
        }

        let matrix_is_rhs = n != 1;
        let transpose_matrix = if matrix_is_rhs {
            !arguments.transpose_b
        } else {
            arguments.transpose_a
        };

        let axpby_source = if arguments.c.is_some() {
            AxpbySource::C
        } else if arguments.bias.is_some() {
            AxpbySource::Bias
        } else {
            AxpbySource::None
        };

        let (do_axpby, alpha, beta, bias_stride) = match axpby_source {
            AxpbySource::None => (false, 1.0f32, 0.0f32, 0),
            AxpbySource::Bias => (true, 1.0f32, 1.0f32, 1),
            AxpbySource::C => {
                (true, arguments.alpha, arguments.beta, arguments.ldd)
            },
        };

        let output_dimension = if matrix_is_rhs {
            arguments.output_dim
        } else {
            arguments.batch
        };
        let pipeline_configuration = select_configuration(
            arguments.transpose_a,
            arguments.transpose_b,
            transpose_matrix,
            arguments.input_dim,
            output_dimension,
            false,
            do_axpby,
        );

        let input_dimension = arguments.input_dim;
        let matrix_leading_dim = if matrix_is_rhs {
            arguments.ldb
        } else {
            arguments.lda
        };

        let batch_ndim = 1i32;
        let batch_groups = arguments.batch_count.max(1);
        let batch_shape = [if arguments.batch_count > 1 {
            arguments.batch_count
        } else {
            1
        }];

        let elements_per_matrix_a =
            (arguments.batch as i64) * (arguments.lda as i64);
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

        let output_elements_per_threadgroup =
            pipeline_configuration.output_elements_per_threadgroup();
        let threadgroup_count_x =
            ((output_dimension as u32 + output_elements_per_threadgroup - 1)
                / output_elements_per_threadgroup) as u64;
        let threadgroup_count_z = batch_groups.max(1) as u64;

        let threadgroups =
            mtl_size(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup =
            pipeline_configuration.threads_per_threadgroup();

        Ok(Some(Self {
            pipeline_configuration,
            matrix_is_rhs,
            axpby_source,
            input_dimension,
            output_dimension,
            matrix_leading_dim,
            alpha,
            beta,
            batch_ndim,
            batch_shape,
            vector_batch_stride,
            matrix_batch_stride,
            bias_batch_stride,
            bias_stride,
            threadgroups,
            threads_per_threadgroup,
        }))
    }

    pub(crate) fn bias_is_fused(&self) -> bool {
        matches!(self.axpby_source, AxpbySource::Bias)
    }
}
