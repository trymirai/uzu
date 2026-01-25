use super::pipeline_configuration::{
    PipelineConfiguration, select_configuration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, MTLSize,
        kernel::mlp_fused::common::MlpFusedArguments,
    },
};

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) input_dim: i32,
    pub(crate) hidden_dim: i32,
    pub(crate) weights_ld: i32,
    pub(crate) vector_batch_stride: i64,
    pub(crate) matrix_batch_stride: i64,
    pub(crate) threadgroups: MTLSize,
    pub(crate) threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MTLContext,
        data_type: DataType,
        arguments: &MlpFusedArguments,
    ) -> Result<Option<Self>, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMV: {data_type:?}"
            )));
        }

        if arguments.batch != 1 {
            return Ok(None);
        }

        let pipeline_configuration =
            select_configuration(arguments.hidden_dim, arguments.activation);

        let output_elements_per_threadgroup =
            pipeline_configuration.output_elements_per_threadgroup();
        let threadgroup_count_x =
            ((arguments.hidden_dim as u32 + output_elements_per_threadgroup
                - 1)
                / output_elements_per_threadgroup) as u64;
        let threadgroup_count_z = arguments.batch_count.max(1) as u64;

        let threadgroups = MTLSize::new(
            threadgroup_count_x as usize,
            1,
            threadgroup_count_z as usize,
        );
        let threads_per_threadgroup =
            pipeline_configuration.threads_per_threadgroup();

        Ok(Some(Self {
            pipeline_configuration,
            input_dim: arguments.input_dim,
            hidden_dim: arguments.hidden_dim,
            weights_ld: arguments.ldb,
            vector_batch_stride: arguments.input_dim as i64,
            matrix_batch_stride: 0,
            threadgroups,
            threads_per_threadgroup,
        }))
    }
}
