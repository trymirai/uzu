use std::collections::HashMap;

use crate::backends::metal::{
    ComputeCommandEncoderRef, ComputeEncoderLegacy, ComputePipelineState,
    FunctionConstantValues, FunctionConstantValuesLegacy,
};

use super::{
    DispatchDescriptor, pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, kernel::mlp_fused::common::MlpFusedArguments,
    },
};

fn kernel_name(
    data_type: DataType,
    configuration: &PipelineConfiguration,
) -> Result<String, MTLError> {
    let dtype_name = match data_type {
        DataType::F16 => "float16",
        DataType::BF16 => "bfloat16",
        DataType::F32 => "float32",
        _ => {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for MLP fused GEMV: {data_type:?}"
            )));
        },
    };

    Ok(format!(
        "gemv_mlp_fused_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}",
        dtype_name,
        configuration.threadgroup_rows,
        configuration.threadgroup_cols,
        configuration.threads_per_simdgroup_row,
        configuration.threads_per_simdgroup_col,
        configuration.elements_per_thread_row,
        configuration.elements_per_thread_col,
    ))
}

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, ComputePipelineState>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMV: {data_type:?}"
            )));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    fn get_or_compile_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(configuration) {
            let kernel_name = kernel_name(self.data_type, configuration)?;

            let function_constants = FunctionConstantValues::new();
            let activation_val = configuration.activation as u32;
            function_constants.set_constant_value_at_index(
                &activation_val as *const u32 as *const _,
                metal::MTLDataType::UInt,
                52,
            );

            let pipeline_state = context.compute_pipeline_state(
                &kernel_name,
                Some(&function_constants),
            )?;
            self.pipelines.insert(*configuration, pipeline_state);
        }
        Ok(self.pipelines.get(configuration).unwrap())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: ComputeCommandEncoderRef<'_>,
        arguments: &MlpFusedArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<(), MTLError> {
        let pipeline_state = self.get_or_compile_pipeline(
            context,
            &descriptor.pipeline_configuration,
        )?;
        encoder.set_compute_pipeline_state(pipeline_state);

        encoder.set_buffer(0, Some(arguments.weights), 0);
        encoder.set_buffer(1, Some(arguments.input), arguments.input_offset);
        encoder.set_buffer(3, Some(arguments.output), 0);

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &descriptor.input_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &descriptor.hidden_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &descriptor.weights_ld as *const i32 as *const std::ffi::c_void,
        );

        encoder.set_bytes(
            11,
            std::mem::size_of::<i64>() as u64,
            &descriptor.vector_batch_stride as *const i64
                as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            12,
            std::mem::size_of::<i64>() as u64,
            &descriptor.matrix_batch_stride as *const i64
                as *const std::ffi::c_void,
        );

        encoder.dispatch_thread_groups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );
        Ok(())
    }
}
