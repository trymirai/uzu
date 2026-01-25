use std::{collections::HashMap, ptr::NonNull};

use super::{DispatchDescriptor, pipeline_configuration::PipelineConfiguration};
use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext,
        MTLError, MTLFunctionConstantValues, ProtocolObject, Retained,
        kernel::mlp_fused::common::MlpFusedArguments,
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
    pipelines: HashMap<PipelineConfiguration, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        if !self.pipelines.contains_key(configuration) {
            let kernel_name = kernel_name(self.data_type, configuration)?;

            let function_constants = MTLFunctionConstantValues::new();
            let activation_val = configuration.activation as u32;
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&activation_val).cast(),
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
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MlpFusedArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<(), MTLError> {
        let pipeline_state = self.get_or_compile_pipeline(
            context,
            &descriptor.pipeline_configuration,
        )?;
        encoder.set_compute_pipeline_state(pipeline_state);

        encoder.set_buffer(Some(arguments.weights), 0, 0);
        encoder.set_buffer(
            Some(arguments.input),
            arguments.input_offset as usize,
            1,
        );
        encoder.set_buffer(Some(arguments.output), 0, 3);

        encoder.set_value(&descriptor.input_dim, 4);
        encoder.set_value(&descriptor.hidden_dim, 5);
        encoder.set_value(&descriptor.weights_ld, 6);
        encoder.set_value(&descriptor.vector_batch_stride, 11);
        encoder.set_value(&descriptor.matrix_batch_stride, 12);

        encoder.dispatch_threadgroups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );
        Ok(())
    }
}
