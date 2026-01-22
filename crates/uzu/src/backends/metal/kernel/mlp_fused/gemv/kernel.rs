use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use super::{
    DispatchDescriptor, pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        ComputeCommandEncoderRef, ComputePipelineState, FunctionConstantValues,
        FunctionConstantValuesLegacy, MTLContext, MTLError,
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

        encoder.set_buffer(Some(arguments.weights), 0, 0);
        encoder.set_buffer(
            Some(arguments.input),
            arguments.input_offset as usize,
            1,
        );
        encoder.set_buffer(Some(arguments.output), 0, 3);

        unsafe {
            encoder.set_bytes(
                NonNull::new(
                    &descriptor.input_dim as *const i32 as *mut c_void,
                )
                .unwrap(),
                std::mem::size_of::<i32>(),
                4,
            );
        }
        unsafe {
            encoder.set_bytes(
                NonNull::new(
                    &descriptor.hidden_dim as *const i32 as *mut c_void,
                )
                .unwrap(),
                std::mem::size_of::<i32>(),
                5,
            );
        }
        unsafe {
            encoder.set_bytes(
                NonNull::new(
                    &descriptor.weights_ld as *const i32 as *mut c_void,
                )
                .unwrap(),
                std::mem::size_of::<i32>(),
                6,
            );
        }

        unsafe {
            encoder.set_bytes(
                NonNull::new(
                    &descriptor.vector_batch_stride as *const i64
                        as *mut c_void,
                )
                .unwrap(),
                std::mem::size_of::<i64>(),
                11,
            );
        }
        unsafe {
            encoder.set_bytes(
                NonNull::new(
                    &descriptor.matrix_batch_stride as *const i64
                        as *mut c_void,
                )
                .unwrap(),
                std::mem::size_of::<i64>(),
                12,
            );
        }

        encoder.dispatch_threadgroups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );
        Ok(())
    }
}
