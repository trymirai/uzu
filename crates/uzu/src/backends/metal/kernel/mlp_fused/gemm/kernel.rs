use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    MTLComputePipelineState, MTLFunctionConstantValues, ProtocolObject, Retained,
};

use super::{
    DispatchDescriptor, pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::{
            matmul::common::GEMMParams, mlp_fused::common::MlpFusedArguments,
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    weights_transposed: bool,
    pipelines: HashMap<PipelineConfiguration, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Kernel {
    pub fn new(
        data_type: DataType,
        weights_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMM: {data_type:?}"
            )));
        }
        Ok(Self {
            data_type,
            weights_transposed,
            pipelines: HashMap::new(),
        })
    }

    pub fn weights_transposed(&self) -> bool {
        self.weights_transposed
    }

    fn kernel_name(
        &self,
        configuration: &PipelineConfiguration,
    ) -> String {
        let type_name = match self.data_type {
            DataType::F16 => "half",
            DataType::BF16 => "bfloat",
            DataType::F32 => "float",
            _ => unreachable!(),
        };
        let transpose_char = if configuration.weights_transposed {
            "t"
        } else {
            "n"
        };
        let mn_str = if configuration.mn_aligned {
            "true"
        } else {
            "false"
        };
        let k_str = if configuration.k_aligned {
            "true"
        } else {
            "false"
        };

        format!(
            "steel_gemm_mlp_fused_{}_{}_{}_bm{}_bn{}_bk16_wm2_wn2_align_MN_{}_K_{}",
            transpose_char,
            type_name,
            type_name,
            configuration.tile.block_rows,
            configuration.tile.block_cols,
            mn_str,
            k_str
        )
    }

    fn get_or_compile_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        if !self.pipelines.contains_key(configuration) {
            let kernel_name = self.kernel_name(configuration);

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
            self.pipelines.insert(configuration.clone(), pipeline_state);
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

        encoder.set_buffer(Some(arguments.input), arguments.input_offset as usize, 0);
        encoder.set_buffer(Some(arguments.weights), 0, 1);
        encoder.set_buffer(Some(arguments.output), 0, 2);

        unsafe {
            encoder.set_bytes(
                NonNull::new(&descriptor.params as *const GEMMParams as *mut c_void).unwrap(),
                std::mem::size_of::<GEMMParams>(),
                3,
            );
        }
        unsafe {

            encoder.set_bytes(

                NonNull::new(&descriptor.hidden_dim as *const i32 as *mut c_void).unwrap(),

                std::mem::size_of::<i32>(),

                10,

            );

        }

        encoder.dispatch_threadgroups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );
        Ok(())
    }
}
