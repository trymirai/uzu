use std::{collections::HashMap, ptr::NonNull};

use super::{DispatchDescriptor, pipeline_configuration::PipelineConfiguration};
use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLContext, MTLDeviceExt, MTLError, MTLFunctionConstantValues, MTLResourceOptions,
        ProtocolObject, Retained,
        kernel::{mlp::MlpActivationType, mlp_fused::common::MlpFusedArguments},
    },
};

pub struct Kernel {
    data_type: DataType,
    partial_pipelines: HashMap<PipelineConfiguration, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    accum_pipelines: HashMap<MlpActivationType, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    up_accumulator_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    gate_accumulator_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    accumulator_buffer_bytes: usize,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16) {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused Split-K: {data_type:?}"
            )));
        }
        Ok(Self {
            data_type,
            partial_pipelines: HashMap::new(),
            accum_pipelines: HashMap::new(),
            up_accumulator_buffer: None,
            gate_accumulator_buffer: None,
            accumulator_buffer_bytes: 0,
        })
    }

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP Fused Split-K: {:?}",
                self.data_type
            ))),
        }
    }

    fn partial_kernel_name(
        &self,
        configuration: &PipelineConfiguration,
    ) -> Result<String, MTLError> {
        let in_name = self.steel_type_name()?;
        let mn_tag = if configuration.mn_aligned {
            "taligned"
        } else {
            "naligned"
        };
        let k_tag = if configuration.k_aligned {
            "taligned"
        } else {
            "naligned"
        };
        Ok(format!(
            "steel_gemm_splitk_mlp_fused_nt_{}_float32_bm{}_bn{}_bk{}_wm{}_wn{}_MN_{}_K_{}",
            in_name,
            configuration.tile.tile_rows,
            configuration.tile.tile_cols,
            configuration.tile.tile_depth,
            configuration.tile.warps_per_row,
            configuration.tile.warps_per_col,
            mn_tag,
            k_tag,
        ))
    }

    fn accum_kernel_name(&self) -> Result<String, MTLError> {
        let out_name = self.steel_type_name()?;
        Ok(format!("steel_gemm_splitk_mlp_fused_accum_{}_float32", out_name))
    }

    fn get_partial_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        if !self.partial_pipelines.contains_key(configuration) {
            let kernel_name = self.partial_kernel_name(configuration)?;
            let pipeline_state =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.partial_pipelines
                .insert(configuration.clone(), pipeline_state);
        }
        Ok(self.partial_pipelines.get(configuration).unwrap())
    }

    fn get_accum_pipeline(
        &mut self,
        context: &MTLContext,
        activation: MlpActivationType,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        if !self.accum_pipelines.contains_key(&activation) {
            let kernel_name = self.accum_kernel_name()?;
            let function_constants = MTLFunctionConstantValues::new();
            let activation_val = activation as u32;
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&activation_val).cast(),
                metal::MTLDataType::UInt,
                52,
            );
            let pipeline_state = context.compute_pipeline_state(
                &kernel_name,
                Some(&function_constants),
            )?;
            self.accum_pipelines.insert(activation, pipeline_state);
        }
        Ok(self.accum_pipelines.get(&activation).unwrap())
    }

    fn ensure_accumulator_buffers(
        &mut self,
        context: &MTLContext,
        required_bytes: usize,
    ) {
        if required_bytes <= self.accumulator_buffer_bytes
            && self.up_accumulator_buffer.is_some()
            && self.gate_accumulator_buffer.is_some()
        {
            return;
        }
        self.up_accumulator_buffer = context.device.new_buffer(
            required_bytes,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
        );
        self.gate_accumulator_buffer = context.device.new_buffer(
            required_bytes,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
        );
        self.accumulator_buffer_bytes = required_bytes;
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MlpFusedArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<(), MTLError> {
        self.ensure_accumulator_buffers(context, descriptor.accumulator_bytes);
        self.get_partial_pipeline(context, &descriptor.pipeline_configuration)?;
        self.get_accum_pipeline(
            context,
            descriptor.pipeline_configuration.activation,
        )?;

        let partial_pipeline_state = self
            .partial_pipelines
            .get(&descriptor.pipeline_configuration)
            .expect("Partial pipeline must be initialized");
        let accum_pipeline_state = self
            .accum_pipelines
            .get(&descriptor.pipeline_configuration.activation)
            .expect("Accum pipeline must be initialized");
        let up_accumulator_buffer = self
            .up_accumulator_buffer
            .as_ref()
            .expect("Up accumulator buffer must be initialized");
        let gate_accumulator_buffer = self
            .gate_accumulator_buffer
            .as_ref()
            .expect("Gate accumulator buffer must be initialized");

        encoder.set_compute_pipeline_state(partial_pipeline_state);
        encoder.set_buffer(
            Some(arguments.input),
            arguments.input_offset as usize,
            0,
        );
        encoder.set_buffer(Some(arguments.weights), 0, 1);
        encoder.set_buffer(Some(up_accumulator_buffer), 0, 2);
        encoder.set_buffer(Some(gate_accumulator_buffer), 0, 3);
        encoder.set_value(&descriptor.params, 4);
        encoder.dispatch_threadgroups(
            descriptor.partial_threadgroups,
            descriptor.partial_threads_per_threadgroup,
        );

        encoder.set_compute_pipeline_state(accum_pipeline_state);
        encoder.set_buffer(Some(up_accumulator_buffer), 0, 0);
        encoder.set_buffer(Some(gate_accumulator_buffer), 0, 1);
        encoder.set_buffer(Some(arguments.output), 0, 2);
        encoder.set_value(&descriptor.partition_count, 3);
        encoder.set_value(&descriptor.output_elements_per_partition, 4);
        encoder.set_value(&descriptor.ldd, 5);

        encoder.dispatch_threads(
            descriptor.accum_total_threads,
            descriptor.accum_threads_per_threadgroup,
        );

        Ok(())
    }
}
