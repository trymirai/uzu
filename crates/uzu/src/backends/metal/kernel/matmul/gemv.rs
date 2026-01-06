use std::collections::HashMap;

use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use super::arguments::MatmulArguments;
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GemvConfiguration {
    simdgroups_per_row: u32,
    simdgroups_per_reduction: u32,
    threads_per_simdgroup_row: u32,
    threads_per_simdgroup_col: u32,
    elements_per_thread_row: u32,
    elements_per_thread_col: u32,
    has_bias: bool,
}

impl GemvConfiguration {
    fn output_elements_per_threadgroup(&self) -> u32 {
        let total_threads_row = self.simdgroups_per_row * self.threads_per_simdgroup_row;
        total_threads_row * self.elements_per_thread_row
    }

    fn total_threads_per_threadgroup(&self) -> u32 {
        self.simdgroups_per_row * self.simdgroups_per_reduction * 32
    }
}

fn select_gemv_configuration(input_dimension: i32, output_dimension: i32, has_bias: bool) -> GemvConfiguration {
    let is_small_input_dimension = input_dimension <= 64;
    let is_large_input_relative_to_output = input_dimension >= 16 * output_dimension;
    let is_large_output_dimension = output_dimension >= 4096;
    let is_very_small_output_dimension = output_dimension < 4;

    if is_small_input_dimension {
        GemvConfiguration {
            simdgroups_per_row: 1,
            simdgroups_per_reduction: 1,
            threads_per_simdgroup_row: 8,
            threads_per_simdgroup_col: 4,
            elements_per_thread_row: if is_very_small_output_dimension { 1 } else { 4 },
            elements_per_thread_col: 4,
            has_bias,
        }
    } else if is_large_input_relative_to_output {
        GemvConfiguration {
            simdgroups_per_row: 1,
            simdgroups_per_reduction: 8,
            threads_per_simdgroup_row: 1,
            threads_per_simdgroup_col: 32,
            elements_per_thread_row: if is_very_small_output_dimension { 1 } else { 4 },
            elements_per_thread_col: 4,
            has_bias,
        }
    } else {
        GemvConfiguration {
            simdgroups_per_row: if is_large_output_dimension { 8 } else { 4 },
            simdgroups_per_reduction: 1,
            threads_per_simdgroup_row: 1,
            threads_per_simdgroup_col: 32,
            elements_per_thread_row: if is_very_small_output_dimension { 1 } else { 4 },
            elements_per_thread_col: 4,
            has_bias,
        }
    }
}

fn kernel_name_for_configuration(data_type: DataType, configuration: &GemvConfiguration) -> Result<String, MTLError> {
    let type_name = match data_type {
        DataType::F16 => "half",
        DataType::BF16 => "bfloat",
        DataType::F32 => "float",
        _ => {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for GEMV: {:?}",
                data_type
            )));
        }
    };

    let base_name = format!(
        "gemv_{}_rows{}_reduction{}_elements{}",
        type_name,
        configuration.simdgroups_per_row,
        configuration.simdgroups_per_reduction,
        configuration.elements_per_thread_row
    );

    Ok(if configuration.has_bias {
        format!("{}_bias", base_name)
    } else {
        base_name
    })
}

pub struct GemvKernel {
    data_type: DataType,
    pipelines: HashMap<GemvConfiguration, ComputePipelineState>,
}

impl GemvKernel {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            pipelines: HashMap::new(),
        }
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: GemvConfiguration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(&configuration) {
            let kernel_name = kernel_name_for_configuration(self.data_type, &configuration)?;
            let pipeline = context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(configuration, pipeline);
        }
        Ok(self.pipelines.get(&configuration).unwrap())
    }

    fn encode_with_configuration(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: MatmulArguments,
        configuration: GemvConfiguration,
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        let pipeline = self.get_pipeline(context, configuration)?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(arguments.b), 0);
        encoder.set_buffer(1, Some(arguments.a), 0);
        if let Some(bias_buffer) = bias {
            encoder.set_buffer(2, Some(bias_buffer), 0);
        }
        encoder.set_buffer(3, Some(arguments.d), 0);

        let input_dimension = arguments.input_dim;
        let output_dimension = arguments.output_dim;
        let weight_row_stride = arguments.ldb;
        let input_batch_stride = arguments.lda;
        let output_batch_stride = arguments.ldd;

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &input_dimension as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &output_dimension as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &weight_row_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &input_batch_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<i32>() as u64,
            &output_batch_stride as *const i32 as *const std::ffi::c_void,
        );

        let output_elements_per_threadgroup = configuration.output_elements_per_threadgroup();
        let threadgroup_count_x =
            ((output_dimension as u32 + output_elements_per_threadgroup - 1) / output_elements_per_threadgroup) as u64;
        let threadgroup_count_z = arguments.batch_count as u64;

        let threadgroup_count = MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup = MTLSize::new(configuration.total_threads_per_threadgroup() as u64, 1, 1);

        encoder.dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: MatmulArguments,
        _rows_per_threadgroup: u32,
    ) -> Result<(), MTLError> {
        let configuration = select_gemv_configuration(
            arguments.input_dim,
            arguments.output_dim,
            false,
        );
        self.encode_with_configuration(context, encoder, arguments, configuration, None)
    }

    pub fn encode_with_bias(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: MatmulArguments,
        _rows_per_threadgroup: u32,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        let configuration = select_gemv_configuration(
            arguments.input_dim,
            arguments.output_dim,
            true,
        );
        self.encode_with_configuration(context, encoder, arguments, configuration, Some(bias))
    }
}
