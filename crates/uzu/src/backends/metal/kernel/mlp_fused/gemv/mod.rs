use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState,
    MTLSize,
};

use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError, kernel::mlp::MlpActivationType},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Configuration {
    threadgroup_rows: u32,
    threadgroup_cols: u32,
    threads_per_simdgroup_row: u32,
    threads_per_simdgroup_col: u32,
    elements_per_thread_row: u32,
    elements_per_thread_col: u32,
    activation: MlpActivationType,
}

impl Configuration {
    fn output_elements_per_threadgroup(&self) -> u32 {
        self.threadgroup_rows
            * self.threads_per_simdgroup_row
            * self.elements_per_thread_row
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize::new(
            32,
            self.threadgroup_cols as u64,
            self.threadgroup_rows as u64,
        )
    }
}

fn kernel_name(
    data_type: DataType,
    config: &Configuration,
) -> Result<String, MTLError> {
    let dtype_name = match data_type {
        DataType::F16 => "float16",
        DataType::BF16 => "bfloat16",
        DataType::F32 => "float32",
        _ => {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for MLP fused GEMV: {:?}",
                data_type
            )));
        },
    };

    Ok(format!(
        "gemv_mlp_fused_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}",
        dtype_name,
        config.threadgroup_rows,
        config.threadgroup_cols,
        config.threads_per_simdgroup_row,
        config.threads_per_simdgroup_col,
        config.elements_per_thread_row,
        config.elements_per_thread_col,
    ))
}

fn select_configuration(
    hidden_dim: i32,
    activation: MlpActivationType,
) -> Configuration {
    let threadgroup_rows = if hidden_dim >= 4096 {
        8
    } else {
        4
    };

    Configuration {
        threadgroup_rows,
        threadgroup_cols: 1,
        threads_per_simdgroup_row: 1,
        threads_per_simdgroup_col: 32,
        elements_per_thread_row: 4,
        elements_per_thread_col: 4,
        activation,
    }
}

#[derive(Debug)]
pub struct Arguments<'a> {
    pub weights: &'a MTLBuffer,
    pub input: &'a MTLBuffer,
    pub input_offset: u64,
    pub output: &'a MTLBuffer,
    pub input_dim: i32,
    pub hidden_dim: i32,
    pub weights_ld: i32,
    pub batch_count: i32,
    pub activation: MlpActivationType,
}

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<Configuration, ComputePipelineState>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMV: {:?}",
                data_type
            )));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        config: Configuration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(&config) {
            let name = kernel_name(self.data_type, &config)?;

            let fcv = metal::FunctionConstantValues::new();
            let activation_val = config.activation as u32;
            fcv.set_constant_value_at_index(
                &activation_val as *const u32 as *const _,
                metal::MTLDataType::UInt,
                52,
            );

            let pipeline = context.compute_pipeline_state(&name, Some(&fcv))?;
            self.pipelines.insert(config, pipeline);
        }
        Ok(self.pipelines.get(&config).unwrap())
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: &Arguments,
    ) -> Result<(), MTLError> {
        let config = select_configuration(args.hidden_dim, args.activation);
        let pipeline = self.get_pipeline(context, config)?;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(args.weights), 0);
        encoder.set_buffer(1, Some(args.input), args.input_offset);
        encoder.set_buffer(3, Some(args.output), 0);

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &args.input_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &args.hidden_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &args.weights_ld as *const i32 as *const std::ffi::c_void,
        );

        let vector_batch_stride = args.input_dim as i64;
        let matrix_batch_stride = 0i64;
        encoder.set_bytes(
            11,
            std::mem::size_of::<i64>() as u64,
            &vector_batch_stride as *const i64 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            12,
            std::mem::size_of::<i64>() as u64,
            &matrix_batch_stride as *const i64 as *const std::ffi::c_void,
        );

        let output_elements_per_threadgroup =
            config.output_elements_per_threadgroup();
        let threadgroup_count_x =
            ((args.hidden_dim as u32 + output_elements_per_threadgroup - 1)
                / output_elements_per_threadgroup) as u64;
        let threadgroup_count_z = args.batch_count.max(1) as u64;

        let threadgroup_count =
            MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup = config.threads_per_threadgroup();

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }
}
