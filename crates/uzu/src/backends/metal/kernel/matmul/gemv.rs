use std::collections::HashMap;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLSize,
};

use super::arguments::MatmulArguments;
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineVariant {
    Tiled {
        rows_per_threadgroup: u32,
    },
    Fast,
}

pub struct GemvKernel {
    data_type: DataType,
    pipelines: HashMap<PipelineVariant, MTLComputePipelineState>,
}

impl GemvKernel {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            pipelines: HashMap::new(),
        }
    }

    fn fast_kernel_name(&self) -> Result<String, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("gemv_fast_f16".to_string()),
            DataType::BF16 => Ok("gemv_fast_bf16".to_string()),
            DataType::F32 => Ok("gemv_fast_f32".to_string()),
            _ => Err(MTLError::Generic(format!(
                "Unsupported data type for fast GEMV: {:?}",
                self.data_type
            ))),
        }
    }

    fn tiled_kernel_name(
        &self,
        rows_per_threadgroup: u32,
    ) -> Result<String, MTLError> {
        let prefix = match self.data_type {
            DataType::F16 => "gemv_f16",
            DataType::BF16 => "gemv_bf16",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported data type for tiled GEMV: {:?}",
                    self.data_type
                )));
            },
        };
        Ok(format!("{prefix}_rows{rows_per_threadgroup}"))
    }

    fn get_fast_pipeline(
        &mut self,
        context: &MTLContext,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let variant = PipelineVariant::Fast;
        if !self.pipelines.contains_key(&variant) {
            let kernel_name = self.fast_kernel_name()?;
            let pipeline =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(variant, pipeline);
        }
        Ok(self.pipelines.get(&variant).unwrap())
    }

    fn get_tiled_pipeline(
        &mut self,
        context: &MTLContext,
        rows_per_threadgroup: u32,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let variant = PipelineVariant::Tiled {
            rows_per_threadgroup,
        };
        if !self.pipelines.contains_key(&variant) {
            let kernel_name = self.tiled_kernel_name(rows_per_threadgroup)?;
            let pipeline =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(variant, pipeline);
        }
        Ok(self.pipelines.get(&variant).unwrap())
    }

    fn can_use_fast_kernel(
        &self,
        args: &MatmulArguments,
    ) -> bool {
        args.input_dim >= 128 && args.input_dim % 4 == 0
    }

    fn encode_fast(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let pipeline = self.get_fast_pipeline(context)?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.b), 0);
        encoder.set_buffer(1, Some(args.a), 0);
        encoder.set_buffer(2, Some(args.d), 0);

        let input_dim = args.input_dim;
        let output_dim = args.output_dim;
        let weight_stride = args.ldb;
        let input_batch_stride = args.lda;
        let output_batch_stride = args.ldd;

        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &input_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &output_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &weight_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &input_batch_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &output_batch_stride as *const i32 as *const std::ffi::c_void,
        );

        const ROWS_PER_SIMDGROUP: i32 = 4;
        let threadgroup_count_x =
            ((output_dim + ROWS_PER_SIMDGROUP - 1) / ROWS_PER_SIMDGROUP) as u64;
        let threadgroup_count_z = args.batch_count as u64;

        let threadgroup_count =
            MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup = MTLSize::new(32, 1, 1);

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }

    fn encode_tiled(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_threadgroup: u32,
    ) -> Result<(), MTLError> {
        let rows_per_threadgroup = rows_per_threadgroup.max(1);
        let pipeline =
            self.get_tiled_pipeline(context, rows_per_threadgroup)?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.b), 0);
        encoder.set_buffer(1, Some(args.a), 0);
        encoder.set_buffer(2, Some(args.d), 0);

        let input_dim = args.input_dim as u32;
        let output_dim = args.output_dim as u32;
        let weight_stride = args.ldb as u32;
        let input_stride = args.lda as u32;
        let output_stride = args.ldd as u32;

        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &input_dim as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &output_dim as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &weight_stride as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &input_stride as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &output_stride as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup =
            MTLSize::new((rows_per_threadgroup * 32) as u64, 1, 1);
        let threadgroup_count_x = ((output_dim + rows_per_threadgroup - 1)
            / rows_per_threadgroup) as u64;
        let threadgroup_count_z = args.batch_count as u64;
        let threadgroup_count =
            MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_threadgroup: u32,
    ) -> Result<(), MTLError> {
        if self.can_use_fast_kernel(&args) {
            self.encode_fast(context, encoder, args)
        } else {
            self.encode_tiled(context, encoder, args, rows_per_threadgroup)
        }
    }
}
