use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer,
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
    Fast {
        has_bias: bool,
    },
    Tiled {
        rows_per_threadgroup: u32,
        has_bias: bool,
    },
    SplitK {
        rows_per_threadgroup: u32,
        kparts: u32,
        has_bias: bool,
    },
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

    fn fast_kernel_name(
        &self,
        has_bias: bool,
    ) -> Result<String, MTLError> {
        let base = match self.data_type {
            DataType::F16 => "gemv_fast_f16",
            DataType::BF16 => "gemv_fast_bf16",
            DataType::F32 => "gemv_fast_f32",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported data type for fast GEMV: {:?}",
                    self.data_type
                )));
            },
        };
        Ok(if has_bias {
            format!("{base}_bias")
        } else {
            base.to_string()
        })
    }

    fn tiled_kernel_name(
        &self,
        rows_per_threadgroup: u32,
        has_bias: bool,
    ) -> Result<String, MTLError> {
        let prefix = match self.data_type {
            DataType::F16 => "gemv_f16",
            DataType::BF16 => "gemv_bf16",
            DataType::F32 => "gemv_f32",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported data type for tiled GEMV: {:?}",
                    self.data_type
                )));
            },
        };
        Ok(if has_bias {
            format!("{prefix}_rows{rows_per_threadgroup}_bias")
        } else {
            format!("{prefix}_rows{rows_per_threadgroup}")
        })
    }

    fn split_k_kernel_name(
        &self,
        kparts: u32,
        has_bias: bool,
    ) -> Result<String, MTLError> {
        let dtype = match self.data_type {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported data type for split-k GEMV: {:?}",
                    self.data_type
                )));
            },
        };
        if !matches!(kparts, 2 | 4) {
            return Err(MTLError::Generic(format!(
                "Unsupported split-k kparts={kparts} (expected 2 or 4)"
            )));
        }
        let base = format!("gemv_split_k_{dtype}_rows16_kparts{kparts}");
        Ok(if has_bias {
            format!("{base}_bias")
        } else {
            base
        })
    }

    fn get_fast_pipeline(
        &mut self,
        context: &MTLContext,
        has_bias: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let variant = PipelineVariant::Fast {
            has_bias,
        };
        if !self.pipelines.contains_key(&variant) {
            let kernel_name = self.fast_kernel_name(has_bias)?;
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
        has_bias: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let variant = PipelineVariant::Tiled {
            rows_per_threadgroup,
            has_bias,
        };
        if !self.pipelines.contains_key(&variant) {
            let kernel_name =
                self.tiled_kernel_name(rows_per_threadgroup, has_bias)?;
            let pipeline =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(variant, pipeline);
        }
        Ok(self.pipelines.get(&variant).unwrap())
    }

    fn get_split_k_pipeline(
        &mut self,
        context: &MTLContext,
        kparts: u32,
        has_bias: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let variant = PipelineVariant::SplitK {
            rows_per_threadgroup: 16,
            kparts,
            has_bias,
        };
        if !self.pipelines.contains_key(&variant) {
            let kernel_name = self.split_k_kernel_name(kparts, has_bias)?;
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

    fn split_k_kparts(
        &self,
        args: &MatmulArguments,
    ) -> Option<u32> {
        let k = args.input_dim;
        let n = args.output_dim;
        if k >= 8192 && n >= 2048 {
            return Some(4);
        }
        if k >= 4096 && n >= 2048 {
            return Some(2);
        }
        None
    }

    fn is_logits_like(
        &self,
        args: &MatmulArguments,
    ) -> bool {
        let k = args.input_dim;
        let n = args.output_dim;
        n >= 50_000 || (n >= 16_384 && n >= k * 7)
    }

    fn encode_fast(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        let has_bias = bias.is_some();
        let pipeline = self.get_fast_pipeline(context, has_bias)?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.b), 0);
        encoder.set_buffer(1, Some(args.a), 0);
        if let Some(bias) = bias {
            encoder.set_buffer(2, Some(bias), 0);
            encoder.set_buffer(3, Some(args.d), 0);
        } else {
            encoder.set_buffer(2, Some(args.d), 0);
        }

        let input_dim = args.input_dim;
        let output_dim = args.output_dim;
        let weight_stride = args.ldb;
        let input_batch_stride = args.lda;
        let output_batch_stride = args.ldd;

        let base = if has_bias { 4 } else { 3 };
        encoder.set_bytes(
            base,
            std::mem::size_of::<i32>() as u64,
            &input_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 1,
            std::mem::size_of::<i32>() as u64,
            &output_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 2,
            std::mem::size_of::<i32>() as u64,
            &weight_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 3,
            std::mem::size_of::<i32>() as u64,
            &input_batch_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 4,
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
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        let rows_per_threadgroup = rows_per_threadgroup.max(1);
        let has_bias = bias.is_some();
        let pipeline = self.get_tiled_pipeline(
            context,
            rows_per_threadgroup,
            has_bias,
        )?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.b), 0);
        encoder.set_buffer(1, Some(args.a), 0);
        if let Some(bias) = bias {
            encoder.set_buffer(2, Some(bias), 0);
            encoder.set_buffer(3, Some(args.d), 0);
        } else {
            encoder.set_buffer(2, Some(args.d), 0);
        }

        let input_dim = args.input_dim as u32;
        let output_dim = args.output_dim as u32;
        let weight_stride = args.ldb as u32;
        let input_stride = args.lda as u32;
        let output_stride = args.ldd as u32;

        let base = if has_bias { 4 } else { 3 };
        encoder.set_bytes(
            base,
            std::mem::size_of::<u32>() as u64,
            &input_dim as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 1,
            std::mem::size_of::<u32>() as u64,
            &output_dim as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 2,
            std::mem::size_of::<u32>() as u64,
            &weight_stride as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 3,
            std::mem::size_of::<u32>() as u64,
            &input_stride as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 4,
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

    fn encode_split_k(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        kparts: u32,
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        let has_bias = bias.is_some();
        let pipeline = self.get_split_k_pipeline(context, kparts, has_bias)?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.b), 0);
        encoder.set_buffer(1, Some(args.a), 0);
        if let Some(bias) = bias {
            encoder.set_buffer(2, Some(bias), 0);
            encoder.set_buffer(3, Some(args.d), 0);
        } else {
            encoder.set_buffer(2, Some(args.d), 0);
        }

        let input_dim = args.input_dim;
        let output_dim = args.output_dim;
        let weight_stride = args.ldb;
        let input_batch_stride = args.lda;
        let output_batch_stride = args.ldd;

        let base = if has_bias { 4 } else { 3 };
        encoder.set_bytes(
            base,
            std::mem::size_of::<i32>() as u64,
            &input_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 1,
            std::mem::size_of::<i32>() as u64,
            &output_dim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 2,
            std::mem::size_of::<i32>() as u64,
            &weight_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 3,
            std::mem::size_of::<i32>() as u64,
            &input_batch_stride as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            base + 4,
            std::mem::size_of::<i32>() as u64,
            &output_batch_stride as *const i32 as *const std::ffi::c_void,
        );

        const ROWS_PER_THREADGROUP: i32 = 16;
        let threadgroup_count_x = ((output_dim + ROWS_PER_THREADGROUP - 1)
            / ROWS_PER_THREADGROUP) as u64;
        let threadgroup_count_z = args.batch_count as u64;

        let threads_per_threadgroup_x = match kparts {
            2 => 512,
            4 => 1024,
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported split-k kparts={kparts} (expected 2 or 4)"
                )));
            },
        };

        let threadgroup_count =
            MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup =
            MTLSize::new(threads_per_threadgroup_x, 1, 1);

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }

    fn encode_inner(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_threadgroup: u32,
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        if self.is_logits_like(&args) {
            if args.input_dim >= 8192 {
                if let Some(kparts) = self.split_k_kparts(&args) {
                    return self.encode_split_k(
                        context,
                        encoder,
                        args,
                        kparts,
                        bias,
                    );
                }
            }
            return self.encode_tiled(context, encoder, args, 16, bias);
        }
        if let Some(kparts) = self.split_k_kparts(&args) {
            return self.encode_split_k(context, encoder, args, kparts, bias);
        }
        if self.can_use_fast_kernel(&args) {
            self.encode_fast(context, encoder, args, bias)
        } else {
            self.encode_tiled(context, encoder, args, rows_per_threadgroup, bias)
        }
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_threadgroup: u32,
    ) -> Result<(), MTLError> {
        self.encode_inner(context, encoder, args, rows_per_threadgroup, None)
    }

    pub fn encode_with_bias(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_threadgroup: u32,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        self.encode_inner(
            context,
            encoder,
            args,
            rows_per_threadgroup,
            Some(bias),
        )
    }
}
