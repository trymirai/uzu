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
enum PipelineKind {
    Legacy {
        rows_per_group: u32,
    },
    Fast,
    Wide,
    Parallel,
}

pub struct GemvKernel {
    dt: DataType,
    pipelines: HashMap<PipelineKind, MTLComputePipelineState>,
}

impl GemvKernel {
    pub fn new(dt: DataType) -> Self {
        Self {
            dt,
            pipelines: HashMap::new(),
        }
    }

    fn fast_kernel_name(&self) -> Result<String, MTLError> {
        match self.dt {
            DataType::F16 => Ok("gemv_fast_f16".to_string()),
            DataType::BF16 => Ok("gemv_fast_bf16".to_string()),
            DataType::F32 => Ok("gemv_fast_f32".to_string()),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for fast GEMV: {:?}",
                self.dt
            ))),
        }
    }

    fn wide_kernel_name(&self) -> Result<String, MTLError> {
        match self.dt {
            DataType::F16 => Ok("gemv_wide_f16".to_string()),
            DataType::BF16 => Ok("gemv_wide_bf16".to_string()),
            DataType::F32 => Ok("gemv_wide_f32".to_string()),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for wide GEMV: {:?}",
                self.dt
            ))),
        }
    }

    fn parallel_kernel_name(&self) -> Result<String, MTLError> {
        match self.dt {
            DataType::F16 => Ok("gemv_parallel_f16".to_string()),
            DataType::BF16 => Ok("gemv_parallel_bf16".to_string()),
            DataType::F32 => Ok("gemv_parallel_f32".to_string()),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for parallel GEMV: {:?}",
                self.dt
            ))),
        }
    }

    fn legacy_kernel_name(
        &self,
        rows: u32,
    ) -> Result<String, MTLError> {
        let base = match self.dt {
            DataType::F16 => "gemv_f16",
            DataType::BF16 => "gemv_bf16",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported dtype for legacy GEMV: {:?}",
                    self.dt
                )));
            },
        };
        Ok(format!("{base}_rows{rows}"))
    }

    fn get_fast_pipeline(
        &mut self,
        mtl: &MTLContext,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKind::Fast;
        if !self.pipelines.contains_key(&key) {
            let name = self.fast_kernel_name()?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.pipelines.insert(key, ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn get_wide_pipeline(
        &mut self,
        mtl: &MTLContext,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKind::Wide;
        if !self.pipelines.contains_key(&key) {
            let name = self.wide_kernel_name()?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.pipelines.insert(key, ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn get_parallel_pipeline(
        &mut self,
        mtl: &MTLContext,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKind::Parallel;
        if !self.pipelines.contains_key(&key) {
            let name = self.parallel_kernel_name()?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.pipelines.insert(key, ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn get_legacy_pipeline(
        &mut self,
        mtl: &MTLContext,
        rows: u32,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKind::Legacy {
            rows_per_group: rows,
        };
        if !self.pipelines.contains_key(&key) {
            let name = self.legacy_kernel_name(rows)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.pipelines.insert(key, ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    /// Check if wide GEMV is applicable.
    /// Currently disabled as it doesn't improve MLP up pattern.
    fn should_use_wide(
        &self,
        _args: &MatmulArguments,
    ) -> bool {
        false
    }

    /// Check if parallel GEMV is applicable.
    /// Parallel kernel (8 simdgroups/TG) is best for small K with large N.
    fn should_use_parallel(
        &self,
        args: &MatmulArguments,
    ) -> bool {
        let k = args.input_dim;
        let n = args.output_dim;
        // Use parallel kernel when:
        // - K is small (< 4096) so threadgroup memory can hold a good chunk
        // - N is large (≥ 4096) so we benefit from reduced TG count
        // - K is aligned for vectorized loads
        k < 4096 && n >= 4096 && k % 4 == 0 && k >= 128
    }

    /// Check if fast GEMV is applicable.
    /// Fast kernel works best when K is large enough for vectorized loads.
    fn can_use_fast(
        &self,
        args: &MatmulArguments,
    ) -> bool {
        // Fast kernel uses BLOCK_K = 128 (32 threads * 4 elements)
        // Works well when K >= 128 and is reasonably aligned
        args.input_dim >= 128 && args.input_dim % 4 == 0
    }

    /// Encode using the fast optimized GEMV kernel.
    pub fn encode_fast(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let ps = self.get_fast_pipeline(mtl)?;
        enc.set_compute_pipeline_state(ps);

        // Bind buffers: A (weights/matrix), x (input vector), y (output)
        enc.set_buffer(0, Some(args.b), 0); // A = weights
        enc.set_buffer(1, Some(args.a), 0); // x = input
        enc.set_buffer(2, Some(args.d), 0); // y = output

        // Scalars (as i32 to match Metal kernel)
        let k = args.input_dim;
        let n = args.output_dim;
        let lda = args.ldb; // Weight matrix leading dimension
        // For batched GEMV: stride between input/output vectors is the leading dimension
        let batch_stride_x = args.lda; // stride between rows of A (input vectors)
        let batch_stride_y = args.ldd; // stride between rows of D (output vectors)

        enc.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &k as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &n as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &lda as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_x as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_y as *const i32 as *const std::ffi::c_void,
        );

        // Fast kernel: 4 rows per simdgroup, 1 simdgroup per threadgroup
        const ROWS_PER_SIMDGROUP: i32 = 4;
        let tg_x = ((n + ROWS_PER_SIMDGROUP - 1) / ROWS_PER_SIMDGROUP) as u64;
        let tg_z = args.batch_count as u64;

        let tgs = MTLSize::new(tg_x, 1, tg_z);
        let threads_per_tg = MTLSize::new(32, 1, 1); // Single simdgroup

        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    /// Encode using the wide GEMV kernel (16 rows per simdgroup).
    /// Best for small K with large N (MLP up pattern).
    pub fn encode_wide(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let ps = self.get_wide_pipeline(mtl)?;
        enc.set_compute_pipeline_state(ps);

        // Same buffer layout as fast kernel
        enc.set_buffer(0, Some(args.b), 0); // A = weights
        enc.set_buffer(1, Some(args.a), 0); // x = input
        enc.set_buffer(2, Some(args.d), 0); // y = output

        let k = args.input_dim;
        let n = args.output_dim;
        let lda = args.ldb;
        let batch_stride_x = args.lda; // stride between rows of A
        let batch_stride_y = args.ldd; // stride between rows of D

        enc.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &k as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &n as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &lda as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_x as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_y as *const i32 as *const std::ffi::c_void,
        );

        // Wide kernel: 8 rows per simdgroup
        const ROWS_PER_SIMDGROUP: i32 = 8;
        let tg_x = ((n + ROWS_PER_SIMDGROUP - 1) / ROWS_PER_SIMDGROUP) as u64;
        let tg_z = args.batch_count as u64;

        let tgs = MTLSize::new(tg_x, 1, tg_z);
        let threads_per_tg = MTLSize::new(32, 1, 1); // Single simdgroup

        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    /// Encode using the parallel GEMV kernel (8 simdgroups per threadgroup).
    /// Best for small K with large N (reduces threadgroup count).
    pub fn encode_parallel(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let ps = self.get_parallel_pipeline(mtl)?;
        enc.set_compute_pipeline_state(ps);

        // Same buffer layout as other kernels
        enc.set_buffer(0, Some(args.b), 0); // A = weights
        enc.set_buffer(1, Some(args.a), 0); // x = input
        enc.set_buffer(2, Some(args.d), 0); // y = output

        let k = args.input_dim;
        let n = args.output_dim;
        let lda = args.ldb;
        let batch_stride_x = args.lda; // stride between rows of A
        let batch_stride_y = args.ldd; // stride between rows of D

        enc.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &k as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &n as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &lda as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_x as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &batch_stride_y as *const i32 as *const std::ffi::c_void,
        );

        // Parallel kernel: 32 simdgroups per TG, each computes 1 row
        const SIMDGROUPS_PER_TG: i32 = 32;
        let tg_x = ((n + SIMDGROUPS_PER_TG - 1) / SIMDGROUPS_PER_TG) as u64;
        let tg_z = args.batch_count as u64;

        let tgs = MTLSize::new(tg_x, 1, tg_z);
        let threads_per_tg =
            MTLSize::new((SIMDGROUPS_PER_TG * 32) as u64, 1, 1);

        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    /// Encode using the legacy GEMV kernel (for fallback).
    pub fn encode_legacy(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_group: u32,
    ) -> Result<(), MTLError> {
        let rows_per_group = rows_per_group.max(1);
        let ps = self.get_legacy_pipeline(mtl, rows_per_group)?;
        enc.set_compute_pipeline_state(ps);

        // Bind buffers: matrix (B), vector (A), output (D)
        enc.set_buffer(0, Some(args.b), 0);
        enc.set_buffer(1, Some(args.a), 0);
        enc.set_buffer(2, Some(args.d), 0);

        // Scalars
        let k = args.input_dim as u32;
        let n = args.output_dim as u32;
        let ldb = args.ldb as u32;
        let lda = args.lda as u32;
        let ldd = args.ldd as u32;

        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &ldb as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &lda as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &ldd as *const u32 as *const std::ffi::c_void,
        );

        // Threadgroup sizing
        let threads_per_tg = MTLSize::new((rows_per_group * 32) as u64, 1, 1);
        let tg_x = ((n + rows_per_group - 1) / rows_per_group) as u64;
        let tg_z = args.batch_count as u64;
        let tgs = MTLSize::new(tg_x, 1, tg_z);

        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    /// Main encode entry point - automatically selects best kernel variant.
    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_group: u32,
    ) -> Result<(), MTLError> {
        // Priority: parallel > wide > fast > legacy
        if self.should_use_parallel(&args) {
            self.encode_parallel(mtl, enc, args)
        } else if self.should_use_wide(&args) {
            self.encode_wide(mtl, enc, args)
        } else if self.can_use_fast(&args) {
            self.encode_fast(mtl, enc, args)
        } else {
            self.encode_legacy(mtl, enc, args, rows_per_group)
        }
    }
}
