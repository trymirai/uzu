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
enum SplitKKernelKind {
    Partial,
    Convert,
}

pub struct SplitKGemm {
    dt: DataType,
    pipelines: HashMap<SplitKKernelKind, MTLComputePipelineState>,
}

impl SplitKGemm {
    pub fn new(dt: DataType) -> Self {
        Self {
            dt,
            pipelines: HashMap::new(),
        }
    }

    fn kernel_name(
        &self,
        kind: SplitKKernelKind,
    ) -> Result<&'static str, MTLError> {
        match (self.dt, kind) {
            (DataType::F16, SplitKKernelKind::Partial) => {
                Ok("splitk_partial_f16")
            },
            (DataType::F16, SplitKKernelKind::Convert) => {
                Ok("splitk_convert_f16")
            },
            (DataType::BF16, SplitKKernelKind::Partial) => {
                Ok("splitk_partial_bf16")
            },
            (DataType::BF16, SplitKKernelKind::Convert) => {
                Ok("splitk_convert_bf16")
            },
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                self.dt
            ))),
        }
    }

    fn pipeline(
        &mut self,
        mtl: &MTLContext,
        kind: SplitKKernelKind,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(&kind) {
            let name = self.kernel_name(kind)?;
            let ps = mtl.compute_pipeline_state(name, None)?;
            self.pipelines.insert(kind, ps);
        }
        Ok(self.pipelines.get(&kind).unwrap())
    }

    /// Heuristic from plan: use Split-K when batch_size == 1, small M*N but large K.
    pub fn should_use_splitk(
        &self,
        m: i32,
        n: i32,
        k: i32,
        batch_count: i32,
    ) -> bool {
        if batch_count != 1 {
            return false;
        }
        if m <= 0 || n <= 0 || k <= 0 {
            return false;
        }
        let tm = m / 16;
        let tn = n / 16;
        let tk = k / 16;
        (tm * tn) <= 32 && tk >= 8
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let m = args.batch as u32;
        let n = args.output_dim as u32;
        let k = args.input_dim as u32;

        // Choose split count (fixed 4-way)
        let split_k = 4u32;
        let chunk = (k + split_k - 1) / split_k;

        // Temporary accumulation buffer (float)
        let output_len = (args.batch_count as u32) * m * (args.ldd as u32);
        let accum_bytes = (output_len as usize) * std::mem::size_of::<f32>();
        let accum_buf = mtl.device.new_buffer(
            accum_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            // Zero initialize
            std::ptr::write_bytes(
                accum_buf.contents() as *mut u8,
                0,
                accum_bytes,
            );
        }

        let partial_ps = self.pipeline(mtl, SplitKKernelKind::Partial)?;

        let threads = MTLSize::new(8, 8, 1);
        let tg_x = (n as u64 + 7) / 8;
        let tg_y = (m as u64 + 7) / 8;
        let tg_z = args.batch_count as u64;
        let grid = MTLSize::new(tg_x, tg_y, tg_z);

        for split in 0..split_k {
            let k_start = split * chunk;
            let k_end = ((split + 1) * chunk).min(k);

            enc.set_compute_pipeline_state(partial_ps);
            enc.set_buffer(0, Some(args.a), 0);
            enc.set_buffer(1, Some(args.b), 0);
            enc.set_buffer(2, Some(&accum_buf), 0);

            enc.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                6,
                4,
                &(args.lda as u32) as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                7,
                4,
                &(args.ldb as u32) as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                8,
                4,
                &(args.ldd as u32) as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                9,
                4,
                &k_start as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                10,
                4,
                &k_end as *const u32 as *const std::ffi::c_void,
            );

            enc.dispatch_thread_groups(grid, threads);
        }

        // Convert to target type
        let convert_ps = self.pipeline(mtl, SplitKKernelKind::Convert)?;
        enc.set_compute_pipeline_state(convert_ps);
        enc.set_buffer(0, Some(&accum_buf), 0);
        enc.set_buffer(1, Some(args.d), 0);

        enc.set_bytes(2, 4, &m as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(
            4,
            4,
            &(args.ldd as u32) as *const u32 as *const std::ffi::c_void,
        );

        enc.dispatch_thread_groups(grid, threads);
        Ok(())
    }
}
