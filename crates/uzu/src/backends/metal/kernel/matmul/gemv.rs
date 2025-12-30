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
struct PipelineKey {
    rows_per_group: u32,
}

pub struct GemvKernel {
    dt: DataType,
    pipelines: HashMap<PipelineKey, MTLComputePipelineState>,
}

impl GemvKernel {
    pub fn new(dt: DataType) -> Self {
        Self {
            dt,
            pipelines: HashMap::new(),
        }
    }

    fn kernel_name(
        &self,
        rows: u32,
    ) -> Result<String, MTLError> {
        let base = match self.dt {
            DataType::F16 => "gemv_f16",
            DataType::BF16 => "gemv_bf16",
            _ => {
                return Err(MTLError::Generic(format!(
                    "Unsupported dtype for GEMV: {:?}",
                    self.dt
                )));
            },
        };
        Ok(format!("{base}_rows{rows}"))
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        rows: u32,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKey {
            rows_per_group: rows,
        };
        if !self.pipelines.contains_key(&key) {
            let name = self.kernel_name(rows)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.pipelines.insert(key, ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
        rows_per_group: u32,
    ) -> Result<(), MTLError> {
        let rows_per_group = rows_per_group.max(1);
        let ps = self.get_or_compile_pipeline(mtl, rows_per_group)?;
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
        let tg_y = 1;
        let tg_z = args.batch_count as u64;
        let tgs = MTLSize::new(tg_x, tg_y, tg_z);

        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }
}
