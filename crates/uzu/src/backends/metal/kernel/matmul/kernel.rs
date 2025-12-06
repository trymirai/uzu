use std::collections::HashMap;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLSize,
};

use super::{
    arguments::MatmulArguments, pipeline::PipelineKey,
    shared_types::GEMMParams, transpose::transpose_configuration,
};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

pub struct MatmulKernel {
    dt: DataType,
    transpose_a: bool,
    transpose_b: bool,
    bm: i32,
    bn: i32,
    bk: i32,
    wm: u64,
    wn: u64,
    pipelines: HashMap<PipelineKey, MTLComputePipelineState>,
}

impl MatmulKernel {
    fn dtype_suffix(dt: DataType) -> &'static str {
        match dt {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
            _ => unreachable!(),
        }
    }

    fn kernel_name(&self) -> String {
        let t = Self::dtype_suffix(self.dt);
        let cfg = transpose_configuration(self.transpose_a, self.transpose_b);
        let tcfg = cfg.as_str();
        // Shapes we currently instantiate in matmul.metal: bm64/bn64/bk16/wm2/wn2
        format!("gemm_{}_{}_bm64_bn64_bk16_wm2_wn2", tcfg, t)
    }

    pub fn new(
        _mtl: &MTLContext,
        dt: DataType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(dt, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MatmulKernel: {dt:?}"
            )));
        }

        Ok(Self {
            dt,
            transpose_a,
            transpose_b,
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            pipelines: HashMap::new(),
        })
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        align_m: bool,
        align_n: bool,
        align_k: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKey {
            name: name.to_string(),
            align_m,
            align_n,
            align_k,
        };
        if !self.pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
            // Base specializations
            let has_batch = false;
            let use_out_source = false;
            let do_axpby = false;
            fcv.set_constant_value_at_index(
                &has_batch as *const bool as *const _,
                metal::MTLDataType::Bool,
                10,
            );
            fcv.set_constant_value_at_index(
                &use_out_source as *const bool as *const _,
                metal::MTLDataType::Bool,
                100,
            );
            fcv.set_constant_value_at_index(
                &do_axpby as *const bool as *const _,
                metal::MTLDataType::Bool,
                110,
            );
            fcv.set_constant_value_at_index(
                &align_m as *const bool as *const _,
                metal::MTLDataType::Bool,
                200,
            );
            fcv.set_constant_value_at_index(
                &align_n as *const bool as *const _,
                metal::MTLDataType::Bool,
                201,
            );
            fcv.set_constant_value_at_index(
                &align_k as *const bool as *const _,
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{}_am{}_an{}_ak{}",
                name, align_m as u8, align_n as u8, align_k as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                name,
                Some(&fcv),
            )?;
            self.pipelines.insert(key.clone(), ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let kname = self.kernel_name();

        // M = batch, N = output_dim, K = input_dim
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let am = (m % self.bm) == 0;
        let an = (n % self.bn) == 0;
        let ak = (k % self.bk) == 0;
        let ps = self.get_or_compile_pipeline(mtl, &kname, am, an, ak)?;

        enc.set_compute_pipeline_state(ps);

        // Set buffers (C is elided since use_out_source=false)
        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(3, Some(args.d), 0);

        // Params
        let tiles_n = (n + self.bn - 1) / self.bn;
        let tiles_m = (m + self.bm - 1) / self.bm;
        let params = GEMMParams {
            batch: m,
            output_dim: n,
            input_dim: k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
            tiles_n,
            tiles_m,
            batch_stride_a: (args.lda as i64) * (k as i64),
            batch_stride_b: (args.ldb as i64) * (n as i64),
            batch_stride_d: (args.ldd as i64) * (n as i64),
            swizzle_log: 0,
            gemm_k_iterations_aligned: k / self.bk,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        // Threadgroup sizing
        let threads_per_tg = MTLSize::new(32, self.wn, self.wm);
        let tg_x = ((n as i64 + self.bn as i64 - 1) / self.bn as i64) as u64;
        let tg_y = ((m as i64 + self.bm as i64 - 1) / self.bm as i64) as u64;
        let tg_z = args.batch_count as u64;
        let tgs = MTLSize::new(tg_x, tg_y, tg_z);
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }
}
