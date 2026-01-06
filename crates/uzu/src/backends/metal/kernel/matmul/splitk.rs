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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct SplitKGEMMParams {
    m: i32,
    n: i32,
    k: i32,
    leading_dim_a: i32,
    leading_dim_b: i32,
    leading_dim_accumulator: i32,
    tile_count_n: i32,
    tile_count_m: i32,
    partition_count: i32,
    output_elements_per_partition: i32,
    k_elements_per_partition: i32,
    gemm_k_iterations_aligned: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileConfig {
    tile_rows: i32,
    tile_cols: i32,
    tile_depth: i32,
    warps_per_row: i32,
    warps_per_col: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PartialKernelKey {
    tile_config: TileConfig,
}

pub struct SplitKGemm {
    data_type: DataType,
    transpose_a: bool,
    transpose_b: bool,
    partial_pipelines: HashMap<PartialKernelKey, MTLComputePipelineState>,
    accum_pipeline: Option<MTLComputePipelineState>,
}

impl SplitKGemm {
    pub fn new(
        data_type: DataType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Self {
        Self {
            data_type,
            transpose_a,
            transpose_b,
            partial_pipelines: HashMap::new(),
            accum_pipeline: None,
        }
    }

    fn type_suffix(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("f16"),
            DataType::BF16 => Ok("bf16"),
            DataType::F32 => Ok("f32"),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                self.data_type
            ))),
        }
    }

    fn transpose_suffix(&self) -> &'static str {
        match (self.transpose_a, self.transpose_b) {
            (false, false) => "nn",
            (false, true) => "nt",
            (true, false) => "tn",
            (true, true) => "tt",
        }
    }

    fn partial_kernel_name(
        &self,
        config: &TileConfig,
    ) -> Result<String, MTLError> {
        let type_suffix = self.type_suffix()?;
        let transpose_suffix = self.transpose_suffix();
        Ok(format!(
            "splitk_partial_{}_{}_tm{}_tn{}_tk{}_wm{}_wn{}",
            transpose_suffix,
            type_suffix,
            config.tile_rows,
            config.tile_cols,
            config.tile_depth,
            config.warps_per_row,
            config.warps_per_col
        ))
    }

    fn accum_kernel_name(&self) -> Result<String, MTLError> {
        let type_suffix = self.type_suffix()?;
        Ok(format!("splitk_accum_{}", type_suffix))
    }

    fn get_partial_pipeline(
        &mut self,
        mtl: &MTLContext,
        config: TileConfig,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PartialKernelKey {
            tile_config: config,
        };
        if !self.partial_pipelines.contains_key(&key) {
            let name = self.partial_kernel_name(&config)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.partial_pipelines.insert(key.clone(), ps);
        }
        Ok(self.partial_pipelines.get(&key).unwrap())
    }

    fn get_accum_pipeline(
        &mut self,
        mtl: &MTLContext,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        if self.accum_pipeline.is_none() {
            let name = self.accum_kernel_name()?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.accum_pipeline = Some(ps);
        }
        Ok(self.accum_pipeline.as_ref().unwrap())
    }

    fn select_tile_config(
        m: i32,
        n: i32,
    ) -> TileConfig {
        let tile_rows = if m < 40 { 16 } else { 32 };
        let tile_cols = if n < 40 { 16 } else { 32 };
        TileConfig {
            tile_rows,
            tile_cols,
            tile_depth: 16,
            warps_per_row: 2,
            warps_per_col: 2,
        }
    }

    fn compute_partition_count(k: i32) -> i32 {
        let k_tiles = k / 16;
        if k_tiles < 16 {
            2
        } else if k_tiles < 32 {
            4
        } else if k_tiles < 64 {
            8
        } else {
            16
        }
    }

    pub fn should_use_splitk(
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
        let m_tiles = m / 16;
        let n_tiles = n / 16;
        let k_tiles = k / 16;
        (m_tiles * n_tiles) <= 32 && k_tiles >= 8
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<(), MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let tile_config = Self::select_tile_config(m, n);
        let partition_count = Self::compute_partition_count(k);

        let tile_count_m = (m + tile_config.tile_rows - 1) / tile_config.tile_rows;
        let tile_count_n = (n + tile_config.tile_cols - 1) / tile_config.tile_cols;

        let gemm_k_iterations = (k / tile_config.tile_depth) / partition_count;
        let k_elements_per_partition = gemm_k_iterations * tile_config.tile_depth;
        let output_elements_per_partition = m * n;

        let accumulator_element_count =
            partition_count * output_elements_per_partition * args.batch_count;
        let accumulator_bytes =
            (accumulator_element_count as usize) * std::mem::size_of::<f32>();
        let accumulator_buffer = mtl.device.new_buffer(
            accumulator_bytes as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );

        let params = SplitKGEMMParams {
            m,
            n,
            k,
            leading_dim_a: args.lda,
            leading_dim_b: args.ldb,
            leading_dim_accumulator: n,
            tile_count_n,
            tile_count_m,
            partition_count,
            output_elements_per_partition,
            k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
        };

        let partial_ps = self.get_partial_pipeline(mtl, tile_config)?;

        enc.set_compute_pipeline_state(partial_ps);
        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(2, Some(&accumulator_buffer), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<SplitKGEMMParams>() as u64,
            &params as *const SplitKGEMMParams as *const _,
        );

        let threads_per_threadgroup = MTLSize::new(
            32,
            tile_config.warps_per_col as u64,
            tile_config.warps_per_row as u64,
        );
        let threadgroups = MTLSize::new(
            tile_count_n as u64,
            tile_count_m as u64,
            partition_count as u64,
        );
        enc.dispatch_thread_groups(threadgroups, threads_per_threadgroup);

        let accum_ps = self.get_accum_pipeline(mtl)?;
        enc.set_compute_pipeline_state(accum_ps);
        enc.set_buffer(0, Some(&accumulator_buffer), 0);
        enc.set_buffer(1, Some(args.d), 0);
        enc.set_bytes(
            2,
            4,
            &partition_count as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            3,
            4,
            &output_elements_per_partition as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            4,
            &(args.ldd) as *const i32 as *const std::ffi::c_void,
        );

        let accum_total_threads = MTLSize::new(n as u64, m as u64, 1);
        let accum_threads_per_threadgroup = MTLSize::new(
            16.min(n as u64),
            16.min(m as u64),
            1,
        );
        enc.dispatch_threads(accum_total_threads, accum_threads_per_threadgroup);

        Ok(())
    }
}
