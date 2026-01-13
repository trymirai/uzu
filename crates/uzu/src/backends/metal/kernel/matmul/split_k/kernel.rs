use std::collections::HashMap;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLSize,
};

use super::{
    pipeline_configuration::PipelineConfiguration,
    tile_configuration::{TileConfiguration, select_tile_configuration},
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::matmul::common::{GEMMSpiltKParams as SplitKGEMMParams, MatmulArguments},
    },
};

pub struct Kernel {
    data_type: DataType,
    transpose_a: bool,
    transpose_b: bool,
    partial_pipelines: HashMap<PipelineConfiguration, MTLComputePipelineState>,
    accum_pipeline: Option<MTLComputePipelineState>,
}

impl Kernel {
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

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            DataType::F32 => Ok("float32"),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                self.data_type
            ))),
        }
    }

    fn splitk_partial_out_name(&self) -> &'static str {
        "float32"
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
        tile: &TileConfiguration,
        mn_aligned: bool,
        k_aligned: bool,
    ) -> Result<String, MTLError> {
        let in_name = self.steel_type_name()?;
        let out_name = self.splitk_partial_out_name();
        let transpose_suffix = self.transpose_suffix();
        let mn_tag = if mn_aligned { "taligned" } else { "naligned" };
        let k_tag = if k_aligned { "taligned" } else { "naligned" };
        Ok(format!(
            "steel_gemm_splitk_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}_MN_{}_K_{}",
            transpose_suffix,
            in_name,
            out_name,
            tile.tile_rows,
            tile.tile_cols,
            tile.tile_depth,
            tile.warps_per_row,
            tile.warps_per_col,
            mn_tag,
            k_tag,
        ))
    }

    fn accum_kernel_name(&self) -> Result<String, MTLError> {
        let out_name = self.steel_type_name()?;
        Ok(format!(
            "steel_gemm_splitk_accum_{}_{}",
            out_name,
            self.splitk_partial_out_name()
        ))
    }

    fn get_partial_pipeline(
        &mut self,
        mtl: &MTLContext,
        config: &PipelineConfiguration,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        if !self.partial_pipelines.contains_key(config) {
            let name =
                self.partial_kernel_name(&config.tile, config.mn_aligned, config.k_aligned)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.partial_pipelines.insert(config.clone(), ps);
        }
        Ok(self.partial_pipelines.get(config).unwrap())
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
        args: &MatmulArguments,
    ) -> Result<(), MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let tile = select_tile_configuration(m, n);
        let partition_count = Self::compute_partition_count(k);
        let mn_aligned =
            (m % tile.tile_rows) == 0 && (n % tile.tile_cols) == 0;
        let k_aligned = (k % tile.tile_depth) == 0;

        let config = PipelineConfiguration {
            tile,
            mn_aligned,
            k_aligned,
        };

        let tile_count_m = (m + tile.tile_rows - 1) / tile.tile_rows;
        let tile_count_n = (n + tile.tile_cols - 1) / tile.tile_cols;

        let gemm_k_iterations = (k / tile.tile_depth) / partition_count;
        let k_elements_per_partition = gemm_k_iterations * tile.tile_depth;
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
            M: m,
            N: n,
            K: k,
            lda: args.lda,
            ldb: args.ldb,
            ldc: n,
            tiles_n: tile_count_n,
            tiles_m: tile_count_m,
            split_k_partitions: partition_count,
            split_k_partition_stride: output_elements_per_partition,
            split_k_partition_size: k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
        };

        let partial_ps = self.get_partial_pipeline(mtl, &config)?;

        enc.set_compute_pipeline_state(partial_ps);
        enc.set_buffer(0, Some(args.a), args.a_offset);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(2, Some(&accumulator_buffer), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<SplitKGEMMParams>() as u64,
            &params as *const SplitKGEMMParams as *const _,
        );

        let threads_per_threadgroup = MTLSize::new(
            32,
            tile.warps_per_col as u64,
            tile.warps_per_row as u64,
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
        let accum_threads_per_threadgroup =
            MTLSize::new(16.min(n as u64), 16.min(m as u64), 1);
        enc.dispatch_threads(accum_total_threads, accum_threads_per_threadgroup);

        Ok(())
    }
}
