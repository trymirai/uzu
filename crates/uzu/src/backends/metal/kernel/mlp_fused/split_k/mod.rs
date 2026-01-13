use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::{
            matmul::common::GEMMSpiltKMlpFusedParams, mlp::MlpActivationType,
        },
    },
};

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
    mn_aligned: bool,
    k_aligned: bool,
}

#[derive(Debug)]
pub struct Arguments<'a> {
    pub input: &'a MTLBuffer,
    pub input_offset: u64,
    pub weights: &'a MTLBuffer,
    pub output: &'a MTLBuffer,
    pub batch: i32,
    pub input_dim: i32,
    pub hidden_dim: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    pub activation: MlpActivationType,
}

pub struct Kernel {
    data_type: DataType,
    partial_pipelines: HashMap<PartialKernelKey, MTLComputePipelineState>,
    accum_pipelines: HashMap<MlpActivationType, MTLComputePipelineState>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            partial_pipelines: HashMap::new(),
            accum_pipelines: HashMap::new(),
        }
    }

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP Fused Split-K: {:?}",
                self.data_type
            ))),
        }
    }

    fn partial_kernel_name(
        &self,
        config: &TileConfig,
        mn_aligned: bool,
        k_aligned: bool,
    ) -> Result<String, MTLError> {
        let in_name = self.steel_type_name()?;
        let mn_tag = if mn_aligned {
            "taligned"
        } else {
            "naligned"
        };
        let k_tag = if k_aligned {
            "taligned"
        } else {
            "naligned"
        };
        Ok(format!(
            "steel_gemm_splitk_mlp_fused_nt_{}_float32_bm{}_bn{}_bk{}_wm{}_wn{}_MN_{}_K_{}",
            in_name,
            config.tile_rows,
            config.tile_cols,
            config.tile_depth,
            config.warps_per_row,
            config.warps_per_col,
            mn_tag,
            k_tag,
        ))
    }

    fn accum_kernel_name(&self) -> Result<String, MTLError> {
        let out_name = self.steel_type_name()?;
        Ok(format!("steel_gemm_splitk_mlp_fused_accum_{}_float32", out_name))
    }

    fn get_partial_pipeline(
        &mut self,
        mtl: &MTLContext,
        config: TileConfig,
        mn_aligned: bool,
        k_aligned: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PartialKernelKey {
            tile_config: config,
            mn_aligned,
            k_aligned,
        };
        if !self.partial_pipelines.contains_key(&key) {
            let name =
                self.partial_kernel_name(&config, mn_aligned, k_aligned)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.partial_pipelines.insert(key.clone(), ps);
        }
        Ok(self.partial_pipelines.get(&key).unwrap())
    }

    fn get_accum_pipeline(
        &mut self,
        mtl: &MTLContext,
        activation: MlpActivationType,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        if !self.accum_pipelines.contains_key(&activation) {
            let name = self.accum_kernel_name()?;
            let fcv = metal::FunctionConstantValues::new();
            let activation_val = activation as u32;
            fcv.set_constant_value_at_index(
                &activation_val as *const u32 as *const _,
                metal::MTLDataType::UInt,
                52,
            );
            let ps = mtl.compute_pipeline_state(&name, Some(&fcv))?;
            self.accum_pipelines.insert(activation, ps);
        }
        Ok(self.accum_pipelines.get(&activation).unwrap())
    }

    fn select_tile_config(
        m: i32,
        n: i32,
    ) -> TileConfig {
        let tile_rows = if m < 40 {
            16
        } else {
            32
        };
        let tile_cols = if n < 40 {
            16
        } else {
            32
        };
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
        hidden_dim: i32,
        k: i32,
    ) -> bool {
        if m != 1 {
            return false;
        }
        if hidden_dim <= 0 || k <= 0 {
            return false;
        }
        let n_tiles = hidden_dim / 16;
        let k_tiles = k / 16;
        n_tiles <= 32 && k_tiles >= 8
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &Arguments,
    ) -> Result<(), MTLError> {
        let m = args.batch;
        let hidden_dim = args.hidden_dim;
        let k = args.input_dim;

        let tile_config = Self::select_tile_config(m, hidden_dim);
        let partition_count = Self::compute_partition_count(k);
        let mn_aligned = (m % tile_config.tile_rows) == 0
            && (hidden_dim % tile_config.tile_cols) == 0;
        let k_aligned = (k % tile_config.tile_depth) == 0;

        let tile_count_m =
            (m + tile_config.tile_rows - 1) / tile_config.tile_rows;
        let tile_count_n =
            (hidden_dim + tile_config.tile_cols - 1) / tile_config.tile_cols;

        let gemm_k_iterations = (k / tile_config.tile_depth) / partition_count;
        let k_elements_per_partition =
            gemm_k_iterations * tile_config.tile_depth;
        let output_elements_per_partition = m * hidden_dim;

        let accumulator_element_count =
            partition_count * output_elements_per_partition;
        let accumulator_bytes =
            (accumulator_element_count as usize) * std::mem::size_of::<f32>();

        let up_accumulator_buffer = mtl.device.new_buffer(
            accumulator_bytes as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let gate_accumulator_buffer = mtl.device.new_buffer(
            accumulator_bytes as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );

        let params = GEMMSpiltKMlpFusedParams {
            M: m,
            N: hidden_dim,
            K: k,
            lda: args.lda,
            ldb: args.ldb,
            ldc: hidden_dim,
            tiles_n: tile_count_n,
            tiles_m: tile_count_m,
            split_k_partitions: partition_count,
            split_k_partition_stride: output_elements_per_partition,
            split_k_partition_size: k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
            hidden_dim,
        };

        let partial_ps =
            self.get_partial_pipeline(mtl, tile_config, mn_aligned, k_aligned)?;

        enc.set_compute_pipeline_state(partial_ps);
        enc.set_buffer(0, Some(args.input), args.input_offset);
        enc.set_buffer(1, Some(args.weights), 0);
        enc.set_buffer(2, Some(&up_accumulator_buffer), 0);
        enc.set_buffer(3, Some(&gate_accumulator_buffer), 0);
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMSpiltKMlpFusedParams>() as u64,
            &params as *const GEMMSpiltKMlpFusedParams as *const _,
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

        let accum_ps = self.get_accum_pipeline(mtl, args.activation)?;
        enc.set_compute_pipeline_state(accum_ps);
        enc.set_buffer(0, Some(&up_accumulator_buffer), 0);
        enc.set_buffer(1, Some(&gate_accumulator_buffer), 0);
        enc.set_buffer(2, Some(args.output), 0);
        enc.set_bytes(
            3,
            4,
            &partition_count as *const i32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            4,
            4,
            &output_elements_per_partition as *const i32
                as *const std::ffi::c_void,
        );
        enc.set_bytes(5, 4, &args.ldd as *const i32 as *const std::ffi::c_void);

        let accum_total_threads = MTLSize::new(hidden_dim as u64, m as u64, 1);
        let accum_threads_per_threadgroup =
            MTLSize::new(16.min(hidden_dim as u64), 16.min(m as u64), 1);
        enc.dispatch_threads(
            accum_total_threads,
            accum_threads_per_threadgroup,
        );

        Ok(())
    }
}
