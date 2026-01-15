use metal::MTLSize;

use super::{
    pipeline_configuration::PipelineConfiguration,
    tile_configuration::select_tile_configuration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::{
            matmul::common::GEMMSpiltKMlpFusedParams,
            mlp_fused::common::MlpFusedArguments,
        },
    },
};

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) params: GEMMSpiltKMlpFusedParams,
    pub(crate) partition_count: i32,
    pub(crate) output_elements_per_partition: i32,
    pub(crate) accumulator_bytes: usize,
    pub(crate) ldd: i32,
    pub(crate) partial_threadgroups: MTLSize,
    pub(crate) partial_threads_per_threadgroup: MTLSize,
    pub(crate) accum_total_threads: MTLSize,
    pub(crate) accum_threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MTLContext,
        data_type: DataType,
        arguments: &MlpFusedArguments,
    ) -> Result<Option<Self>, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16) {
            return Ok(None);
        }

        if !should_use_splitk(
            arguments.batch,
            arguments.hidden_dim,
            arguments.input_dim,
        ) {
            return Ok(None);
        }

        let batch = arguments.batch;
        let hidden_dim = arguments.hidden_dim;
        let input_dim = arguments.input_dim;

        let tile = select_tile_configuration(batch, hidden_dim);
        let partition_count = compute_partition_count(input_dim);
        let mn_aligned =
            (batch % tile.tile_rows) == 0 && (hidden_dim % tile.tile_cols) == 0;
        let k_aligned = (input_dim % tile.tile_depth) == 0;

        let pipeline_configuration = PipelineConfiguration {
            tile,
            mn_aligned,
            k_aligned,
            activation: arguments.activation,
        };

        let tile_count_m = (batch + tile.tile_rows - 1) / tile.tile_rows;
        let tile_count_n = (hidden_dim + tile.tile_cols - 1) / tile.tile_cols;

        let gemm_k_iterations = (input_dim / tile.tile_depth) / partition_count;
        let k_elements_per_partition = gemm_k_iterations * tile.tile_depth;
        let output_elements_per_partition = batch * hidden_dim;

        let accumulator_element_count =
            partition_count * output_elements_per_partition;
        let accumulator_bytes =
            (accumulator_element_count as usize) * std::mem::size_of::<f32>();

        let params = GEMMSpiltKMlpFusedParams {
            M: batch,
            N: hidden_dim,
            K: input_dim,
            lda: arguments.lda,
            ldb: arguments.ldb,
            ldc: hidden_dim,
            tiles_n: tile_count_n,
            tiles_m: tile_count_m,
            split_k_partitions: partition_count,
            split_k_partition_stride: output_elements_per_partition,
            split_k_partition_size: k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
            hidden_dim,
        };

        let partial_threads_per_threadgroup = MTLSize::new(
            32,
            tile.warps_per_col as u64,
            tile.warps_per_row as u64,
        );
        let partial_threadgroups = MTLSize::new(
            tile_count_n as u64,
            tile_count_m as u64,
            partition_count as u64,
        );

        let accum_total_threads =
            MTLSize::new(hidden_dim as u64, batch as u64, 1);
        let accum_threads_per_threadgroup =
            MTLSize::new(16.min(hidden_dim as u64), 16.min(batch as u64), 1);

        Ok(Some(Self {
            pipeline_configuration,
            params,
            partition_count,
            output_elements_per_partition,
            accumulator_bytes,
            ldd: arguments.ldd,
            partial_threadgroups,
            partial_threads_per_threadgroup,
            accum_total_threads,
            accum_threads_per_threadgroup,
        }))
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
    batch: i32,
    hidden_dim: i32,
    input_dim: i32,
) -> bool {
    if batch != 1 {
        return false;
    }
    if hidden_dim <= 0 || input_dim <= 0 {
        return false;
    }
    let n_tiles = hidden_dim / 16;
    let k_tiles = input_dim / 16;
    n_tiles <= 32 && k_tiles >= 8
}
