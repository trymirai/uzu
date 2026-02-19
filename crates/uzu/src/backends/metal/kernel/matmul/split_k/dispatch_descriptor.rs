use metal::MTLSize;

use super::{
    pipeline_configuration::PipelineConfiguration,
    tile_configuration::{TileConfiguration, select_tile_configuration},
};
use crate::{
    DataType,
    backends::metal::{
        context::MetalContext,
        error::MetalError,
        kernel::matmul::common::{GEMMSpiltKParams as SplitKGEMMParams, MatmulArguments},
    },
};

#[derive(Debug, Clone)]
pub(crate) struct DispatchDescriptor {
    pub(crate) params: SplitKGEMMParams,
    pub(crate) partition_count: i32,
    pub(crate) output_elements_per_partition: i32,
    pub(crate) accumulator_bytes: usize,
    pub(crate) partial_threadgroups: MTLSize,
    pub(crate) accum_total_threads: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MetalContext,
        data_type: DataType,
        arguments: &MatmulArguments,
    ) -> Result<Option<Self>, MetalError> {
        if !matches!(data_type, DataType::BF16) {
            return Ok(None);
        }

        if arguments.c.is_some() {
            return Ok(None);
        }

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;
        let batch_count = arguments.batch_count;

        if !should_use_splitk(m, n, k, batch_count) {
            return Ok(None);
        }

        let tile = select_tile_configuration(m, n);
        let partition_count = compute_partition_count(k);
        let mn_aligned = (m % tile.tile_rows) == 0 && (n % tile.tile_cols) == 0;
        let k_aligned = (k % tile.tile_depth) == 0;

        let pipeline_configuration = PipelineConfiguration {
            tile,
            transpose_a: arguments.transpose_a,
            transpose_b: arguments.transpose_b,
            mn_aligned,
            k_aligned,
        };

        if !is_supported_pipeline_configuration(&pipeline_configuration) {
            return Ok(None);
        }

        let tile_count_m = (m + tile.tile_rows - 1) / tile.tile_rows;
        let tile_count_n = (n + tile.tile_cols - 1) / tile.tile_cols;

        let gemm_k_iterations = (k / tile.tile_depth) / partition_count;
        let k_elements_per_partition = gemm_k_iterations * tile.tile_depth;
        let output_elements_per_partition = m * n;

        let accumulator_element_count = partition_count * output_elements_per_partition * arguments.batch_count;
        let accumulator_bytes = (accumulator_element_count as usize) * std::mem::size_of::<f32>();

        let params = SplitKGEMMParams {
            M: m,
            N: n,
            K: k,
            lda: arguments.lda,
            ldb: arguments.ldb,
            ldc: n,
            tiles_n: tile_count_n,
            tiles_m: tile_count_m,
            split_k_partitions: partition_count,
            split_k_partition_stride: output_elements_per_partition,
            split_k_partition_size: k_elements_per_partition,
            gemm_k_iterations_aligned: gemm_k_iterations,
        };

        let partial_threadgroups = MTLSize::new(tile_count_n as usize, tile_count_m as usize, partition_count as usize);

        let accum_total_threads = MTLSize::new(n as usize, m as usize, 1);

        Ok(Some(Self {
            params,
            partition_count,
            output_elements_per_partition,
            accumulator_bytes,
            partial_threadgroups,
            accum_total_threads,
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

fn should_use_splitk(
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

fn is_supported_pipeline_configuration(config: &PipelineConfiguration) -> bool {
    let supported_tile = TileConfiguration {
        tile_rows: 16,
        tile_cols: 32,
        tile_depth: 16,
        warps_per_row: 2,
        warps_per_col: 2,
    };

    config.tile == supported_tile && !config.transpose_a && config.transpose_b && !config.mn_aligned && config.k_aligned
}
