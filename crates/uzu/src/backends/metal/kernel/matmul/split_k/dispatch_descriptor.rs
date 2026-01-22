use super::{
    pipeline_configuration::PipelineConfiguration,
    tile_configuration::select_tile_configuration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, MTLSize,
        kernel::matmul::common::{
            GEMMSpiltKParams as SplitKGEMMParams, MatmulArguments,
        },
    },
};

#[derive(Debug, Clone)]
pub(crate) struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) params: SplitKGEMMParams,
    pub(crate) partition_count: i32,
    pub(crate) output_elements_per_partition: i32,
    pub(crate) accumulator_bytes: usize,
    pub(crate) partial_threadgroups: MTLSize,
    pub(crate) partial_threads_per_threadgroup: MTLSize,
    pub(crate) accum_total_threads: MTLSize,
    pub(crate) accum_threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn try_new(
        _context: &MTLContext,
        data_type: DataType,
        arguments: &MatmulArguments,
    ) -> Result<Option<Self>, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                data_type
            )));
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

        let tile_count_m = (m + tile.tile_rows - 1) / tile.tile_rows;
        let tile_count_n = (n + tile.tile_cols - 1) / tile.tile_cols;

        let gemm_k_iterations = (k / tile.tile_depth) / partition_count;
        let k_elements_per_partition = gemm_k_iterations * tile.tile_depth;
        let output_elements_per_partition = m * n;

        let accumulator_element_count = partition_count
            * output_elements_per_partition
            * arguments.batch_count;
        let accumulator_bytes =
            (accumulator_element_count as usize) * std::mem::size_of::<f32>();

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

        let partial_threads_per_threadgroup = MTLSize::new(
            32,
            tile.warps_per_col as usize,
            tile.warps_per_row as usize,
        );
        let partial_threadgroups = MTLSize::new(
            tile_count_n as usize,
            tile_count_m as usize,
            partition_count as usize,
        );

        let accum_total_threads = MTLSize::new(n as usize, m as usize, 1);
        let accum_threads_per_threadgroup =
            MTLSize::new(16.min(n as usize), 16.min(m as usize), 1);

        Ok(Some(Self {
            pipeline_configuration,
            params,
            partition_count,
            output_elements_per_partition,
            accumulator_bytes,
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
