pub use crate::backends::common::kernel::matmul::gemm_mpp::DispatchDescriptor;
use crate::{
    DataType,
    backends::{
        common::{
            gpu_types::GEMMParams,
            kernel::matmul::{GridSize, MatmulArguments, MatmulError, gemm_mpp::Specialization},
        },
        metal::Metal,
    },
};

impl DispatchDescriptor {
    pub fn new(
        _data_type: DataType,
        arguments: &MatmulArguments<Metal>,
    ) -> Result<Self, MatmulError<Metal>> {
        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let (block_rows, block_cols, block_depth, warps_per_row, warps_per_col) = if m >= 128 && n >= 128 {
            (128, 128, 512, 4i64, 4i64)
        } else {
            (64, 64, 256, 2i64, 2i64)
        };

        let swizzle_log = 0i32;

        let tiles_n = (n + block_cols - 1) / block_cols;
        let tiles_m = (m + block_rows - 1) / block_rows;

        let tile_swizzle = 1 << swizzle_log;
        let tm_swizzled = (tiles_m + tile_swizzle - 1) / tile_swizzle;
        let tn_swizzled = tiles_n * tile_swizzle;

        let elements_per_matrix_a = (arguments.batch as i64) * (arguments.lda as i64);
        let elements_per_matrix_b = if arguments.transpose_b {
            (arguments.output_dim as i64) * (arguments.ldb as i64)
        } else {
            (arguments.input_dim as i64) * (arguments.ldb as i64)
        };
        let elements_per_matrix_d = (arguments.batch as i64) * (arguments.ldd as i64);

        let params = GEMMParams {
            M: m,
            N: n,
            K: k,
            lda: arguments.lda,
            ldb: arguments.ldb,
            ldd: arguments.ldd,
            tiles_n,
            tiles_m,
            batch_stride_a: elements_per_matrix_a,
            batch_stride_b: elements_per_matrix_b,
            batch_stride_d: elements_per_matrix_d,
            swizzle_log,
            gemm_k_iterations_aligned: k / block_depth,
            batch_ndim: 1,
        };

        let config = Specialization {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row: warps_per_row as u64,
            warps_per_col: warps_per_col as u64,
            swizzle_log2: swizzle_log,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
            align_k: (k % block_depth) == 0,
        };

        let threadgroups = GridSize {
            x: tn_swizzled as usize,
            y: tm_swizzled as usize,
            z: arguments.batch_count as usize,
        };

        Ok(Self {
            specialization: config,
            params,
            threadgroups,
        })
    }
}
