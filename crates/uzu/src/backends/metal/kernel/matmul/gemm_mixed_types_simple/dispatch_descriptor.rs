pub use crate::backends::common::kernel::matmul::gemm_mixed_types_simple::DispatchDescriptor;
use crate::{
    DataType,
    backends::{
        common::{
            gpu_types::GEMMParams,
            kernel::matmul::{GridSize, MatmulArguments},
        },
        metal::{Metal, error::MetalError},
    },
};

const TILE_M: i32 = 32;
const TILE_N: i32 = 32;

impl DispatchDescriptor {
    pub fn new(
        _data_type: DataType,
        arguments: &MatmulArguments<Metal>,
    ) -> Result<Self, MetalError> {
        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let tiles_m = (m + TILE_M - 1) / TILE_M;
        let tiles_n = (n + TILE_N - 1) / TILE_N;

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
            swizzle_log: 0,
            gemm_k_iterations_aligned: 0,
            batch_ndim: 1,
        };

        let threadgroups = GridSize {
            x: tiles_n as usize,
            y: tiles_m as usize,
            z: arguments.batch_count as usize,
        };

        Ok(Self {
            params,
            threadgroups,
        })
    }
}
