// Re-export the common DispatchDescriptor type — we add a constructor here
pub use crate::backends::common::kernel::matmul::gemm::DispatchDescriptor;
use crate::{
    DataType,
    backends::{
        common::{
            gpu_types::GEMMParams,
            kernel::matmul::{GridSize, MatmulArguments, gemm::Specialization},
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
};

impl DispatchDescriptor {
    pub fn new(
        context: &MetalContext,
        data_type: DataType,
        arguments: &MatmulArguments<Metal>,
    ) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported dtype for GEMM: {data_type:?}")));
        }

        let config = Specialization::select(context, data_type, arguments);

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let tiles_n = (n + config.block_cols - 1) / config.block_cols;
        let tiles_m = (m + config.block_rows - 1) / config.block_rows;
        let swizzle_log = config.swizzle_log2;

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
            gemm_k_iterations_aligned: k / config.block_depth,
            batch_ndim: 1,
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
