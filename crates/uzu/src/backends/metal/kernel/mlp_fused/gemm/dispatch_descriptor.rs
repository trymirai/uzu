use super::{pipeline_configuration::PipelineConfiguration, tile_configuration::select_tile_configuration};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, MTLSize,
        kernel::{matmul::common::GEMMParams, mlp_fused::common::MlpFusedArguments},
    },
};

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) params: GEMMParams,
    pub(crate) hidden_dim: i32,
    pub(crate) threadgroups: MTLSize,
    pub(crate) threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn new(
        _context: &MTLContext,
        data_type: DataType,
        weights_transposed: bool,
        arguments: &MlpFusedArguments,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MTLError::Generic(format!("Unsupported dtype for MLP fused GEMM: {data_type:?}")));
        }

        let tile = select_tile_configuration(arguments.batch, arguments.hidden_dim);

        let mn_aligned = arguments.batch % tile.block_rows == 0 && arguments.hidden_dim % tile.block_cols == 0;
        let k_aligned = arguments.input_dim % tile.block_depth == 0;

        let pipeline_configuration = PipelineConfiguration {
            tile,
            weights_transposed,
            mn_aligned,
            k_aligned,
            activation: arguments.activation,
        };

        let tiles_m = (arguments.batch + tile.block_rows - 1) / tile.block_rows;
        let tiles_n = (arguments.hidden_dim + tile.block_cols - 1) / tile.block_cols;
        let gemm_k_iterations = (arguments.input_dim + tile.block_depth - 1) / tile.block_depth;

        let params = GEMMParams {
            M: arguments.batch,
            N: arguments.hidden_dim * 2,
            K: arguments.input_dim,
            lda: arguments.lda,
            ldb: arguments.ldb,
            ldd: arguments.ldd,
            tiles_n,
            tiles_m,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: 0,
            swizzle_log: 0,
            gemm_k_iterations_aligned: if k_aligned {
                gemm_k_iterations
            } else {
                gemm_k_iterations - 1
            },
            batch_ndim: 0,
        };

        let threadgroups = MTLSize::new(tiles_n as usize, tiles_m as usize, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 2);

        Ok(Self {
            pipeline_configuration,
            params,
            hidden_dim: arguments.hidden_dim,
            threadgroups,
            threads_per_threadgroup,
        })
    }
}
