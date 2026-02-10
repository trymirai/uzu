use super::{pipeline_configuration::PipelineConfiguration, tile_configuration::TileConfiguration};
use crate::{
    DataType,
    backends::metal::{
        DeviceClass, MTLContext, MTLError, MTLSize,
        kernel::matmul::common::{GEMMAddMMParams, GEMMParams, MatmulArguments},
    },
};

#[derive(Debug, Clone)]
pub(crate) struct DispatchDescriptor {
    pub(crate) pipeline_configuration: PipelineConfiguration,
    pub(crate) params: GEMMParams,
    pub(crate) addmm_params: Option<GEMMAddMMParams>,
    pub(crate) threadgroups: MTLSize,
    pub(crate) threads_per_threadgroup: MTLSize,
}

impl DispatchDescriptor {
    pub(crate) fn new(
        context: &MTLContext,
        data_type: DataType,
        arguments: &MatmulArguments,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MTLError::Generic(format!("Unsupported dtype for GEMM: {data_type:?}")));
        }

        let tile = select_tile(context, data_type, arguments);

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let pipeline_configuration = PipelineConfiguration {
            tile,
            transpose_a: arguments.transpose_a,
            transpose_b: arguments.transpose_b,
            align_m: (m % tile.block_rows) == 0,
            align_n: (n % tile.block_cols) == 0,
            align_k: (k % tile.block_depth) == 0,
            has_batch: arguments.batch_count > 1,
            use_out_source: arguments.c.is_some(),
            do_axpby: arguments.c.is_some() && (arguments.alpha != 1.0 || arguments.beta != 0.0),
        };

        let tiles_n = (n + tile.block_cols - 1) / tile.block_cols;
        let tiles_m = (m + tile.block_rows - 1) / tile.block_rows;
        let swizzle_log = tile.swizzle_log2;

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
            gemm_k_iterations_aligned: k / tile.block_depth,
            batch_ndim: 1,
        };

        let addmm_params = if pipeline_configuration.use_out_source {
            let batch_stride_c = if arguments.batch_count > 1 {
                (arguments.ldd as i64) * (arguments.output_dim as i64)
            } else {
                0
            };
            Some(GEMMAddMMParams {
                ldc: arguments.ldd,
                fdc: 1,
                batch_stride_c,
                alpha: arguments.alpha,
                beta: arguments.beta,
            })
        } else {
            None
        };

        let threads_per_threadgroup = MTLSize::new(32, tile.warps_per_col as usize, tile.warps_per_row as usize);
        let threadgroups = MTLSize::new(tn_swizzled as usize, tm_swizzled as usize, arguments.batch_count as usize);

        Ok(Self {
            pipeline_configuration,
            params,
            addmm_params,
            threadgroups,
            threads_per_threadgroup,
        })
    }
}

fn select_tile(
    context: &MTLContext,
    data_type: DataType,
    arguments: &MatmulArguments,
) -> TileConfiguration {
    let overall_work_elements =
        (arguments.batch_count as i64) * (arguments.batch as i64) * (arguments.output_dim as i64);
    let is_float32 = matches!(data_type, DataType::F32);
    let prefer_half_or_tf32 = !is_float32 || context.tf32_enabled();

    if context.is_nax_available() && prefer_half_or_tf32 {
        let base_tile = TileConfiguration::new(128, 128, 512, 4, 4, 0);
        let tile_rows = (arguments.batch + base_tile.block_rows - 1) / base_tile.block_rows;
        let swizzle_log2 = if tile_rows <= 3 {
            0
        } else {
            1
        };
        return TileConfiguration {
            swizzle_log2,
            ..base_tile
        };
    }

    match context.architecture.device_class {
        DeviceClass::Integrated | DeviceClass::Phone | DeviceClass::Unknown(_) => {
            if prefer_half_or_tf32 {
                if !arguments.transpose_a && arguments.transpose_b {
                    TileConfiguration::new(64, 32, 32, 2, 2, 0)
                } else {
                    TileConfiguration::new(64, 64, 16, 1, 2, 0)
                }
            } else if !arguments.transpose_a && arguments.transpose_b {
                TileConfiguration::new(32, 64, 16, 1, 2, 0)
            } else {
                TileConfiguration::new(64, 32, 32, 2, 2, 0)
            }
        },
        DeviceClass::Desktop => {
            if overall_work_elements >= (1_i64 << 20) {
                if prefer_half_or_tf32 {
                    if 2 * std::cmp::max(arguments.batch, arguments.output_dim) > arguments.input_dim {
                        TileConfiguration::new(64, 64, 16, 2, 2, 0)
                    } else if !arguments.transpose_a && arguments.transpose_b {
                        TileConfiguration::new(64, 32, 32, 2, 2, 0)
                    } else {
                        TileConfiguration::new(32, 64, 16, 1, 2, 0)
                    }
                } else if !arguments.transpose_a && arguments.transpose_b {
                    TileConfiguration::new(32, 64, 16, 1, 2, 0)
                } else {
                    TileConfiguration::new(64, 32, 32, 2, 2, 0)
                }
            } else if prefer_half_or_tf32 {
                if !arguments.transpose_a && arguments.transpose_b {
                    TileConfiguration::new(64, 32, 32, 2, 2, 0)
                } else {
                    TileConfiguration::new(64, 64, 16, 1, 2, 0)
                }
            } else if !arguments.transpose_a && arguments.transpose_b {
                TileConfiguration::new(32, 64, 16, 1, 2, 0)
            } else {
                TileConfiguration::new(64, 32, 32, 2, 2, 0)
            }
        },
    }
}
