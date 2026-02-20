use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{MatmulArguments, gemm::Specialization},
        metal::{
            Metal,
            context::{DeviceClass, MetalContext},
        },
    },
};

impl Specialization {
    pub fn select(
        context: &MetalContext,
        data_type: DataType,
        arguments: &MatmulArguments<Metal>,
    ) -> Self {
        let overall_work_elements =
            (arguments.batch_count as i64) * (arguments.batch as i64) * (arguments.output_dim as i64);
        let is_float32 = matches!(data_type, DataType::F32);
        let prefer_half_or_tf32 = !is_float32 || context.tf32_enabled();

        let (block_rows, block_cols, block_depth, warps_per_row, warps_per_col, swizzle_log2) =
            if context.is_nax_available() && prefer_half_or_tf32 {
                let tile_rows = (arguments.batch + 128 - 1) / 128;
                let swizzle_log2 = if tile_rows <= 3 {
                    0
                } else {
                    1
                };
                (128, 128, 512, 4, 4, swizzle_log2)
            } else {
                match context.device_class() {
                    DeviceClass::Integrated | DeviceClass::Phone | DeviceClass::Unknown(_) => {
                        if prefer_half_or_tf32 {
                            if !arguments.transpose_a && arguments.transpose_b {
                                (64, 32, 32, 2, 2, 0)
                            } else {
                                (64, 64, 16, 1, 2, 0)
                            }
                        } else if !arguments.transpose_a && arguments.transpose_b {
                            (32, 64, 16, 1, 2, 0)
                        } else {
                            (64, 32, 32, 2, 2, 0)
                        }
                    },
                    DeviceClass::Desktop => {
                        if overall_work_elements >= (1_i64 << 20) {
                            if prefer_half_or_tf32 {
                                if 2 * std::cmp::max(arguments.batch, arguments.output_dim) > arguments.input_dim {
                                    (64, 64, 16, 2, 2, 0)
                                } else if !arguments.transpose_a && arguments.transpose_b {
                                    (64, 32, 32, 2, 2, 0)
                                } else {
                                    (32, 64, 16, 1, 2, 0)
                                }
                            } else if !arguments.transpose_a && arguments.transpose_b {
                                (32, 64, 16, 1, 2, 0)
                            } else {
                                (64, 32, 32, 2, 2, 0)
                            }
                        } else if prefer_half_or_tf32 {
                            if !arguments.transpose_a && arguments.transpose_b {
                                (64, 32, 32, 2, 2, 0)
                            } else {
                                (64, 64, 16, 1, 2, 0)
                            }
                        } else if !arguments.transpose_a && arguments.transpose_b {
                            (32, 64, 16, 1, 2, 0)
                        } else {
                            (64, 32, 32, 2, 2, 0)
                        }
                    },
                }
            };

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        Self {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row,
            warps_per_col,
            swizzle_log2,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
            align_k: (k % block_depth) == 0,
        }
    }
}
