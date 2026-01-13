use std::collections::HashMap;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLSize,
};

use super::{
    pipeline_configuration::PipelineConfiguration,
    tile_configuration::TileConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        DeviceClass, MTLContext, MTLError,
        kernel::matmul::common::{
            GEMMAddMMParams, GEMMParams, MatmulArguments,
            transpose_configuration,
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    transpose_a: bool,
    transpose_b: bool,
    pipelines: HashMap<PipelineConfiguration, MTLComputePipelineState>,
}

impl Kernel {
    pub fn new(
        data_type: DataType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for GEMM: {data_type:?}"
            )));
        }
        Ok(Self {
            data_type,
            transpose_a,
            transpose_b,
            pipelines: HashMap::new(),
        })
    }

    fn type_name(&self) -> &'static str {
        match self.data_type {
            DataType::F16 => "float16",
            DataType::BF16 => "bfloat16",
            DataType::F32 => "float32",
            _ => unreachable!(),
        }
    }

    fn kernel_name(
        &self,
        tile: &TileConfiguration,
    ) -> String {
        let type_name = self.type_name();
        let transpose_suffix =
            transpose_configuration(self.transpose_a, self.transpose_b);
        let prefix = if tile.is_nax() {
            "steel_gemm_nax"
        } else {
            "steel_gemm"
        };
        format!(
            "{}_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}",
            prefix,
            transpose_suffix.as_str(),
            type_name,
            type_name,
            tile.block_rows,
            tile.block_cols,
            tile.block_depth,
            tile.warps_per_row,
            tile.warps_per_col
        )
    }

    pub fn select_tile(
        &self,
        mtl: &MTLContext,
        args: &MatmulArguments,
    ) -> TileConfiguration {
        let overall_work_elements = (args.batch_count as i64)
            * (args.batch as i64)
            * (args.output_dim as i64);
        let is_float32 = matches!(self.data_type, DataType::F32);
        let prefer_half_or_tf32 = !is_float32 || mtl.tf32_enabled();

        if mtl.is_nax_available() && prefer_half_or_tf32 {
            let base_tile = TileConfiguration::new(128, 128, 512, 4, 4, 0);
            let tile_rows =
                (args.batch + base_tile.block_rows - 1) / base_tile.block_rows;
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

        let device_class_code = match mtl.architecture.device_class {
            DeviceClass::Desktop => 'd',
            DeviceClass::Integrated => 'g',
            DeviceClass::Phone => 'p',
            DeviceClass::Unknown(_) => 'g',
        };

        match device_class_code {
            'g' | 'p' => {
                if prefer_half_or_tf32 {
                    if !self.transpose_a && self.transpose_b {
                        TileConfiguration::new(64, 32, 32, 2, 2, 0)
                    } else {
                        TileConfiguration::new(64, 64, 16, 1, 2, 0)
                    }
                } else if !self.transpose_a && self.transpose_b {
                    TileConfiguration::new(32, 64, 16, 1, 2, 0)
                } else {
                    TileConfiguration::new(64, 32, 32, 2, 2, 0)
                }
            },
            'd' => {
                if overall_work_elements >= (1_i64 << 20) {
                    if prefer_half_or_tf32 {
                        if 2 * std::cmp::max(args.batch, args.output_dim)
                            > args.input_dim
                        {
                            TileConfiguration::new(64, 64, 16, 2, 2, 0)
                        } else if !self.transpose_a && self.transpose_b {
                            TileConfiguration::new(64, 32, 32, 2, 2, 0)
                        } else {
                            TileConfiguration::new(32, 64, 16, 1, 2, 0)
                        }
                    } else if !self.transpose_a && self.transpose_b {
                        TileConfiguration::new(32, 64, 16, 1, 2, 0)
                    } else {
                        TileConfiguration::new(64, 32, 32, 2, 2, 0)
                    }
                } else if prefer_half_or_tf32 {
                    if !self.transpose_a && self.transpose_b {
                        TileConfiguration::new(64, 32, 32, 2, 2, 0)
                    } else {
                        TileConfiguration::new(64, 64, 16, 1, 2, 0)
                    }
                } else if !self.transpose_a && self.transpose_b {
                    TileConfiguration::new(32, 64, 16, 1, 2, 0)
                } else {
                    TileConfiguration::new(64, 32, 32, 2, 2, 0)
                }
            },
            _ => TileConfiguration::new(64, 64, 16, 2, 2, 0),
        }
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        config: &PipelineConfiguration,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(config) {
            let name = self.kernel_name(&config.tile);
            let fcv = metal::FunctionConstantValues::new();
            fcv.set_constant_value_at_index(
                &config.has_batch as *const bool as *const _,
                metal::MTLDataType::Bool,
                10,
            );
            fcv.set_constant_value_at_index(
                &config.use_out_source as *const bool as *const _,
                metal::MTLDataType::Bool,
                100,
            );
            fcv.set_constant_value_at_index(
                &config.do_axpby as *const bool as *const _,
                metal::MTLDataType::Bool,
                110,
            );
            fcv.set_constant_value_at_index(
                &config.align_m as *const bool as *const _,
                metal::MTLDataType::Bool,
                200,
            );
            fcv.set_constant_value_at_index(
                &config.align_n as *const bool as *const _,
                metal::MTLDataType::Bool,
                201,
            );
            fcv.set_constant_value_at_index(
                &config.align_k as *const bool as *const _,
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{}_am{}_an{}_ak{}_hb{}_uo{}_ax{}",
                name,
                config.align_m as u8,
                config.align_n as u8,
                config.align_k as u8,
                config.has_batch as u8,
                config.use_out_source as u8,
                config.do_axpby as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                &name,
                Some(&fcv),
            )?;
            self.pipelines.insert(config.clone(), ps);
        }
        Ok(self.pipelines.get(config).unwrap())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<(), MTLError> {
        let tile = self.select_tile(mtl, args);
        self.encode_with_tile(mtl, enc, args, &tile)
    }

    pub fn encode_with_tile(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        tile: &TileConfiguration,
    ) -> Result<(), MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let config = PipelineConfiguration {
            tile: *tile,
            align_m: (m % tile.block_rows) == 0,
            align_n: (n % tile.block_cols) == 0,
            align_k: (k % tile.block_depth) == 0,
            has_batch: args.batch_count > 1,
            use_out_source: args.c.is_some(),
            do_axpby: args.c.is_some()
                && (args.alpha != 1.0 || args.beta != 0.0),
        };

        let ps = self.get_or_compile_pipeline(mtl, &config)?;
        enc.set_compute_pipeline_state(ps);

        enc.set_buffer(0, Some(args.a), args.a_offset);
        enc.set_buffer(1, Some(args.b), 0);
        if config.use_out_source {
            if let Some(c_buf) = args.c {
                enc.set_buffer(2, Some(c_buf), 0);
            }
        }
        enc.set_buffer(3, Some(args.d), 0);

        let tiles_n = (n + tile.block_cols - 1) / tile.block_cols;
        let tiles_m = (m + tile.block_rows - 1) / tile.block_rows;
        let swizzle_log = tile.swizzle_log2;

        let tile_swizzle = 1 << swizzle_log;
        let tm_swizzled = (tiles_m + tile_swizzle - 1) / tile_swizzle;
        let tn_swizzled = tiles_n * tile_swizzle;

        let elements_per_matrix_a = (args.batch as i64) * (args.lda as i64);
        let elements_per_matrix_b = if self.transpose_b {
            (args.output_dim as i64) * (args.ldb as i64)
        } else {
            (args.input_dim as i64) * (args.ldb as i64)
        };
        let elements_per_matrix_d = (args.batch as i64) * (args.ldd as i64);

        let params = GEMMParams {
            M: m,
            N: n,
            K: k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
            tiles_n,
            tiles_m,
            batch_stride_a: elements_per_matrix_a,
            batch_stride_b: elements_per_matrix_b,
            batch_stride_d: elements_per_matrix_d,
            swizzle_log,
            gemm_k_iterations_aligned: k / tile.block_depth,
            batch_ndim: 1,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        if config.use_out_source {
            let batch_stride_c = if args.batch_count > 1 {
                (args.ldd as i64) * (args.output_dim as i64)
            } else {
                0
            };
            let addmm_params = GEMMAddMMParams {
                ldc: args.ldd,
                fdc: 1,
                batch_stride_c,
                alpha: args.alpha,
                beta: args.beta,
            };
            enc.set_bytes(
                5,
                std::mem::size_of::<GEMMAddMMParams>() as u64,
                &addmm_params as *const GEMMAddMMParams as *const _,
            );
        }

        let threads_per_tg =
            MTLSize::new(32, tile.warps_per_col, tile.warps_per_row);
        let tgs = MTLSize::new(
            tn_swizzled as u64,
            tm_swizzled as u64,
            args.batch_count as u64,
        );
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }
}
