use std::collections::HashMap;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLSize,
};

use super::{
    arguments::MatmulArguments, gemv::GemvKernel, pipeline::PipelineKey,
    shared_types::GEMMParams, splitk::SplitKGemm,
    transpose::transpose_configuration,
};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

pub struct MatmulKernel {
    dt: DataType,
    transpose_a: bool,
    transpose_b: bool,
    gemv: Option<GemvKernel>,
    splitk: Option<SplitKGemm>,
    pipelines: HashMap<PipelineKey, MTLComputePipelineState>,
}

impl MatmulKernel {
    fn dtype_suffix(dt: DataType) -> &'static str {
        match dt {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
            _ => unreachable!(),
        }
    }

    fn kernel_name(
        &self,
        bm: i32,
        bn: i32,
        bk: i32,
        wm: u64,
        wn: u64,
    ) -> String {
        let t = Self::dtype_suffix(self.dt);
        let cfg = transpose_configuration(self.transpose_a, self.transpose_b);
        let tcfg = cfg.as_str();
        format!(
            "gemm_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}",
            tcfg, t, bm, bn, bk, wm, wn
        )
    }

    pub fn new(
        _mtl: &MTLContext,
        dt: DataType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(dt, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MatmulKernel: {dt:?}"
            )));
        }

        Ok(Self {
            dt,
            transpose_a,
            transpose_b,
            gemv: None,
            splitk: None,
            pipelines: HashMap::new(),
        })
    }

    fn select_gemv_rows(
        &self,
        output_dim: i32,
    ) -> u32 {
        if output_dim >= 2048 {
            8
        } else if output_dim >= 512 {
            4
        } else {
            2
        }
    }

    fn maybe_use_gemv(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<bool, MTLError> {
        // GEMV kernel requires:
        // - A not transposed (input vector is contiguous)
        // - B transposed (weight matrix is [N, K] row-major, each row contiguous)
        // Without transpose_b, B is [K, N] and columns are not contiguous
        let m = args.batch;
        let n = args.output_dim;
        if self.transpose_a || !self.transpose_b {
            return Ok(false);
        }

        // For small M (≤ 8), treat as M independent GEMV operations in batch dimension
        // This is more efficient than using GEMM with mostly empty tiles
        let use_batched_gemv =
            m > 1 && m <= 8 && n > 1 && args.batch_count == 1;
        if use_batched_gemv {
            // Reshape: treat M rows as M batches of 1-row GEMV
            let batched_args = MatmulArguments {
                a: args.a,
                b: args.b,
                d: args.d,
                batch: 1,
                input_dim: args.input_dim,
                output_dim: args.output_dim,
                lda: args.lda, // stride between input vectors
                ldb: args.ldb,
                ldd: args.ldd,  // stride between output vectors
                batch_count: m, // M becomes batch count
            };
            let rows = self.select_gemv_rows(n);
            let gemv =
                self.gemv.get_or_insert_with(|| GemvKernel::new(self.dt));
            gemv.encode(mtl, enc, batched_args, rows)?;
            return Ok(true);
        }

        // Standard GEMV for M=1 or N=1
        if m != 1 && n != 1 {
            return Ok(false);
        }
        let rows = self.select_gemv_rows(n);
        let gemv = self.gemv.get_or_insert_with(|| GemvKernel::new(self.dt));
        gemv.encode(mtl, enc, args, rows)?;
        Ok(true)
    }

    fn maybe_use_splitk(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: MatmulArguments,
    ) -> Result<bool, MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;
        let batch_count = args.batch_count;

        let splitk =
            self.splitk.get_or_insert_with(|| SplitKGemm::new(self.dt));
        if splitk.should_use_splitk(m, n, k, batch_count) {
            splitk.encode(mtl, enc, args)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn select_tiles(
        &self,
        mtl: &MTLContext,
        args: &MatmulArguments,
    ) -> (i32, i32, i32, u64, u64) {
        use crate::backends::metal::DeviceClass;
        let mut bm = 64;
        let mut bn = 64;
        let mut bk = 16;
        let mut wm = 2;
        let mut wn = 2;

        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;
        let batch_size_out = args.batch_count;
        let large_mat =
            (batch_size_out as i64) * (m as i64) * (n as i64) >= (1_i64 << 20);

        // Small M optimization: use smaller BM tiles to avoid wasted computation
        // For M <= 32, use 32×32 tiles for better efficiency
        let small_m = m <= 32;
        if small_m && !self.transpose_a {
            // 32×32×16 tiles with 2×2 warp config work well for small M
            return (32, 32, 16, 2, 2);
        }

        // Prefer NAX tiles when available on M4+ hardware.
        if mtl.is_nax_available()
            && (!matches!(self.dt, DataType::F32) || mtl.tf32_enabled())
        {
            if large_mat {
                return (128, 128, 32, 2, 2);
            }
            return (64, 64, 32, 2, 2);
        }
        let devc = match mtl.architecture.device_class {
            DeviceClass::Desktop => 'd',
            DeviceClass::Integrated => 'g',
            DeviceClass::Phone => 'p',
            DeviceClass::Unknown(_) => 'g',
        };

        let is_float = matches!(self.dt, DataType::F32);
        let prefer_halfish = !is_float || mtl.tf32_enabled();

        if devc == 'g' || devc == 'p' {
            if prefer_halfish {
                if !self.transpose_a && self.transpose_b {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                } else {
                    bm = 64;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                }
            } else if !self.transpose_a && self.transpose_b {
                bm = 32;
                bn = 64;
                bk = 16;
                wm = 1;
                wn = 2;
            } else {
                bm = 64;
                bn = 32;
                bk = 32;
                wm = 2;
                wn = 2;
            }
        } else if devc == 'd' {
            if large_mat {
                if prefer_halfish {
                    if 2 * std::cmp::max(m, n) > k {
                        bm = 64;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    } else if !self.transpose_a && self.transpose_b {
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        bm = 32;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    }
                } else if !self.transpose_a && self.transpose_b {
                    bm = 32;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                } else {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                }
            } else {
                if prefer_halfish {
                    if !self.transpose_a && self.transpose_b {
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        bm = 64;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    }
                } else if !self.transpose_a && self.transpose_b {
                    bm = 32;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                } else {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                }
            }
        }

        (bm, bn, bk, wm, wn)
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        align_m: bool,
        align_n: bool,
        align_k: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKey {
            name: name.to_string(),
            align_m,
            align_n,
            align_k,
        };
        if !self.pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
            // Base specializations
            let has_batch = false;
            let use_out_source = false;
            let do_axpby = false;
            fcv.set_constant_value_at_index(
                &has_batch as *const bool as *const _,
                metal::MTLDataType::Bool,
                10,
            );
            fcv.set_constant_value_at_index(
                &use_out_source as *const bool as *const _,
                metal::MTLDataType::Bool,
                100,
            );
            fcv.set_constant_value_at_index(
                &do_axpby as *const bool as *const _,
                metal::MTLDataType::Bool,
                110,
            );
            fcv.set_constant_value_at_index(
                &align_m as *const bool as *const _,
                metal::MTLDataType::Bool,
                200,
            );
            fcv.set_constant_value_at_index(
                &align_n as *const bool as *const _,
                metal::MTLDataType::Bool,
                201,
            );
            fcv.set_constant_value_at_index(
                &align_k as *const bool as *const _,
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{}_am{}_an{}_ak{}",
                name, align_m as u8, align_n as u8, align_k as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                name,
                Some(&fcv),
            )?;
            self.pipelines.insert(key.clone(), ps);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        mut args: MatmulArguments,
    ) -> Result<(), MTLError> {
        self.apply_batch_collapse(&mut args);

        // Try GEMV fast path for decode shapes.
        if self.maybe_use_gemv(mtl, enc, args)? {
            return Ok(());
        }

        // Try Split-K for small M*N and large K.
        if self.maybe_use_splitk(mtl, enc, args)? {
            return Ok(());
        }

        let (bm, bn, bk, wm, wn) = self.select_tiles(mtl, &args);
        let kname = self.kernel_name(bm, bn, bk, wm, wn);

        // M = batch, N = output_dim, K = input_dim
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let am = (m % bm) == 0;
        let an = (n % bn) == 0;
        let ak = (k % bk) == 0;
        let ps = self.get_or_compile_pipeline(mtl, &kname, am, an, ak)?;

        enc.set_compute_pipeline_state(ps);

        // Set buffers (C is elided since use_out_source=false)
        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(3, Some(args.d), 0);

        // Params
        let tiles_n = (n + bn - 1) / bn;
        let tiles_m = (m + bm - 1) / bm;
        // Swizzle: small tm -> no swizzle, otherwise simple 2-way.
        let swizzle_log = if tiles_m <= 3 {
            0
        } else {
            1
        };

        let tile = 1 << swizzle_log;
        let tm_swizzled = (tiles_m + tile - 1) / tile;
        let tn_swizzled = tiles_n * tile;
        let params = GEMMParams {
            batch: m,
            output_dim: n,
            input_dim: k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
            tiles_n: tn_swizzled,
            tiles_m: tm_swizzled,
            batch_stride_a: (args.lda as i64) * (k as i64),
            batch_stride_b: (args.ldb as i64) * (n as i64),
            batch_stride_d: (args.ldd as i64) * (n as i64),
            swizzle_log,
            gemm_k_iterations_aligned: k / bk,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        // Threadgroup sizing
        let threads_per_tg = MTLSize::new(32, wn, wm);
        let tg_x = tn_swizzled as u64;
        let tg_y = tm_swizzled as u64;
        let tg_z = args.batch_count as u64;
        let tgs = MTLSize::new(tg_x, tg_y, tg_z);
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    fn apply_batch_collapse(
        &self,
        args: &mut MatmulArguments,
    ) {
        if self.transpose_a {
            return;
        }
        if args.batch_count <= 1 {
            return;
        }
        // Collapse batch_count into M when A is contiguous and B is broadcast.
        if args.lda == args.input_dim && self.transpose_b {
            args.batch *= args.batch_count;
            args.batch_count = 1;
        }
    }
}
