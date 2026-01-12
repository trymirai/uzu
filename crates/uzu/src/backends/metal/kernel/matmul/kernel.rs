use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{
    super::{KernelDataType, TensorAddBias, mlp::MlpActivationType},
    arguments::MatmulArguments,
    gemv::GemvKernel,
    pipeline::PipelineKey,
    shared_types::{GEMMAddMMParams, GEMMParams},
    splitk::SplitKGemm,
    transpose::transpose_configuration,
};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
    utils::env_utils::debug_matmul_enabled,
};

#[derive(Debug, Clone, Copy)]
struct TileSelection {
    block_rows: i32,
    block_cols: i32,
    block_depth: i32,
    warps_per_row: u64,
    warps_per_col: u64,
    swizzle_log2: i32,
}

impl TileSelection {
    fn new(
        block_rows: i32,
        block_cols: i32,
        block_depth: i32,
        warps_per_row: u64,
        warps_per_col: u64,
        swizzle_log2: i32,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row,
            warps_per_col,
            swizzle_log2,
        }
    }
}

pub struct MatmulKernel {
    data_type: DataType,
    lhs_is_transposed: bool,
    rhs_is_transposed: bool,
    gemv: Option<GemvKernel>,
    splitk: Option<SplitKGemm>,
    bias_add: Option<TensorAddBias>,
    pipelines: HashMap<PipelineKey, MTLComputePipelineState>,
}

impl MatmulKernel {
    fn steel_type_name(dt: DataType) -> &'static str {
        match dt {
            DataType::F16 => "float16",
            DataType::BF16 => "bfloat16",
            DataType::F32 => "float32",
            _ => unreachable!(),
        }
    }

    fn kernel_name(
        &self,
        tile: &TileSelection,
    ) -> String {
        let type_name = Self::steel_type_name(self.data_type);
        let transpose_suffix = transpose_configuration(
            self.lhs_is_transposed,
            self.rhs_is_transposed,
        );
        let prefix = if tile.block_depth >= 256 {
            "steel_gemm_fused_nax"
        } else {
            "steel_gemm_fused"
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

    pub fn new(
        _mtl: &MTLContext,
        data_type: DataType,
        lhs_is_transposed: bool,
        rhs_is_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MatmulKernel: {data_type:?}"
            )));
        }

        Ok(Self {
            data_type,
            lhs_is_transposed,
            rhs_is_transposed,
            gemv: None,
            splitk: None,
            bias_add: None,
            pipelines: HashMap::new(),
        })
    }

    fn maybe_use_gemv_impl(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: Option<&MTLBuffer>,
    ) -> Result<bool, MTLError> {
        // GEMV kernel requires:
        // - A not transposed (input vector is contiguous)
        // - B transposed (weight matrix is [N, K] row-major, each row contiguous)
        // Without transpose_b, B is [K, N] and columns are not contiguous
        let m = args.batch;
        let n = args.output_dim;
        if self.lhs_is_transposed || !self.rhs_is_transposed {
            return Ok(false);
        }

        if m != 1 && n != 1 {
            return Ok(false);
        }
        let gemv = self.gemv.get_or_insert_with(|| {
            GemvKernel::new(
                self.data_type,
                self.lhs_is_transposed,
                self.rhs_is_transposed,
            )
        });
        if let Some(bias) = bias {
            gemv.encode_with_bias(mtl, enc, args, bias)?;
        } else {
            gemv.encode(mtl, enc, args)?;
        }
        Ok(true)
    }

    fn maybe_use_gemv(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<bool, MTLError> {
        self.maybe_use_gemv_impl(mtl, enc, args, None)
    }

    fn maybe_use_gemv_with_bias(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<bool, MTLError> {
        self.maybe_use_gemv_impl(mtl, enc, args, Some(bias))
    }

    fn maybe_use_splitk(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<bool, MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;
        let batch_count = args.batch_count;

        if !SplitKGemm::should_use_splitk(m, n, k, batch_count) {
            return Ok(false);
        }

        let splitk = self.splitk.get_or_insert_with(|| {
            SplitKGemm::new(
                self.data_type,
                self.lhs_is_transposed,
                self.rhs_is_transposed,
            )
        });
        splitk.encode(mtl, enc, args)?;
        Ok(true)
    }

    fn select_tile_configuration(
        &self,
        mtl: &MTLContext,
        args: &MatmulArguments,
    ) -> TileSelection {
        use crate::backends::metal::DeviceClass;

        let overall_work_elements = (args.batch_count as i64)
            * (args.batch as i64)
            * (args.output_dim as i64);
        let is_float32 = matches!(self.data_type, DataType::F32);
        let prefer_half_or_tf32 = !is_float32 || mtl.tf32_enabled();

        // NAX path: prefer deeper K tiles on NAX-capable devices.
        if mtl.is_nax_available() && prefer_half_or_tf32 {
            let base_tile = TileSelection::new(
                128, 128, 512, 4, 4, /*swizzle_log2=*/ 0,
            );
            let tile_rows =
                (args.batch + base_tile.block_rows - 1) / base_tile.block_rows;
            let swizzle_log2 = if tile_rows <= 3 {
                0
            } else {
                1
            };
            return TileSelection {
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

        // Default gemm matmul heuristics (non-NAX), matching observed
        // choices on M2 Max: prefer 64x64x16 wm2/wn2 for large shapes and
        // fall back to 64x32x32 when transpose_b=true and not N-aligned.
        match device_class_code {
            'g' | 'p' => {
                if prefer_half_or_tf32 {
                    if !self.lhs_is_transposed && self.rhs_is_transposed {
                        TileSelection::new(64, 32, 32, 2, 2, 0)
                    } else {
                        TileSelection::new(64, 64, 16, 1, 2, 0)
                    }
                } else if !self.lhs_is_transposed && self.rhs_is_transposed {
                    TileSelection::new(32, 64, 16, 1, 2, 0)
                } else {
                    TileSelection::new(64, 32, 32, 2, 2, 0)
                }
            },
            'd' => {
                if overall_work_elements >= (1_i64 << 20) {
                    if prefer_half_or_tf32 {
                        if 2 * std::cmp::max(args.batch, args.output_dim)
                            > args.input_dim
                        {
                            TileSelection::new(64, 64, 16, 2, 2, 0)
                        } else if !self.lhs_is_transposed
                            && self.rhs_is_transposed
                        {
                            TileSelection::new(64, 32, 32, 2, 2, 0)
                        } else {
                            TileSelection::new(32, 64, 16, 1, 2, 0)
                        }
                    } else if !self.lhs_is_transposed && self.rhs_is_transposed
                    {
                        TileSelection::new(32, 64, 16, 1, 2, 0)
                    } else {
                        TileSelection::new(64, 32, 32, 2, 2, 0)
                    }
                } else if prefer_half_or_tf32 {
                    if !self.lhs_is_transposed && self.rhs_is_transposed {
                        TileSelection::new(64, 32, 32, 2, 2, 0)
                    } else {
                        TileSelection::new(64, 64, 16, 1, 2, 0)
                    }
                } else if !self.lhs_is_transposed && self.rhs_is_transposed {
                    TileSelection::new(32, 64, 16, 1, 2, 0)
                } else {
                    TileSelection::new(64, 32, 32, 2, 2, 0)
                }
            },
            _ => TileSelection::new(64, 64, 16, 2, 2, 0),
        }
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        align_m: bool,
        align_n: bool,
        align_k: bool,
        has_batch: bool,
        use_out_source: bool,
        do_axpby: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = PipelineKey {
            name: name.to_string(),
            align_m,
            align_n,
            align_k,
            has_batch,
            use_out_source,
            do_axpby,
        };
        if !self.pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
            // Base specializations
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
                "{}_am{}_an{}_ak{}_hb{}_uo{}_ax{}",
                name,
                align_m as u8,
                align_n as u8,
                align_k as u8,
                has_batch as u8,
                use_out_source as u8,
                do_axpby as u8
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

    fn encode_gemm(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<(), MTLError> {
        let tile = self.select_tile_configuration(mtl, &args);
        let kname = self.kernel_name(&tile);

        // M = batch, N = output_dim, K = input_dim
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;

        let align_m = (m % tile.block_rows) == 0;
        let align_n = (n % tile.block_cols) == 0;
        let align_k = (k % tile.block_depth) == 0;
        let has_batch = args.batch_count > 1;
        let use_out_source = args.c.is_some();
        let do_axpby =
            use_out_source && (args.alpha != 1.0 || args.beta != 0.0);
        let ps = self.get_or_compile_pipeline(
            mtl,
            &kname,
            align_m,
            align_n,
            align_k,
            has_batch,
            use_out_source,
            do_axpby,
        )?;

        enc.set_compute_pipeline_state(ps);

        // Set buffers
        enc.set_buffer(0, Some(args.a), args.a_offset);
        enc.set_buffer(1, Some(args.b), 0);
        if use_out_source {
            if let Some(c_buf) = args.c {
                enc.set_buffer(2, Some(c_buf), 0);
            }
        }
        enc.set_buffer(3, Some(args.d), 0);

        // Params
        let tiles_n = (n + tile.block_cols - 1) / tile.block_cols;
        let tiles_m = (m + tile.block_rows - 1) / tile.block_rows;
        let swizzle_log = tile.swizzle_log2;

        let tile_swizzle = 1 << swizzle_log;
        let tm_swizzled = (tiles_m + tile_swizzle - 1) / tile_swizzle;
        let tn_swizzled = tiles_n * tile_swizzle;
        let elements_per_matrix_a = (args.batch as i64) * (args.lda as i64);
        let elements_per_matrix_b = if self.rhs_is_transposed {
            (args.output_dim as i64) * (args.ldb as i64)
        } else {
            (args.input_dim as i64) * (args.ldb as i64)
        };
        let elements_per_matrix_d = (args.batch as i64) * (args.ldd as i64);
        let batch_ndim = 1;
        let params = GEMMParams {
            M: m,
            N: n,
            K: k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
            // NOTE: tiles_{n,m} are the *unswizzled* tile counts used for bounds
            // checks inside the GEMM kernel. The dispatched grid dimensions are
            // swizzled separately via tn_swizzled/tm_swizzled.
            tiles_n,
            tiles_m,
            batch_stride_a: elements_per_matrix_a,
            batch_stride_b: elements_per_matrix_b,
            batch_stride_d: elements_per_matrix_d,
            swizzle_log,
            gemm_k_iterations_aligned: k / tile.block_depth,
            batch_ndim,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        if use_out_source {
            let ldc = args.ldd;
            let fdc = 1;
            let batch_stride_c = if args.batch_count > 1 {
                (args.ldd as i64) * (args.output_dim as i64)
            } else {
                0
            };
            let addmm_params = GEMMAddMMParams {
                ldc,
                fdc,
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

        // Threadgroup sizing
        let threads_per_tg =
            MTLSize::new(32, tile.warps_per_col, tile.warps_per_row);
        let tg_x = tn_swizzled as u64;
        let tg_y = tm_swizzled as u64;
        let tg_z = args.batch_count as u64;
        let tgs = MTLSize::new(tg_x, tg_y, tg_z);
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        mut args: MatmulArguments,
    ) -> Result<(), MTLError> {
        self.apply_batch_collapse(&mut args);

        // Try GEMV fast path for decode shapes.
        if self.maybe_use_gemv(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_gemv(&args);
            }
            return Ok(());
        }

        // Try Split-K for small M*N and large K.
        if self.maybe_use_splitk(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_splitk(&args);
            }
            return Ok(());
        }

        if debug_matmul_enabled() {
            let tile = self.select_tile_configuration(mtl, &args);
            self.log_gemm(&args, &tile);
        }

        self.encode_gemm(mtl, enc, &args)
    }

    pub fn encode_with_bias(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        mut args: MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        self.apply_batch_collapse(&mut args);

        if self.maybe_use_gemv_with_bias(mtl, enc, &args, bias)? {
            if debug_matmul_enabled() {
                self.log_gemv(&args);
            }
            return Ok(());
        }

        if self.maybe_use_splitk(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_splitk(&args);
            }
            self.apply_bias_add(mtl, enc, &args, bias)?;
            return Ok(());
        }

        if debug_matmul_enabled() {
            let tile = self.select_tile_configuration(mtl, &args);
            self.log_gemm(&args, &tile);
        }

        self.encode_gemm(mtl, enc, &args)?;
        self.apply_bias_add(mtl, enc, &args, bias)?;
        Ok(())
    }

    fn apply_bias_add(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        let m = args.batch as usize;
        let n = args.output_dim as usize;
        let batch_count = args.batch_count as usize;
        let total_len = m * n * batch_count;
        if total_len == 0 {
            return Ok(());
        }

        if self.bias_add.is_none() {
            self.bias_add = Some(TensorAddBias::new(
                mtl,
                KernelDataType::from(self.data_type),
            )?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add
            .encode_with_encoder(args.d, bias, args.d, n, total_len, enc, None);
        Ok(())
    }

    fn apply_batch_collapse(
        &self,
        args: &mut MatmulArguments,
    ) {
        if self.lhs_is_transposed {
            return;
        }
        if args.batch_count <= 1 {
            return;
        }
        // Collapse batch_count into M when A is contiguous and B is broadcast.
        if args.lda == args.input_dim && self.rhs_is_transposed {
            args.batch *= args.batch_count;
            args.batch_count = 1;
        }
    }

    fn log_gemv(
        &self,
        args: &MatmulArguments,
    ) {
        eprintln!(
            "[matmul] GEMV m={} k={} n={} batch={} dtype={:?}",
            args.batch,
            args.input_dim,
            args.output_dim,
            args.batch_count,
            self.data_type
        );
    }

    fn log_splitk(
        &self,
        args: &MatmulArguments,
    ) {
        eprintln!(
            "[matmul] SplitK m={} k={} n={} batch={} dtype={:?}",
            args.batch,
            args.input_dim,
            args.output_dim,
            args.batch_count,
            self.data_type
        );
    }

    fn log_gemm(
        &self,
        args: &MatmulArguments,
        tile: &TileSelection,
    ) {
        let kernel_name = self.kernel_name(tile);
        eprintln!(
            "[matmul] GEMM m={} k={} n={} batch={} dtype={:?} tile={}x{}x{} kernel={}",
            args.batch,
            args.input_dim,
            args.output_dim,
            args.batch_count,
            self.data_type,
            tile.block_rows,
            tile.block_cols,
            tile.block_depth,
            kernel_name
        );
    }
}

/// Arguments for MLP fused GEMM
#[derive(Debug)]
pub struct MlpFusedGemmArguments<'a> {
    /// Input activations [M, K]
    pub input: &'a MTLBuffer,
    /// Input byte offset
    pub input_offset: u64,
    /// Weight matrix [K, 2*hidden_dim] or [2*hidden_dim, K] depending on transpose
    pub weights: &'a MTLBuffer,
    /// Output [M, hidden_dim]
    pub output: &'a MTLBuffer,
    /// Batch size (M)
    pub batch: i32,
    /// Input dimension (K)
    pub input_dim: i32,
    /// Hidden dimension (output size, half of weight columns/rows)
    pub hidden_dim: i32,
    /// Leading dimension of input
    pub lda: i32,
    /// Leading dimension of weights
    pub ldb: i32,
    /// Leading dimension of output
    pub ldd: i32,
    /// Activation type for fused epilogue
    pub activation: MlpActivationType,
}

/// MLP Fused GEMM Kernel for prefill path
/// Computes paired up/gate projections with fused activation: out = up * activation(gate)
pub struct MlpFusedGemmKernel {
    data_type: DataType,
    weights_transposed: bool,
    pipelines: HashMap<
        (i32, i32, bool, bool, MlpActivationType),
        MTLComputePipelineState,
    >,
}

impl MlpFusedGemmKernel {
    pub fn new(
        data_type: DataType,
        weights_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMM: {:?}",
                data_type
            )));
        }
        Ok(Self {
            data_type,
            weights_transposed,
            pipelines: HashMap::new(),
        })
    }

    fn kernel_name(
        &self,
        bm: i32,
        bn: i32,
        mn_aligned: bool,
        k_aligned: bool,
    ) -> String {
        let type_name = match self.data_type {
            DataType::F16 => "half",
            DataType::BF16 => "bfloat",
            DataType::F32 => "float",
            _ => unreachable!(),
        };
        let transpose_char = if self.weights_transposed {
            "t"
        } else {
            "n"
        };
        let mn_str = if mn_aligned {
            "true"
        } else {
            "false"
        };
        let k_str = if k_aligned {
            "true"
        } else {
            "false"
        };

        format!(
            "steel_gemm_mlp_fused_{}_{}_{}_bm{}_bn{}_bk16_wm2_wn2_align_MN_{}_K_{}",
            transpose_char, type_name, type_name, bm, bn, mn_str, k_str
        )
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        bm: i32,
        bn: i32,
        mn_aligned: bool,
        k_aligned: bool,
        activation: MlpActivationType,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = (bm, bn, mn_aligned, k_aligned, activation);
        if !self.pipelines.contains_key(&key) {
            let kernel_name = self.kernel_name(bm, bn, mn_aligned, k_aligned);

            // Create function constant values for MLP activation
            let fcv = metal::FunctionConstantValues::new();
            let activation_val = activation as u32;
            fcv.set_constant_value_at_index(
                &activation_val as *const u32 as *const _,
                metal::MTLDataType::UInt,
                52, // MLP_ACTIVATION index
            );

            let pipeline =
                context.compute_pipeline_state(&kernel_name, Some(&fcv))?;
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn select_tile_size(
        &self,
        m: i32,
        hidden_dim: i32,
        k: i32,
    ) -> (i32, i32) {
        // Select tile size based on problem dimensions
        let _ = k; // K doesn't affect tile selection directly
        if m >= 64 && hidden_dim >= 64 {
            (64, 64)
        } else {
            (32, 32)
        }
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: &MlpFusedGemmArguments,
    ) -> Result<(), MTLError> {
        let (bm, bn) =
            self.select_tile_size(args.batch, args.hidden_dim, args.input_dim);
        let bk = 16;

        let mn_aligned = args.batch % bm == 0 && args.hidden_dim % bn == 0;
        let k_aligned = args.input_dim % bk == 0;

        let pipeline = self.get_pipeline(
            context,
            bm,
            bn,
            mn_aligned,
            k_aligned,
            args.activation,
        )?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.input), args.input_offset);
        encoder.set_buffer(1, Some(args.weights), 0);
        encoder.set_buffer(2, Some(args.output), 0);

        // Set GEMM params
        let tiles_m = (args.batch + bm - 1) / bm;
        let tiles_n = (args.hidden_dim + bn - 1) / bn;
        let gemm_k_iterations = (args.input_dim + bk - 1) / bk;

        let params = GEMMParams {
            M: args.batch,
            N: args.hidden_dim * 2, // Full weight width for pointer arithmetic
            K: args.input_dim,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
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

        encoder.set_bytes(
            3,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            std::mem::size_of::<i32>() as u64,
            &args.hidden_dim as *const i32 as *const std::ffi::c_void,
        );

        let threadgroup_count = MTLSize::new(tiles_n as u64, tiles_m as u64, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 2); // WM=2, WN=2, 32 threads per simdgroup

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }
}
