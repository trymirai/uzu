use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{
                GemmParams,
                unified_gemm::{GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind},
            },
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{
                MatmulGemmMetalKernel, MatmulGemmMppMetalKernel, MatmulGemvMetalKernel, TensorAddBiasMetalKernel,
                unified_matmul::gemm::{GemmTilingConfig, GemmWeights, UnifiedGemmDispatch, UnifiedGemmKernel},
            },
            metal_extensions::DeviceExt,
        },
    },
};

mod gemm;
mod gemm_mpp;
mod gemv;

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv_kernels: HashMap<gemv::GemvSpecialization, MatmulGemvMetalKernel>,
    gemm_kernels: HashMap<gemm::GemmSpecialization, MatmulGemmMetalKernel>,
    gemm_mpp_kernels: HashMap<gemm_mpp::GemmMppSpecialization, MatmulGemmMppMetalKernel>,
    unified_gemm: UnifiedGemmKernel,
    bias_add: TensorAddBiasMetalKernel,
}

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;

static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

impl MatmulMetalKernel {
    fn is_mpp_eligible(
        &self,
        context: &MetalContext,
    ) -> bool {
        context.device.supports_mxu() && matches!(self.data_type, DataType::F16 | DataType::BF16)
    }

    fn get_or_create_gemv(
        &mut self,
        context: &MetalContext,
        specialization: gemv::GemvSpecialization,
    ) -> Result<&MatmulGemvMetalKernel, MatmulError<Metal>> {
        match self.gemv_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemvMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.threadgroup_rows,
                    specialization.threadgroup_cols,
                    specialization.threads_per_simdgroup_row,
                    specialization.threads_per_simdgroup_col,
                    specialization.elements_per_thread_row,
                    specialization.elements_per_thread_col,
                    specialization.is_accumulate,
                    specialization.is_bias,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_gemm(
        &mut self,
        context: &MetalContext,
        specialization: gemm::GemmSpecialization,
    ) -> Result<&MatmulGemmMetalKernel, MatmulError<Metal>> {
        match self.gemm_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemmMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.block_rows as u32,
                    specialization.block_cols as u32,
                    specialization.block_depth as u32,
                    specialization.simdgroups_per_row as u32,
                    specialization.simdgroups_per_column as u32,
                    specialization.align_mn,
                    specialization.align_k,
                    specialization.is_accumulate,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_gemm_mpp(
        &mut self,
        context: &MetalContext,
        specialization: gemm_mpp::GemmMppSpecialization,
    ) -> Result<&MatmulGemmMppMetalKernel, MatmulError<Metal>> {
        match self.gemm_mpp_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemmMppMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.block_rows,
                    specialization.block_cols,
                    specialization.simdgroups_per_row,
                    specialization.simdgroups_per_column,
                    specialization.align_m,
                    specialization.align_n,
                    specialization.align_k,
                    specialization.apply_ab_scale,
                    specialization.is_accumulate,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn encode_gemv(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let (is_accumulate, output_bias) = match arguments.c {
            MatmulArgumentC::Accumulate => (true, None),
            MatmulArgumentC::Bias(bias) => (false, Some(bias)),
            MatmulArgumentC::None => (false, None),
        };

        let specialization = gemv::GemvSpecialization::select(
            arguments.input_dim,
            arguments.output_dim,
            is_accumulate,
            output_bias.is_some(),
        );

        self.get_or_create_gemv(context, specialization)?.encode(
            arguments.b,
            (arguments.a, arguments.a_offset as usize),
            output_bias,
            arguments.d,
            arguments.input_dim,
            arguments.output_dim,
            arguments.input_dim,
            arguments.ab_scale,
            arguments.batch_dim,
            specialization.output_rows_per_threadgroup(),
            encoder,
        );

        Ok(())
    }

    fn encode_gemm(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let specialization = gemm::GemmSpecialization::select(context, self.data_type, &arguments);

        let threadgroups_per_row = arguments.output_dim.div_ceil(specialization.block_cols);
        let threadgroups_per_column = arguments.batch_dim.div_ceil(specialization.block_rows);

        let params = GemmParams {
            M: arguments.batch_dim,
            N: arguments.output_dim,
            K: arguments.input_dim,
            leading_dimension_a: arguments.input_dim,
            leading_dimension_b: arguments.input_dim,
            leading_dimension_d: arguments.output_dim,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log: 0,
            aligned_inner_iterations: arguments.input_dim / specialization.block_depth,
            use_morton: false,
        };

        let kernel = self.get_or_create_gemm(context, specialization)?;

        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &mut *arguments.d,
            std::slice::from_ref(&params),
            threadgroups_per_row,
            threadgroups_per_column,
            arguments.ab_scale,
            encoder,
        );

        if let MatmulArgumentC::Bias(bias) = arguments.c {
            self.bias_add.encode(
                None::<&<Metal as crate::backends::common::Backend>::DenseBuffer>,
                bias,
                arguments.d,
                arguments.output_dim,
                arguments.batch_dim * arguments.output_dim,
                encoder,
            );
        }

        Ok(())
    }

    fn encode_gemm_mpp(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let is_accumulate = matches!(arguments.c, MatmulArgumentC::Accumulate);
        let specialization = gemm_mpp::GemmMppSpecialization::select(
            arguments.batch_dim,
            arguments.output_dim,
            arguments.input_dim,
            is_accumulate,
            arguments.ab_scale != 1.0,
        );

        let threadgroups_per_row = arguments.output_dim.div_ceil(specialization.block_cols);
        let threadgroups_per_column = arguments.batch_dim.div_ceil(specialization.block_rows);

        let max_threadgroups_per_dim = threadgroups_per_row.max(threadgroups_per_column);
        let min_threadgroups_per_dim = threadgroups_per_row.min(threadgroups_per_column);
        let morton_dim = max_threadgroups_per_dim.next_power_of_two();
        let morton_total = morton_dim.saturating_mul(morton_dim);
        let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
        // Morton ordering improves cache locality but requires padding to a square power-of-two grid.
        // Allow up to 4x overhead (idle threadgroups) before the padding cost outweighs the benefit.
        let use_morton = min_threadgroups_per_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total);

        let params = GemmParams {
            M: arguments.batch_dim,
            N: arguments.output_dim,
            K: arguments.input_dim,
            leading_dimension_a: arguments.input_dim,
            leading_dimension_b: arguments.input_dim,
            leading_dimension_d: arguments.output_dim,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log: 0,
            aligned_inner_iterations: 0,
            use_morton,
        };

        let (group_count_x, group_count_y) = if use_morton {
            (morton_total, 1_u32)
        } else {
            (threadgroups_per_row, threadgroups_per_column)
        };

        let kernel = self.get_or_create_gemm_mpp(context, specialization)?;
        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &mut *arguments.d,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            arguments.ab_scale,
            encoder,
        );

        if let MatmulArgumentC::Bias(bias) = arguments.c {
            self.bias_add.encode(
                None::<&<Metal as crate::backends::common::Backend>::DenseBuffer>,
                bias,
                arguments.d,
                arguments.output_dim,
                arguments.batch_dim * arguments.output_dim,
                encoder,
            );
        }

        Ok(())
    }

    fn encode_unified_gemm_full_simdgroup(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let tile = select_unified_gemm_simdgroup_tile(self.data_type, &arguments);
        let group_count_x = arguments.output_dim.div_ceil(tile.threadgroup_n);
        let group_count_y = arguments.batch_dim.div_ceil(tile.threadgroup_m);
        let dispatch = UnifiedGemmDispatch {
            tiling_config: tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            compute: GemmComputeKind::SimdgroupMma,
            output_transform: unified_gemm_output_transform(&arguments),
            alignment: unified_gemm_alignment(&arguments, &tile),
            weights: GemmWeights::FullPrecision {
                weights: arguments.b,
            },
            activations: arguments.a,
            activations_offset: arguments.a_offset as usize,
            result: &mut *arguments.d,
            group_count_x,
            group_count_y,
        };

        self.encode_unified_gemm(context, dispatch, encoder).map_err(MatmulError::BackendError)?;

        if let MatmulArgumentC::Bias(bias) = arguments.c {
            self.bias_add.encode(
                None::<&<Metal as crate::backends::common::Backend>::DenseBuffer>,
                bias,
                arguments.d,
                arguments.output_dim,
                arguments.batch_dim * arguments.output_dim,
                encoder,
            );
        }

        Ok(())
    }

    pub(crate) fn encode_unified_gemm(
        &mut self,
        context: &MetalContext,
        dispatch: UnifiedGemmDispatch<'_>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), crate::backends::metal::error::MetalError> {
        self.unified_gemm.encode(context, dispatch, encoder)
    }

    fn encode_unified_gemm_full_mxu(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if !self.is_mpp_eligible(context) {
            return Err(MatmulError::UnsupportedDataType(self.data_type));
        }

        let tile = select_unified_gemm_mxu_tile(&arguments);
        let group_count_x = arguments.output_dim.div_ceil(tile.threadgroup_n);
        let group_count_y = arguments.batch_dim.div_ceil(tile.threadgroup_m);
        let dispatch = UnifiedGemmDispatch {
            tiling_config: tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            compute: GemmComputeKind::MxuMma,
            output_transform: unified_gemm_output_transform(&arguments),
            alignment: unified_gemm_alignment(&arguments, &tile),
            weights: GemmWeights::FullPrecision {
                weights: arguments.b,
            },
            activations: arguments.a,
            activations_offset: arguments.a_offset as usize,
            result: &mut *arguments.d,
            group_count_x,
            group_count_y,
        };

        self.encode_unified_gemm(context, dispatch, encoder).map_err(MatmulError::BackendError)?;

        if let MatmulArgumentC::Bias(bias) = arguments.c {
            self.bias_add.encode(
                None::<&<Metal as crate::backends::common::Backend>::DenseBuffer>,
                bias,
                arguments.d,
                arguments.output_dim,
                arguments.batch_dim * arguments.output_dim,
                encoder,
            );
        }

        Ok(())
    }
}

fn select_unified_gemm_simdgroup_tile(
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> GemmTilingConfig {
    let (threadgroup_m, threadgroup_n, threadgroup_k) = match data_type {
        DataType::F32 => (32u32, 64u32, 16u32),
        _ => {
            if 2 * arguments.batch_dim.max(arguments.output_dim) > arguments.input_dim {
                (64, 64, 16)
            } else {
                (64, 32, 32)
            }
        },
    };
    let simdgroups_m = 2u32;
    let simdgroups_n = 2u32;
    GemmTilingConfig {
        threadgroup_m,
        threadgroup_n,
        threadgroup_k,
        simdgroup_m: threadgroup_m / simdgroups_m,
        simdgroup_n: threadgroup_n / simdgroups_n,
        simdgroup_k: threadgroup_k,
        fragment_m: 8,
        fragment_n: 8,
        fragment_k: 8,
        simdgroups_m,
        simdgroups_n,
    }
}

fn select_unified_gemm_mxu_tile(arguments: &MatmulArguments<Metal>) -> GemmTilingConfig {
    let (threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) =
        if arguments.batch_dim >= 128 && arguments.output_dim >= 128 {
            (128u32, 128u32, 4u32, 4u32)
        } else if arguments.output_dim < 64 {
            (64, 32, 4, 1)
        } else if arguments.batch_dim < 64 {
            (32, 64, 2, 2)
        } else {
            (64, 64, 2, 2)
        };
    let threadgroup_k = 32u32;
    GemmTilingConfig {
        threadgroup_m,
        threadgroup_n,
        threadgroup_k,
        simdgroup_m: threadgroup_m / simdgroups_m,
        simdgroup_n: threadgroup_n / simdgroups_n,
        simdgroup_k: threadgroup_k,
        fragment_m: 16,
        fragment_n: 16,
        fragment_k: 16,
        simdgroups_m,
        simdgroups_n,
    }
}

fn unified_gemm_alignment(
    arguments: &MatmulArguments<Metal>,
    tile: &GemmTilingConfig,
) -> GemmAlignment {
    GemmAlignment {
        m_aligned: arguments.batch_dim % tile.threadgroup_m == 0,
        n_aligned: arguments.output_dim % tile.threadgroup_n == 0,
        k_aligned: arguments.input_dim % tile.threadgroup_k == 0,
    }
}

fn unified_gemm_output_transform(arguments: &MatmulArguments<Metal>) -> GemmOutputTransformKind {
    let scale = arguments.ab_scale != 1.0;
    let accumulate = matches!(arguments.c, MatmulArgumentC::Accumulate);
    match (scale, accumulate) {
        (false, false) => GemmOutputTransformKind::Store,
        (true, false) => GemmOutputTransformKind::Scale,
        (false, true) => GemmOutputTransformKind::Accumulate,
        (true, true) => GemmOutputTransformKind::ScaleAccumulate,
    }
}

/// Explicit dispatch paths for testing individual kernels independent of production routing.
#[derive(Debug, Clone, Copy)]
pub enum MatmulDispatchPath {
    Auto,
    Gemv,
    Gemm,
    GemmMpp,
    UnifiedGemm,
    UnifiedGemmMxuMma,
}

impl MatmulMetalKernel {
    pub fn encode_with_path(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) {
        match path {
            MatmulDispatchPath::Auto => self.encode(context, arguments, encoder),
            MatmulDispatchPath::Gemv => self.encode_gemv(context, encoder, arguments).expect("Failed to encode GEMV"),
            MatmulDispatchPath::Gemm => self.encode_gemm(context, encoder, arguments).expect("Failed to encode GEMM"),
            MatmulDispatchPath::GemmMpp => {
                self.encode_gemm_mpp(context, encoder, arguments).expect("Failed to encode GEMM MPP")
            },
            MatmulDispatchPath::UnifiedGemm => {
                self.encode_unified_gemm_full_simdgroup(context, encoder, arguments).expect("Failed to encode Unified GEMM")
            },
            MatmulDispatchPath::UnifiedGemmMxuMma => self
                .encode_unified_gemm_full_mxu(context, encoder, arguments)
                .expect("Failed to encode Unified GEMM MXU MMA"),
        }
    }
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        let bias_add = TensorAddBiasMetalKernel::new(context, data_type, true).map_err(MatmulError::BackendError)?;
        let unified_gemm = UnifiedGemmKernel::new(context, data_type).map_err(MatmulError::BackendError)?;

        let mut kernel = Self {
            data_type,
            gemv_kernels: HashMap::new(),
            gemm_kernels: HashMap::new(),
            gemm_mpp_kernels: HashMap::new(),
            unified_gemm,
            bias_add,
        };

        for &config in gemv::GemvSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_gemv(context, config)?;
        }
        for &config in gemm::GemmSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_gemm(context, config)?;
        }
        if context.device.supports_mxu() {
            for &config in gemm_mpp::GemmMppSpecialization::precompile_configs(data_type) {
                kernel.get_or_create_gemm_mpp(context, config)?;
            }
        }

        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        if arguments.batch_dim <= max_gemv_batch_threshold() {
            self.encode_gemv(context, encoder, arguments).expect("Failed to encode GEMV kernel");
        } else if self.is_mpp_eligible(context) {
            self.encode_gemm_mpp(context, encoder, arguments).expect("Failed to encode GEMM MPP kernel");
        } else {
            self.encode_gemm(context, encoder, arguments).expect("Failed to encode GEMM kernel");
        }
    }
}
