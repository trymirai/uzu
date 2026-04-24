use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use crate::{
    DataType,
    backends::{
        common::{
            Allocation, Encoder,
            gpu_types::GemmParams,
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
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            a_offset,
            b,
            ab_scale,
            c,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;

        let (is_accumulate, output_bias) = match c {
            MatmulArgumentC::Accumulate => (true, None),
            MatmulArgumentC::Bias(bias) => (false, Some(bias)),
            MatmulArgumentC::None => (false, None),
        };

        let specialization =
            gemv::GemvSpecialization::select(input_dim, output_dim, is_accumulate, output_bias.is_some());

        self.get_or_create_gemv(encoder.context(), specialization)?.encode(
            b,
            (a, a_offset),
            output_bias,
            d,
            input_dim,
            output_dim,
            input_dim,
            ab_scale,
            batch_dim,
            specialization.output_rows_per_threadgroup(),
            encoder,
        );

        Ok(())
    }

    fn encode_gemm(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            a_offset,
            b,
            ab_scale,
            c,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;

        let specialization =
            gemm::GemmSpecialization::select(encoder.context(), self.data_type, batch_dim, input_dim, output_dim, &c);

        let threadgroups_per_row = output_dim.div_ceil(specialization.block_cols);
        let threadgroups_per_column = batch_dim.div_ceil(specialization.block_rows);

        let params = GemmParams {
            M: batch_dim,
            N: output_dim,
            K: input_dim,
            leading_dimension_a: input_dim,
            leading_dimension_b: input_dim,
            leading_dimension_d: output_dim,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log: 0,
            aligned_inner_iterations: input_dim / specialization.block_depth,
            use_morton: false,
        };

        let kernel = self.get_or_create_gemm(encoder.context(), specialization)?;

        kernel.encode(
            (a, a_offset),
            b,
            &mut *d,
            params,
            threadgroups_per_row,
            threadgroups_per_column,
            ab_scale,
            encoder,
        );

        if let MatmulArgumentC::Bias(bias) = c {
            self.bias_add.encode(None::<&Allocation<Metal>>, bias, d, output_dim, batch_dim * output_dim, encoder);
        }

        Ok(())
    }

    fn encode_gemm_mpp(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            a_offset,
            b,
            ab_scale,
            c,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;
        let is_accumulate = matches!(c, MatmulArgumentC::Accumulate);
        let specialization =
            gemm_mpp::GemmMppSpecialization::select(batch_dim, output_dim, input_dim, is_accumulate, ab_scale != 1.0);

        let threadgroups_per_row = output_dim.div_ceil(specialization.block_cols);
        let threadgroups_per_column = batch_dim.div_ceil(specialization.block_rows);

        let max_threadgroups_per_dim = threadgroups_per_row.max(threadgroups_per_column);
        let min_threadgroups_per_dim = threadgroups_per_row.min(threadgroups_per_column);
        let morton_dim = max_threadgroups_per_dim.next_power_of_two();
        let morton_total = morton_dim.saturating_mul(morton_dim);
        let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
        // Morton ordering improves cache locality but requires padding to a square power-of-two grid.
        // Allow up to 4x overhead (idle threadgroups) before the padding cost outweighs the benefit.
        let use_morton = min_threadgroups_per_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total);

        let params = GemmParams {
            M: batch_dim,
            N: output_dim,
            K: input_dim,
            leading_dimension_a: input_dim,
            leading_dimension_b: input_dim,
            leading_dimension_d: output_dim,
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

        let kernel = self.get_or_create_gemm_mpp(encoder.context(), specialization)?;
        kernel.encode((a, a_offset), b, &mut *d, params, group_count_x, group_count_y, ab_scale, encoder);

        if let MatmulArgumentC::Bias(bias) = c {
            self.bias_add.encode(None::<&Allocation<Metal>>, bias, d, output_dim, batch_dim * output_dim, encoder);
        }

        Ok(())
    }
}

/// Explicit dispatch paths for testing individual kernels independent of production routing.
#[derive(Debug, Clone, Copy)]
pub enum MatmulDispatchPath {
    Auto,
    Gemv,
    Gemm,
    GemmMpp,
}

impl MatmulMetalKernel {
    pub fn encode_with_path(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) {
        match path {
            MatmulDispatchPath::Auto => self.encode(arguments, encoder),
            MatmulDispatchPath::Gemv => self.encode_gemv(encoder, arguments).expect("Failed to encode GEMV"),
            MatmulDispatchPath::Gemm => self.encode_gemm(encoder, arguments).expect("Failed to encode GEMM"),
            MatmulDispatchPath::GemmMpp => self.encode_gemm_mpp(encoder, arguments).expect("Failed to encode GEMM MPP"),
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

        let mut kernel = Self {
            data_type,
            gemv_kernels: HashMap::new(),
            gemm_kernels: HashMap::new(),
            gemm_mpp_kernels: HashMap::new(),
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
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        if arguments.batch_dim <= max_gemv_batch_threshold() {
            self.encode_gemv(encoder, arguments).expect("Failed to encode GEMV kernel");
        } else if self.is_mpp_eligible(encoder.context()) {
            self.encode_gemm_mpp(encoder, arguments).expect("Failed to encode GEMM MPP kernel");
        } else {
            self.encode_gemm(encoder, arguments).expect("Failed to encode GEMM kernel");
        }
    }
}
