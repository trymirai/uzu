use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use super::{
    super::dsl::{
        MatmulGemmMetalKernel, MatmulGemmMppDirectMetalKernel, MatmulGemmMppMetalKernel, MatmulGemvMetalKernel,
        TensorAddBiasMetalKernel,
    },
    gemm, gemm_mpp, gemm_mpp_direct, gemv,
};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::GemmParams,
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        metal::{Metal, context::MetalContext},
    },
};

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv_kernels: HashMap<gemv::Specialization, MatmulGemvMetalKernel>,
    gemm_kernels: HashMap<gemm::GemmSpecialization, MatmulGemmMetalKernel>,
    gemm_mpp_kernels: HashMap<gemm_mpp::GemmMppSpecialization, MatmulGemmMppMetalKernel>,
    gemm_mpp_direct_kernels: HashMap<gemm_mpp_direct::GemmMppDirectSpecialization, MatmulGemmMppDirectMetalKernel>,
    bias_add: Option<TensorAddBiasMetalKernel>,
}

const DEFAULT_GEMV_MAX_BATCH: i32 = 8;

static GEMV_MAX_BATCH: OnceLock<i32> = OnceLock::new();
fn max_gemv_batch_threshold() -> i32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

impl MatmulMetalKernel {
    fn is_gemv_eligible(arguments: &MatmulArguments<Metal>) -> bool {
        let m = arguments.batch;
        let n = arguments.output_dim;
        let max_batch = max_gemv_batch_threshold();

        if !arguments.transpose_b {
            return false;
        }

        if n == 1 {
            m == 1
        } else {
            m <= max_batch
        }
    }

    fn get_or_create_gemv(
        &mut self,
        context: &MetalContext,
        specialization: gemv::Specialization,
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
                    specialization.apply_output_scale_and_accumulate,
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
                    specialization.align_m,
                    specialization.align_n,
                    specialization.align_k,
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
                    specialization.block_rows as u32,
                    specialization.block_cols as u32,
                    specialization.simdgroups_per_row as u32,
                    specialization.simdgroups_per_column as u32,
                    specialization.align_m,
                    specialization.align_n,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_gemm_mpp_direct(
        &mut self,
        context: &MetalContext,
        specialization: gemm_mpp_direct::GemmMppDirectSpecialization,
    ) -> Result<&MatmulGemmMppDirectMetalKernel, MatmulError<Metal>> {
        match self.gemm_mpp_direct_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemmMppDirectMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.block_rows as u32,
                    specialization.block_cols as u32,
                    specialization.simdgroups_per_row as u32,
                    specialization.simdgroups_per_column as u32,
                    specialization.align_m,
                    specialization.align_n,
                    specialization.align_k,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode_gemv(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let m = arguments.batch;
        let n = arguments.output_dim;

        let matrix_is_rhs = n != 1;
        let has_bias = arguments.bias.is_some();
        let apply_output_scale_and_accumulate = has_bias;

        let effective_output_dimension = if matrix_is_rhs {
            n
        } else {
            m
        };

        let transpose_matrix = if matrix_is_rhs {
            !arguments.transpose_b
        } else {
            false
        };

        let specialization = gemv::Specialization::select(
            transpose_matrix,
            arguments.input_dim,
            effective_output_dimension,
            apply_output_scale_and_accumulate,
        );

        let (alpha, beta, bias_stride) = if apply_output_scale_and_accumulate {
            (1.0f32, 1.0f32, 1i32)
        } else {
            (1.0f32, 0.0f32, 0i32)
        };

        let matrix_leading_dim = if matrix_is_rhs {
            arguments.leading_dimension_b
        } else {
            arguments.leading_dimension_a
        };

        let (matrix, matrix_offset) = if matrix_is_rhs {
            (arguments.b, 0usize)
        } else {
            (arguments.a, arguments.a_offset as usize)
        };
        let (input_vector, input_vector_offset) = if matrix_is_rhs {
            (arguments.a, arguments.a_offset as usize)
        } else {
            (arguments.b, 0usize)
        };

        let output_source = if apply_output_scale_and_accumulate {
            arguments.bias
        } else {
            None
        };

        let kernel = self.get_or_create_gemv(context, specialization)?;

        kernel.encode(
            (matrix, matrix_offset),
            (input_vector, input_vector_offset),
            output_source.map(|buffer| (buffer, 0usize)),
            &mut *arguments.d,
            arguments.input_dim,
            effective_output_dimension,
            matrix_leading_dim,
            alpha,
            beta,
            bias_stride,
            arguments.batch,
            specialization.output_rows_per_threadgroup() as i32,
            encoder,
        );

        let bias_is_fused = apply_output_scale_and_accumulate && has_bias;
        if !bias_is_fused {
            self.encode_bias_add(
                context,
                arguments.bias,
                arguments.d,
                arguments.batch as usize,
                arguments.output_dim as usize,
                encoder,
            )?;
        }

        Ok(())
    }

    pub fn encode_gemm(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let specialization = gemm::GemmSpecialization::select(context, self.data_type, &arguments);

        let threadgroups_per_row = (n + specialization.block_cols - 1) / specialization.block_cols;
        let threadgroups_per_column = (m + specialization.block_rows - 1) / specialization.block_rows;
        let swizzle_log = specialization.swizzle_log2;

        let swizzle_stride = 1_i32 << swizzle_log;
        let tm_swizzled = (threadgroups_per_column + swizzle_stride - 1) / swizzle_stride;
        let tn_swizzled = threadgroups_per_row * swizzle_stride;

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: arguments.leading_dimension_a,
            leading_dimension_b: arguments.leading_dimension_b,
            leading_dimension_d: arguments.leading_dimension_d,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log,
            aligned_inner_iterations: k / specialization.block_depth,
        };

        let group_count_x =
            u32::try_from(tn_swizzled).map_err(|_| MatmulError::<Metal>::ThreadgroupOverflow(tn_swizzled as usize))?;
        let group_count_y =
            u32::try_from(tm_swizzled).map_err(|_| MatmulError::<Metal>::ThreadgroupOverflow(tm_swizzled as usize))?;

        let kernel = self.get_or_create_gemm(context, specialization)?;

        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &mut *arguments.d,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            encoder,
        );

        self.encode_bias_add(
            context,
            arguments.bias,
            arguments.d,
            arguments.batch as usize,
            arguments.output_dim as usize,
            encoder,
        )?;

        Ok(())
    }

    fn make_mpp_params(
        arguments: &MatmulArguments<Metal>,
        specialization_block_rows: i32,
        specialization_block_cols: i32,
        swizzle_log: i32,
    ) -> (GemmParams, u32, u32) {
        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        let threadgroups_per_row = (n + specialization_block_cols - 1) / specialization_block_cols;
        let threadgroups_per_column = (m + specialization_block_rows - 1) / specialization_block_rows;

        let swizzle_stride = 1_i32 << swizzle_log;
        let tm_swizzled = (threadgroups_per_column + swizzle_stride - 1) / swizzle_stride;
        let tn_swizzled = threadgroups_per_row * swizzle_stride;

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: arguments.leading_dimension_a,
            leading_dimension_b: arguments.leading_dimension_b,
            leading_dimension_d: arguments.leading_dimension_d,
            threadgroups_per_row,
            threadgroups_per_column,
            swizzle_log,
            aligned_inner_iterations: 0,
        };

        (params, tn_swizzled as u32, tm_swizzled as u32)
    }

    pub fn encode_gemm_mpp(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let specialization = gemm_mpp::GemmMppSpecialization::select(arguments.batch, arguments.output_dim);
        let (params, group_count_x, group_count_y) = Self::make_mpp_params(
            &arguments,
            specialization.block_rows,
            specialization.block_cols,
            specialization.swizzle_log2,
        );

        let kernel = self.get_or_create_gemm_mpp(context, specialization)?;

        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &mut *arguments.d,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            encoder,
        );

        self.encode_bias_add(
            context,
            arguments.bias,
            arguments.d,
            arguments.batch as usize,
            arguments.output_dim as usize,
            encoder,
        )?;

        Ok(())
    }

    pub fn encode_gemm_mpp_direct(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let specialization = gemm_mpp_direct::GemmMppDirectSpecialization::select(
            context,
            arguments.batch,
            arguments.output_dim,
            arguments.input_dim,
        );
        let (params, group_count_x, group_count_y) = Self::make_mpp_params(
            &arguments,
            specialization.block_rows,
            specialization.block_cols,
            specialization.swizzle_log2,
        );

        let kernel = self.get_or_create_gemm_mpp_direct(context, specialization)?;

        kernel.encode(
            (arguments.a, arguments.a_offset as usize),
            arguments.b,
            &mut *arguments.d,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            encoder,
        );

        self.encode_bias_add(
            context,
            arguments.bias,
            arguments.d,
            arguments.batch as usize,
            arguments.output_dim as usize,
            encoder,
        )?;

        Ok(())
    }

    fn encode_bias_add(
        &mut self,
        context: &MetalContext,
        bias: Option<&<Metal as crate::backends::common::Backend>::Buffer>,
        output: &mut <Metal as crate::backends::common::Backend>::Buffer,
        batch: usize,
        output_dim: usize,
        encoder: &mut Encoder<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let Some(bias) = bias else {
            return Ok(());
        };
        let total_length = batch * output_dim;
        if total_length == 0 {
            return Ok(());
        }
        if self.bias_add.is_none() {
            self.bias_add =
                Some(TensorAddBiasMetalKernel::new(context, self.data_type, true).map_err(MatmulError::BackendError)?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add.encode(
            None::<&<Metal as crate::backends::common::Backend>::Buffer>,
            bias,
            output,
            output_dim as u32,
            total_length as u32,
            encoder,
        );
        Ok(())
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

        let mut kernel = Self {
            data_type,
            gemv_kernels: HashMap::new(),
            gemm_kernels: HashMap::new(),
            gemm_mpp_kernels: HashMap::new(),
            gemm_mpp_direct_kernels: HashMap::new(),
            bias_add: None,
        };

        for &config in gemv::Specialization::precompile_configs(data_type) {
            kernel.get_or_create_gemv(context, config)?;
        }
        for &config in gemm::GemmSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_gemm(context, config)?;
        }
        if context.device_capabilities().supports_mxu {
            for &config in gemm_mpp::GemmMppSpecialization::precompile_configs().iter() {
                kernel.get_or_create_gemm_mpp(context, config)?;
            }
            for config in gemm_mpp_direct::GemmMppDirectSpecialization::precompile_configs().iter() {
                kernel.get_or_create_gemm_mpp_direct(context, *config)?;
            }
        }

        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<'_, Metal>,
    ) {
        if Self::is_gemv_eligible(&arguments) {
            self.encode_gemv(context, arguments, encoder).expect("Failed to encode GEMV kernel");
        } else if context.device_capabilities().supports_mxu {
            self.encode_gemm_mpp(context, arguments, encoder).expect("Failed to encode GEMM MPP Direct kernel");
        } else {
            self.encode_gemm(context, arguments, encoder).expect("Failed to encode GEMM kernel");
        }
    }
}
