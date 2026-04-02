use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::GemmParams,
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{MatmulGemmMetalKernel, MatmulGemvMetalKernel, TensorAddBiasMetalKernel},
        },
    },
};

mod gemm;
mod gemv;

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv_kernels: HashMap<gemv::GemvSpecialization, MatmulGemvMetalKernel>,
    gemm_kernels: HashMap<gemm::GemmSpecialization, MatmulGemmMetalKernel>,
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
                None::<&<Metal as crate::backends::common::Backend>::Buffer>,
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
            bias_add,
        };

        for &config in gemv::GemvSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_gemv(context, config)?;
        }
        for &config in gemm::GemmSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_gemm(context, config)?;
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
        } else {
            self.encode_gemm(context, encoder, arguments).expect("Failed to encode GEMM kernel");
        }
    }
}
