pub mod gemm;
pub mod gemm_mpp;
pub mod gemv;

use std::{collections::HashMap, collections::hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            CommandBuffer,
            gpu_types::GEMMParams,
            kernel::{
                MatmulGemmKernel, MatmulGemmMppKernel, MatmulGemvKernel, TensorAddBiasKernel,
                matmul::{
                    FullPrecisionMatmulArguments,
                    MatmulKernel as MatmulKernelTrait,
                    MatmulError,
                },
            },
        },
        metal::{GpuTier, Metal, context::MetalContext},
    },
};

use super::dsl::{MatmulGemmMetalKernel, MatmulGemmMppMetalKernel, MatmulGemvMetalKernel, TensorAddBiasMetalKernel};

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv_kernels: HashMap<gemv::Specialization, MatmulGemvMetalKernel>,
    gemm_kernels: HashMap<gemm::Specialization, MatmulGemmMetalKernel>,
    gemm_mpp_kernels: HashMap<gemm_mpp::Specialization, MatmulGemmMppMetalKernel>,
    bias_add: Option<TensorAddBiasMetalKernel>,
}

// -- Level 2: per-variant public methods (testing entry points) ---------------

impl MatmulMetalKernel {
    pub fn encode_gemv(
        &mut self,
        context: &MetalContext,
        command_buffer: &mut <crate::backends::metal::command_buffer::MetalCommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let batch = arguments.batch as i32;
        let input_dim = arguments.input_dim as i32;
        let output_dim = arguments.output_dim as i32;

        let m = batch;
        let n = output_dim;

        let matrix_is_rhs = n != 1;
        let has_bias = arguments.bias.is_some();
        let apply_output_scale_and_accumulate = has_bias;
        let effective_output_dimension = if matrix_is_rhs { n } else { m };

        let specialization = gemv::Specialization::select(
            input_dim,
            effective_output_dimension,
            apply_output_scale_and_accumulate,
        );

        let kernel = self.get_or_create_gemv(context, specialization)
            .expect("Failed to create GEMV kernel");

        let (matrix, matrix_offset) = if matrix_is_rhs {
            (arguments.b, 0usize)
        } else {
            (arguments.a, arguments.a_offset)
        };
        let (input_vector, input_vector_offset) = if matrix_is_rhs {
            (arguments.a, arguments.a_offset)
        } else {
            (arguments.b, 0usize)
        };

        let output_source: Option<(&<Metal as crate::backends::common::Backend>::Buffer, usize)> =
            if apply_output_scale_and_accumulate {
                arguments.bias.map(|b| (b, 0usize))
            } else {
                None
            };

        let bias_is_fused = apply_output_scale_and_accumulate && arguments.bias.is_some();

        kernel.encode(
            (matrix, matrix_offset),
            (input_vector, input_vector_offset),
            output_source,
            &mut *arguments.output,
            input_dim,
            effective_output_dimension,
            m,
            specialization.output_rows_per_threadgroup() as i32,
            command_buffer,
        );

        if !bias_is_fused {
            self.encode_bias_add(context, arguments.bias, arguments.output, arguments.batch, arguments.output_dim, command_buffer);
        }
    }

    pub fn encode_gemm_mpp(
        &mut self,
        context: &MetalContext,
        command_buffer: &mut <crate::backends::metal::command_buffer::MetalCommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let m = arguments.batch as i32;
        let n = arguments.output_dim as i32;
        let k = arguments.input_dim as i32;

        let capabilities = context.device_capabilities();
        let specialization = gemm_mpp::Specialization::select(capabilities, m, n);

        let (params, threadgroups) = GEMMParams::with_grid(
            m, n, k,
            specialization.block_rows,
            specialization.block_cols,
            0, 0,
        );

        let kernel = self.get_or_create_gemm_mpp(context, specialization)
            .expect("Failed to create GEMM MPP kernel");

        kernel.encode(
            (arguments.a, arguments.a_offset),
            arguments.b,
            &mut *arguments.output,
            std::slice::from_ref(&params),
            threadgroups.width as u32,
            threadgroups.height as u32,
            threadgroups.depth as u32,
            command_buffer,
        );

        self.encode_bias_add(context, arguments.bias, arguments.output, arguments.batch, arguments.output_dim, command_buffer);
    }

    pub fn encode_gemm(
        &mut self,
        context: &MetalContext,
        command_buffer: &mut <crate::backends::metal::command_buffer::MetalCommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let m = arguments.batch as i32;
        let n = arguments.output_dim as i32;
        let k = arguments.input_dim as i32;

        let specialization = gemm::Specialization::select(context, self.data_type, m, n, k);

        let (params, threadgroups) = GEMMParams::with_grid(
            m, n, k,
            specialization.block_rows,
            specialization.block_cols,
            specialization.swizzle_log2,
            k / specialization.block_depth,
        );

        let kernel = self.get_or_create_gemm(context, specialization)
            .expect("Failed to create GEMM kernel");

        kernel.encode(
            (arguments.a, arguments.a_offset),
            arguments.b,
            &mut *arguments.output,
            std::slice::from_ref(&params),
            threadgroups.width as u32,
            threadgroups.height as u32,
            threadgroups.depth as u32,
            command_buffer,
        );

        self.encode_bias_add(context, arguments.bias, arguments.output, arguments.batch, arguments.output_dim, command_buffer);
    }
}

// -- Level 3: private helpers -------------------------------------------------

const MIN_GEMV_BATCH: i32 = 4;
const GEMV_CORE_SCALING_FACTOR: i32 = 256;

impl MatmulMetalKernel {
    fn is_gemv_eligible(context: &MetalContext, batch: i32, output_dim: i32) -> bool {
        let device_info = context.device_capabilities();
        let gemv_max_batch = gemv_max_batch(device_info);
        if output_dim == 1 {
            batch == 1
        } else {
            batch <= gemv_max_batch
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
                    context, self.data_type,
                    specialization.threadgroup_rows,
                    specialization.threadgroup_cols,
                    specialization.threads_per_simdgroup_row,
                    specialization.threads_per_simdgroup_col,
                    specialization.elements_per_thread_row,
                    specialization.elements_per_thread_col,
                    specialization.apply_output_scale_and_accumulate,
                ).map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_gemm(
        &mut self,
        context: &MetalContext,
        specialization: gemm::Specialization,
    ) -> Result<&MatmulGemmMetalKernel, MatmulError<Metal>> {
        match self.gemm_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemmMetalKernel::new(
                    context, self.data_type,
                    specialization.block_rows as u32,
                    specialization.block_cols as u32,
                    specialization.block_depth as u32,
                    specialization.warps_per_row as u32,
                    specialization.warps_per_col as u32,
                    specialization.align_m,
                    specialization.align_n,
                    specialization.align_k,
                ).map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_gemm_mpp(
        &mut self,
        context: &MetalContext,
        specialization: gemm_mpp::Specialization,
    ) -> Result<&MatmulGemmMppMetalKernel, MatmulError<Metal>> {
        match self.gemm_mpp_kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemmMppMetalKernel::new(
                    context, self.data_type,
                    specialization.block_rows as u32,
                    specialization.block_cols as u32,
                    specialization.block_depth as u32,
                    specialization.warps_per_row as u32,
                    specialization.warps_per_col as u32,
                    specialization.align_m,
                    specialization.align_n,
                    specialization.align_k,
                    specialization.use_native_fragment_layout,
                    specialization.subtile_rows,
                    specialization.subtile_cols,
                    specialization.matmul_k_step,
                ).map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn encode_bias_add(
        &mut self,
        context: &MetalContext,
        bias: Option<&<Metal as crate::backends::common::Backend>::Buffer>,
        output: &mut <Metal as crate::backends::common::Backend>::Buffer,
        batch: usize,
        output_dim: usize,
        command_buffer: &mut <crate::backends::metal::command_buffer::MetalCommandBuffer as CommandBuffer>::Encoding,
    ) {
        let Some(bias) = bias else { return };
        let total_length = batch * output_dim;
        if total_length == 0 {
            return;
        }
        if self.bias_add.is_none() {
            self.bias_add = Some(
                TensorAddBiasMetalKernel::new(context, self.data_type, true)
                    .expect("Failed to create bias add kernel"),
            );
        }
        let bias_add = self.bias_add.as_ref().expect("bias_add initialized above");
        bias_add.encode(
            None::<&<Metal as crate::backends::common::Backend>::Buffer>,
            bias,
            &mut *output,
            output_dim as u32,
            total_length as u32,
            command_buffer,
        );
    }
}

fn default_gemv_max_batch(device_info: &crate::backends::metal::MetalDeviceCapabilities) -> i32 {
    if device_info.tier == GpuTier::Phone {
        return MIN_GEMV_BATCH;
    }
    let core_count = device_info.gpu_core_count.max(1) as i32;
    (GEMV_CORE_SCALING_FACTOR / core_count).max(MIN_GEMV_BATCH)
}

fn gemv_max_batch(device_info: &crate::backends::metal::MetalDeviceCapabilities) -> i32 {
    if let Ok(val) = std::env::var("UZU_GEMV_MAX_BATCH") {
        if let Ok(n) = val.parse() {
            return n;
        }
    }
    default_gemv_max_batch(device_info)
}

// -- Level 1: MatmulKernel trait impl (production entry point) ----------------

impl MatmulKernelTrait for MatmulMetalKernel {
    type Backend = Metal;

    fn new(context: &MetalContext, data_type: DataType) -> Result<Self, MatmulError<Metal>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        let mut kernel = Self {
            data_type,
            gemv_kernels: HashMap::new(),
            gemm_kernels: HashMap::new(),
            gemm_mpp_kernels: HashMap::new(),
            bias_add: None,
        };

        let capabilities = context.device_capabilities();

        for &config in gemv::Specialization::precompile_configs(data_type) {
            kernel.get_or_create_gemv(context, config)?;
        }
        for config in gemm_mpp::Specialization::precompile_configs(capabilities).iter() {
            kernel.get_or_create_gemm_mpp(context, *config)?;
        }
        for config in gemm::Specialization::precompile_configs(capabilities).iter() {
            kernel.get_or_create_gemm(context, *config)?;
        }

        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &MetalContext,
        command_buffer: &mut <crate::backends::metal::command_buffer::MetalCommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let m = arguments.batch as i32;
        let n = arguments.output_dim as i32;

        if Self::is_gemv_eligible(context, m, n) {
            self.encode_gemv(context, command_buffer, arguments);
        } else if context.device_capabilities().supports_mxu {
            self.encode_gemm_mpp(context, command_buffer, arguments);
        } else {
            self.encode_gemm(context, command_buffer, arguments);
        }
    }
}

#[cfg(test)]
mod tests {
    use byte_unit::Byte;

    use super::default_gemv_max_batch;
    use crate::backends::metal::{DeviceGeneration, MetalDeviceCapabilities, GpuTier};

    fn device_info_with(tier: GpuTier, core_count: u32) -> MetalDeviceCapabilities {
        MetalDeviceCapabilities {
            generation: DeviceGeneration::Unknown(0),
            tier,
            family_name: String::new(),
            gpu_core_count: core_count,
            max_threadgroup_memory: Byte::default(),
            shared_memory_size: Byte::from_u64(0),
            supports_simd_group: true,
            supports_simd_group_matrix: true,
            supports_simd_reduction: true,
            supports_simd_shuffle_and_fill: true,
            supports_simd_shuffles_and_broadcast: true,
            supports_mxu: false,
            supports_tls: true,
        }
    }

    #[test]
    fn phone_always_returns_4() {
        assert_eq!(default_gemv_max_batch(&device_info_with(GpuTier::Phone, 5)), 4);
        assert_eq!(default_gemv_max_batch(&device_info_with(GpuTier::Phone, 38)), 4);
    }

    #[test]
    fn more_cores_means_lower_cutoff() {
        let base_8 = default_gemv_max_batch(&device_info_with(GpuTier::Base, 8));
        let base_10 = default_gemv_max_batch(&device_info_with(GpuTier::Base, 10));
        let max_38 = default_gemv_max_batch(&device_info_with(GpuTier::Max, 38));

        assert!(base_8 > base_10);
        assert!(base_10 > max_38);
        assert!(max_38 >= 4);
    }

    #[test]
    fn minimum_is_4() {
        assert_eq!(default_gemv_max_batch(&device_info_with(GpuTier::Max, 128)), 4);
    }
}
