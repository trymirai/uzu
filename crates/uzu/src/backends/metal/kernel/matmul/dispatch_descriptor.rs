use super::{gemm, gemm_mixed_types_simple, gemm_mpp};
use crate::{
    DataType,
    backends::{
        common::kernel::{
            MatmulGemmMppKernel,
            matmul::{
                MatmulArguments, MatmulDispatchDescriptor, gemm_mpp::Specialization,
                gemm_mixed_types_simple as common_mixed_types_simple, gemv, split_k,
            },
        },
        metal::{Metal, context::MetalContext, error::MetalError, kernel::dsl::MatmulGemmMppMetalKernel},
    },
};

fn mpp_shader_supports_combo(
    context: &MetalContext,
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
) -> bool {
    let Some(config) = Specialization::precompile_configs(output_dtype).first() else {
        return false;
    };

    <MatmulGemmMppMetalKernel as MatmulGemmMppKernel>::new(
        context,
        a_dtype,
        b_dtype,
        output_dtype,
        config.block_rows as u32,
        config.block_cols as u32,
        config.block_depth as u32,
        config.warps_per_row as u32,
        config.warps_per_col as u32,
        config.align_m,
        config.align_n,
        config.align_k,
    )
    .is_ok()
}

pub fn choose_dispatch_descriptor(
    context: &MetalContext,
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
    arguments: &MatmulArguments<Metal>,
) -> Result<MatmulDispatchDescriptor, MetalError> {
    if mpp_shader_supports_combo(context, a_dtype, b_dtype, output_dtype) {
        return Ok(MatmulDispatchDescriptor::GemmMpp(gemm_mpp::DispatchDescriptor::new(output_dtype, arguments)?));
    }

    if common_mixed_types_simple::supports_combo(a_dtype, b_dtype, output_dtype) {
        return Ok(MatmulDispatchDescriptor::GemmMixedTypesSimple(gemm_mixed_types_simple::DispatchDescriptor::new(
            output_dtype,
            arguments,
        )?));
    }

    if let Some(descriptor) = gemv::DispatchDescriptor::try_new::<Metal>(output_dtype, arguments)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) = split_k::DispatchDescriptor::try_new::<Metal>(output_dtype, arguments)? {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(gemm::DispatchDescriptor::new(context, output_dtype, arguments)?))
}
