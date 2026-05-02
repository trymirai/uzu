use crate::backends::{
    common::{
        Encoder,
        gpu_types::{
            QuantizationMode,
            unified_gemm::{GemmAlignment, GemmOutputTransformKind},
        },
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
            QuantizedMatmulKernelEncodable, QuantizedMatmulType,
        },
    },
    metal::{
        Metal,
        context::MetalContext,
        kernel::{
            matmul::MatmulMetalKernel,
            unified_matmul::gemm::{
                GemmComputeKind, GemmInputPrologueKind, GemmTile, UnifiedGemmSpecialization, WeightsStorageFormat,
            },
        },
    },
};

/// Explicit dispatch paths for testing quantized matmul kernels independent of production routing.
#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatmulDispatchPath {
    Auto,
    UnifiedGemm,
}

/// Test-only dispatcher for quantized matmul. The `matmul` cache is borrowed for its
/// embedded unified-gemm kernel cache, used only by the `UnifiedGemm` path.
pub fn encode_quantized_matmul_with_path(
    context: &MetalContext,
    matmul: &mut MatmulMetalKernel,
    encodable: &QuantizedMatmulKernelEncodable<Metal>,
    configuration: &QuantizedMatmulConfiguration,
    arguments: QuantizedMatmulArguments<Metal>,
    encoder: &mut Encoder<Metal>,
    path: QuantizedMatmulDispatchPath,
) -> Result<(), QuantizedMatmulError<Metal>> {
    match path {
        QuantizedMatmulDispatchPath::Auto => encodable.encode(encoder, arguments),
        QuantizedMatmulDispatchPath::UnifiedGemm => {
            let tile = select_unified_quantized_tile(configuration, arguments.batch_dim as u32);
            let specialization = build_quantized_specialization(configuration, arguments.batch_dim as u32, &tile);
            let group_count_x = (configuration.output_dim as u32).div_ceil(tile.threadgroup_n);
            let group_count_y = (arguments.batch_dim as u32).div_ceil(tile.threadgroup_m);
            matmul
                .encode_unified_gemm_quantized(
                    context,
                    specialization,
                    arguments.a_buffer,
                    arguments.a_offset,
                    arguments.b_buffer,
                    arguments.output_buffer,
                    group_count_x,
                    group_count_y,
                    encoder,
                )
                .map_err(QuantizedMatmulError::BackendError)
        },
    }
}

fn build_quantized_specialization(
    configuration: &QuantizedMatmulConfiguration,
    _batch_dim: u32,
    tile: &GemmTile,
) -> UnifiedGemmSpecialization {
    let alignment = GemmAlignment {
        m_aligned: (_batch_dim) % tile.threadgroup_m == 0,
        n_aligned: (configuration.output_dim as u32) % tile.threadgroup_n == 0,
        k_aligned: (configuration.input_dim as u32) % tile.threadgroup_k == 0,
    };

    let bits_per_weight = match configuration.mode {
        QuantizationMode::UINT4 => 4,
        QuantizationMode::INT8 | QuantizationMode::UINT8 => 8,
    };
    let group_size = configuration.group_size as u32;

    let weights_storage = match configuration.quantization_type {
        QuantizedMatmulType::Mlx => WeightsStorageFormat::QuantizedMLXScaleBias {
            bits_per_weight,
            group_size,
        },
        QuantizedMatmulType::ZeroPoint => WeightsStorageFormat::QuantizedAwqScaleZeroPoint {
            bits_per_weight,
            group_size,
        },
    };

    UnifiedGemmSpecialization::new(
        *tile,
        GemmInputPrologueKind::FullPrecision,
        GemmComputeKind::SimdgroupMma,
        GemmOutputTransformKind::Store,
        alignment,
        weights_storage,
    )
}

fn select_unified_quantized_tile(
    configuration: &QuantizedMatmulConfiguration,
    batch_dim: u32,
) -> GemmTile {
    // threadgroup_k must be ≤ group_size (validated by UnifiedGemmSpecialization).
    let group_size = configuration.group_size as u32;
    let threadgroup_k = group_size.min(32);
    let (threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) = if batch_dim < 32 {
        (32u32, 32u32, 1u32, 1u32)
    } else {
        (64u32, 64u32, 2u32, 2u32)
    };
    GemmTile {
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
