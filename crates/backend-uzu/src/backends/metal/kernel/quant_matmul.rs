use crate::backends::{
    common::{
        Encoder,
        gpu_types::{
            QuantizationMethod,
            unified_gemm::{GemmAlignment, GemmOutputTransformKind},
        },
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
            QuantizedMatmulKernelEncodable,
        },
    },
    metal::{
        Metal,
        context::MetalContext,
        kernel::{
            matmul::MatmulMetalKernel,
            unified_matmul::gemm::{
                GemmComputeKind, GemmInputPrologueKind, GemmTilingConfig, GemmWeights, UnifiedGemmDispatch,
            },
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatmulDispatchPath {
    Auto,
    UnifiedGemm,
}

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
            let group_size = configuration.group_size as u32;
            let dispatch = UnifiedGemmDispatch {
                tiling_config: tile,
                input_prologue: GemmInputPrologueKind::FullPrecision,
                compute: GemmComputeKind::SimdgroupMma,
                output_transform: GemmOutputTransformKind::Store,
                alignment: GemmAlignment {
                    m_aligned: (arguments.batch_dim as u32) % tile.threadgroup_m == 0,
                    n_aligned: (configuration.output_dim as u32) % tile.threadgroup_n == 0,
                    k_aligned: (configuration.input_dim as u32) % tile.threadgroup_k == 0,
                },
                weights: match configuration.quantization_method {
                    QuantizationMethod::MLX => GemmWeights::Mlx {
                        weights: arguments.b_buffer,
                        scales: arguments.scales_buffer,
                        biases: arguments.zero_points_or_biases_buffer,
                        mode: configuration.mode,
                        group_size,
                    },
                    QuantizationMethod::AWQ => GemmWeights::Awq {
                        weights: arguments.b_buffer,
                        scales: arguments.scales_buffer,
                        zero_points: arguments.zero_points_or_biases_buffer,
                        mode: configuration.mode,
                        group_size,
                    },
                },
                activations: arguments.a_buffer,
                activations_offset: arguments.a_offset,
                result: arguments.output_buffer,
                group_count_x: (configuration.output_dim as u32).div_ceil(tile.threadgroup_n),
                group_count_y: (arguments.batch_dim as u32).div_ceil(tile.threadgroup_m),
            };
            matmul.encode_unified_gemm(context, dispatch, encoder).map_err(QuantizedMatmulError::BackendError)
        },
    }
}

fn select_unified_quantized_tile(
    configuration: &QuantizedMatmulConfiguration,
    batch_dim: u32,
) -> GemmTilingConfig {
    let group_size = configuration.group_size as u32;
    let threadgroup_k = group_size.min(32);
    let (threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) = if batch_dim < 32 {
        (32u32, 32u32, 1u32, 1u32)
    } else {
        (64u32, 64u32, 2u32, 2u32)
    };
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
