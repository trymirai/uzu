use super::{MatmulMetalKernel, gemm};
use crate::backends::{
    common::{
        Encoder,
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
            QuantizedMatmulKernelEncodable,
        },
    },
    metal::{Metal, context::MetalContext},
};

#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatmulDispatchPath {
    Auto,
    Gemm,
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
        QuantizedMatmulDispatchPath::Auto => {
            encodable.encode(encoder, arguments);
            Ok(())
        },
        QuantizedMatmulDispatchPath::Gemm => {
            gemm::quant::encode(&mut matmul.gemm, context, configuration, arguments, encoder)
        },
    }
}
