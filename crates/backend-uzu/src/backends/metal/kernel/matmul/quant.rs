use super::gemm::{GemmKernel, GemmRequest};
use crate::backends::{
    common::{
        Backend, Encoder,
        kernel::{
            HadamardTransformKernel, Kernels,
            quant_matmul::{
                QuantizedGemmKernel, QuantizedMatmulArguments, QuantizedMatmulConfiguration,
                QuantizedMatmulError, QuantizedMatmulKernelEncodable,
            },
        },
    },
    metal::{Metal, context::MetalContext},
};

pub struct QuantizedGemmMetalKernel {
    gemm: GemmKernel,
    configuration: QuantizedMatmulConfiguration,
    hadamard: Option<<<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel>,
}

impl QuantizedGemmKernel for QuantizedGemmMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<Metal>> {
        let gemm = GemmKernel::new(context, configuration.data_type)
            .map_err(QuantizedMatmulError::BackendError)?;
        let hadamard = if configuration.use_hadamard {
            Some(
                <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
                    context,
                    configuration.data_type,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        } else {
            None
        };
        Ok(Self {
            gemm,
            configuration,
            hadamard,
        })
    }

    fn encode(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: QuantizedMatmulArguments<Metal>,
    ) {
        let QuantizedMatmulArguments {
            a,
            a_offset,
            b,
            scales,
            zero_points_or_biases,
            output,
            hadamard_factors,
            batch_dim,
        } = arguments;

        let output_dim = self.configuration.output_dim;

        let context = encoder.context();
        self.gemm
            .encode(
                context,
                encoder,
                GemmRequest::Quant {
                    configuration: &self.configuration,
                    arguments: QuantizedMatmulArguments {
                        a,
                        a_offset,
                        b,
                        scales,
                        zero_points_or_biases,
                        output: &mut *output,
                        hadamard_factors: None,
                        batch_dim,
                    },
                },
            )
            .expect("Failed to encode unified quantized GEMM");

        if let (Some(hadamard_kernel), Some(factors)) = (self.hadamard.as_ref(), hadamard_factors)
        {
            hadamard_kernel.encode(
                output,
                factors,
                output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatmulDispatchPath {
    Auto,
    Gemm,
}

pub fn encode_quantized_matmul_with_path(
    _context: &MetalContext,
    encodable: &QuantizedMatmulKernelEncodable<Metal>,
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
            encodable.matrix_matrix().encode(encoder, arguments);
            Ok(())
        },
    }
}
