pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel},
        },
        metal::{Metal, context::MetalContext, error::MetalError, metal_extensions::DeviceExt},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvDispatch::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let prefer_mxu_gemm = encoder.context().device.supports_mxu() && self.gemm.prefers_mxu_over_gemv(&arguments);
        if !prefer_mxu_gemm
            && let Some(gemv) = GemvSpecialization::select(
                &arguments,
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
                encoder.context().device_tier(),
            )
        {
            return self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from);
        }
        self.gemm.encode(arguments, encoder)
    }
}
