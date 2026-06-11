pub mod gemm;
pub mod gemv;

use self::gemv::{GemvDispatch, GemvSpecialization};
pub use self::{
    gemm::{GemmDispatchPath, GemmKernel},
    gemv::GemvDispatchPath,
};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            kernel::matmul::{MatmulArguments, MatmulError, MatmulKernel},
        },
        metal::{
            Metal,
            context::{GpuDeviceTier, MetalContext},
            error::MetalError,
        },
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    device_tier: GpuDeviceTier,
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
        let gemv = GemvDispatch::new(context, weights_data_type, input_data_type, output_data_type)
            .map_err(MetalError::from)?;

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
            device_tier: context.gpu_device_tier(),
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        match GemvSpecialization::select(
            &arguments,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            self.device_tier,
        ) {
            Some(spec) => self.gemv.encode(arguments, spec, encoder).map_err(MetalError::from),
            None => self.gemm.encode(arguments, encoder),
        }
    }
}

impl MatmulMetalKernel {
    pub fn encode_gemv_dispatch_path<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        path: GemvDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let specialization = GemvSpecialization::select(
            &arguments,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            self.device_tier,
        )
        .ok_or(MatmulError::<Metal>::UnsupportedLayout {
            path: "Gemv",
        })?
        .with_dispatch_path(path);
        self.gemv.encode(arguments, specialization, encoder).map_err(MetalError::from)
    }
}
