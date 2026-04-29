use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        webgpu::{WebGPU, context::WebGPUContext},
    },
};

include!(concat!(env!("OUT_DIR"), "/webgpu.rs"));

impl ManualKernels for WebGPUKernels {
    type MatmulKernel = WebGPUMatmulKernel;
}

pub struct WebGPUMatmulKernel;

impl MatmulKernel for WebGPUMatmulKernel {
    type Backend = WebGPU;

    fn new(
        _context: &WebGPUContext,
        _data_type: DataType,
    ) -> Result<Self, MatmulError<WebGPU>> {
        Ok(Self)
    }

    fn encode(
        &mut self,
        _context: &WebGPUContext,
        _arguments: MatmulArguments<WebGPU>,
        _encoder: &mut Encoder<WebGPU>,
    ) {
        // todo!()
        // TODO
    }
}
