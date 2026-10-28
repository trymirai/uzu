use std::convert::Infallible;

use crate::backends::{
    common::{Buffer, GpuBuffer, SparseBuffer},
    webgpu::{WebGPU, error::WebGPUError},
};

#[derive(Debug)]
pub struct WebGPUSparseBuffer(Infallible);

impl Buffer for WebGPUSparseBuffer {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        match self.0 {}
    }
}

impl GpuBuffer for WebGPUSparseBuffer {}

impl SparseBuffer for WebGPUSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), WebGPUError> {
        match self.0 {}
    }
}
