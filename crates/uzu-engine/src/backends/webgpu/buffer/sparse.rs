use std::convert::Infallible;

use crate::backends::{
    common::{Buffer, SparseBuffer},
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

impl SparseBuffer for WebGPUSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), WebGPUError> {
        match self.0 {}
    }
}
