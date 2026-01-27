use crate::backends::common::Kernels;

use super::MetalBackend;

pub struct MetalKernels;

impl Kernels for MetalKernels {
    type Backend = MetalBackend;
}
