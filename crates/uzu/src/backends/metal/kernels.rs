use crate::backends::common::Kernels;

use super::Metal;

pub struct MetalKernels;

impl Kernels for MetalKernels {
    type Backend = Metal;
}
