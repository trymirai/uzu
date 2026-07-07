use crate::backends::common::Backend;

pub mod matmul;

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
}

#[cfg(test)]
#[path = "../../../../unit/backends/common/kernel/mod.rs"]
mod tests;
