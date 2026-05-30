#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait ManualKernels: Kernels {
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
}

pub mod matmul;
