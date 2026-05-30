#![allow(non_snake_case)]

use crate::backends::common::Backend;

pub mod matmul;

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
}
