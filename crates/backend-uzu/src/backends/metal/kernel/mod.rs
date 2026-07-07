use crate::backends::{
    common::{
        Kernels,
        gpu_types::gemm::{gemm_tiling_simdgroups_per_column, gemm_tiling_simdgroups_per_row},
    },
    metal::Metal,
};

#[path = "gdn/tree_verify/build_tree_out_dispatch_helper.rs"]
mod build_tree_out_dispatch_helper;
pub mod matmul;
#[path = "gdn/tree_verify/tree_update_solve_dispatch_helper.rs"]
mod tree_update_solve_dispatch_helper;

pub const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

include!(concat!(env!("OUT_DIR"), "/dsl.rs"));

pub struct MetalKernels;

impl Kernels for MetalKernels {
    type Backend = Metal;

    autogen_kernels!();
    type MatmulKernel = matmul::MatmulMetalKernel;
}
