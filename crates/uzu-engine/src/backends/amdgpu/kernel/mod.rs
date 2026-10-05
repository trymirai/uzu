use std::ffi::c_void;

// Used by the GEMM bindings, which need the WMMA path to compile.
#[allow(unused_imports)]
use crate::backends::common::gpu_types::gemm::{gemm_tiling_simdgroups_per_column, gemm_tiling_simdgroups_per_row};
use crate::{
    backends::{
        amdgpu::Amdgpu,
        common::{
            Kernels,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                weaver::{FRONTIER_SELECT_THREADS, TOP_CHILDREN_THREADS},
            },
            kernel::Unsupported,
        },
    },
    data_type::DataType,
};

/// MSL kernels assume 32-wide SIMD groups; gfx11 runs them as wave32.
const METAL_SIMD_SIZE: u32 = 32;

const _: () = {
    assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE);
};

mod attention;
pub(crate) mod matmul;
mod radix_top_k_small;
mod tree_verify;

include!(concat!(env!("OUT_DIR"), "/amdgpu.rs"));

/// Code objects of the AMD-only kernels in `kernel/native` (`build/amdgpu/native.rs`); empty when the source
/// did not compile.
pub(in crate::backends::amdgpu) mod native {
    include!(concat!(env!("OUT_DIR"), "/amdgpu_native.rs"));
}

pub struct AmdgpuKernels;

impl Kernels for AmdgpuKernels {
    type Backend = Amdgpu;

    autogen_kernels!();
    type AttentionKernel = attention::AmdgpuAttentionKernel;
    type DeltaNetChunkedPrefill = Unsupported<Amdgpu>;
    type DeltaNetTreeVerify = tree_verify::AmdgpuDeltaNetTreeVerify;
    type MatmulKernel = matmul::AmdgpuMatmulKernel;
    type RadixTopKSmall = radix_top_k_small::AmdgpuRadixTopKSmall;
}

/// A kernel entry point in a loaded code object.
#[derive(Clone, Copy, Debug)]
pub struct AmdgpuFunction(pub(in crate::backends::amdgpu) *mut c_void);

// HIP function handles are immutable runtime objects.
unsafe impl Send for AmdgpuFunction {}
unsafe impl Sync for AmdgpuFunction {}

/// Explicit kernel arguments in the order the generated `__kernel` wrapper declares them
/// (natural alignment, as in the AMDGPU kernarg segment).
#[derive(Default)]
pub struct Kernarg {
    bytes: Vec<u8>,
}

impl Kernarg {
    pub fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(128),
        }
    }

    fn align(
        &mut self,
        alignment: usize,
    ) {
        let aligned = self.bytes.len().next_multiple_of(alignment);
        self.bytes.resize(aligned, 0);
    }

    pub fn push_address(
        &mut self,
        address: u64,
    ) {
        self.align(8);
        self.bytes.extend_from_slice(&address.to_le_bytes());
    }

    pub fn push_u32(
        &mut self,
        value: u32,
    ) {
        self.align(4);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(in crate::backends::amdgpu) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(in crate::backends::amdgpu) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }
}

/// MSL spelling of a data type, used in kernel variant names (as `MetalDataTypeExt` on Metal).
pub trait MetalDataTypeExt {
    fn metal_type(&self) -> &'static str;
}

impl MetalDataTypeExt for DataType {
    fn metal_type(&self) -> &'static str {
        match self {
            DataType::F16 => "half",
            DataType::BF16 => "bfloat",
            DataType::F32 => "float",
            other => panic!("{other:?} is not a kernel data type"),
        }
    }
}

/// Shard of a kernel variant; must match `build/amdgpu/sharding.rs`.
pub fn shard_index(
    entry_name: &str,
    num_shards: usize,
) -> usize {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in entry_name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash % num_shards as u64) as usize
}
