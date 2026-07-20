use super::super::METAL_SIMD_SIZE;

pub const HADAMARD_TRANSFORM_BLOCK_SIZE: usize = 32;

const _: () = assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE as usize);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HadamardTransformOrder {
    Input,
    Output,
}
