pub const HADAMARD_TRANSFORM_BLOCK_SIZE: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HadamardTransformOrder {
    Input,
    Output,
}
