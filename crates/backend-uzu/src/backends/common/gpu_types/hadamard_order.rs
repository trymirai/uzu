#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HadamardTransformOrder {
    Input,
    Output,
}
