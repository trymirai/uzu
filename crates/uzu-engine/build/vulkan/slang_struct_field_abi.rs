use shader_slang::ScalarType;

/// A scalar or scalar-array field of a struct, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangStructFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub scalar: ScalarType,
    /// `(element count, element stride)` of an array.
    pub array: Option<(usize, usize)>,
}
