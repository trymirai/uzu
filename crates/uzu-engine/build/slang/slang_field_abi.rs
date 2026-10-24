use shader_slang::ScalarType;

/// One push-constant argument of a linked entry point, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    /// `(type name, stride, alignment, fields)` of a `Ptr<T>` pointee or by-value struct, fields in declaration order.
    pub layout: Option<(String, usize, usize, Vec<(String, usize, usize, ScalarType, Option<(usize, usize)>)>)>,
}
