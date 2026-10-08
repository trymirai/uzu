use shader_slang::ScalarType;

/// One push-constant argument of a linked entry point, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    /// Reflected `(type name, stride, alignment, fields)` of the pointee for `Ptr<T>` arguments, or of the struct itself
    /// for by-value struct arguments, where a struct's `fields` are its `(name, offset, size, scalar type, array (count,
    /// stride))` in declaration order.
    pub layout: Option<(String, usize, usize, Vec<(String, usize, usize, ScalarType, Option<(usize, usize)>)>)>,
}
