/// One push-constant argument of a linked entry point, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    /// Reflected `(type name, stride, alignment)` of the pointee for `Ptr<T>` arguments.
    pub pointee: Option<(String, usize, usize)>,
}
