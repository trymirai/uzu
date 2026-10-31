use super::SlangStructFieldAbi;

/// A `Ptr<T>` pointee or by-value struct argument, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangPointeeAbi {
    pub name: String,
    pub stride: usize,
    pub alignment: usize,
    /// In declaration order; empty for a scalar pointee.
    pub fields: Vec<SlangStructFieldAbi>,
}
