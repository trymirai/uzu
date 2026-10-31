use super::SlangStructFieldAbi;

/// A `Ptr<T>` pointee or by-value struct argument, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangStructAbi {
    pub name: String,
    pub stride: usize,
    pub alignment: usize,
    /// In declaration order.
    pub fields: Vec<SlangStructFieldAbi>,
}
