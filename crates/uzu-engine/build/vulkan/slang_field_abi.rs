use super::SlangStructAbi;

/// One push-constant argument of a linked entry point, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    /// The pointee of a `Ptr<T>` argument, or a by-value struct argument itself.
    pub layout: Option<SlangStructAbi>,
}
