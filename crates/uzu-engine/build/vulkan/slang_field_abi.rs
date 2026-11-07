use super::SlangPointeeAbi;

/// One push-constant argument of a linked entry point, laid out by Slang.
#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub layout: Option<SlangPointeeAbi>,
}
