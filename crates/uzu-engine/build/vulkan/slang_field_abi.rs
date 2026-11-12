use crate::vulkan::SlangPointeeAbi;

#[derive(Debug, PartialEq)]
pub struct SlangFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub layout: Option<SlangPointeeAbi>,
}
