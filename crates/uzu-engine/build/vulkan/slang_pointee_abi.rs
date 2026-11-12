use crate::vulkan::SlangStructFieldAbi;

#[derive(Debug, PartialEq)]
pub struct SlangPointeeAbi {
    pub name: String,
    pub stride: usize,
    pub alignment: usize,
    pub fields: Vec<SlangStructFieldAbi>,
}
