use crate::vulkan::SlangEntryPointAbi;

pub struct SlangEntryPointVariant {
    pub types: Vec<String>,
    pub abi: SlangEntryPointAbi,
    pub spirv: String,
    pub native_bf16: Option<(SlangEntryPointAbi, String)>,
}
