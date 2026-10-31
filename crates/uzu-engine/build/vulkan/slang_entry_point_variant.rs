use super::SlangEntryPointAbi;

/// One entry point of a kernel, built for the data types `types`.
pub struct SlangEntryPointVariant {
    pub types: Vec<&'static str>,
    pub abi: SlangEntryPointAbi,
    /// The SPIR-V file.
    pub spirv: String,
    /// The ABI and SPIR-V file built with the native BF16 type, where that SPIR-V differs.
    pub native_bf16: Option<(SlangEntryPointAbi, String)>,
}
