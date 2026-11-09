use std::{collections::HashMap, path::Path};

use proc_macro2::TokenStream;
use shader_slang::{ComponentType, GlobalSession, Module, TargetDesc};

use crate::{
    common::{gpu_type_paths::GpuTypePaths, kernel::Kernel},
    slang::SlangKernel,
};

pub trait SlangTarget: Send + Sync + Clone + 'static {
    const TARGET_NAME: &'static str;

    fn create_target_desc<'a>(global_session: &'a GlobalSession) -> TargetDesc<'a>;

    fn wrapper(
        source_path: &Path,
        kernels: &[SlangKernel],
    ) -> anyhow::Result<String>;

    fn emit(
        source_module: &Module,
        kernels: &[SlangKernel],
        gpu_type_map: &HashMap<String, String>,
        gpu_type_paths: &GpuTypePaths,
        linked_component: &ComponentType,
        output_base_path: &Path,
    ) -> anyhow::Result<()>;

    fn bindgen_global(kernels: &[(impl AsRef<Path>, &[Kernel])]) -> anyhow::Result<TokenStream>;
}
