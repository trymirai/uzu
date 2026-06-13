use std::{collections::HashMap, fs, path::Path};

use anyhow::Context;
use naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use proc_macro2::TokenStream;
use shader_slang::{CompileTarget, ComponentType, GlobalSession, TargetDesc};

use crate::slang::{SlangKernel, SlangTarget};

mod bindgen_file;
mod bindgen_global;

#[derive(Clone)]
pub struct WebGPU;

impl SlangTarget for WebGPU {
    const TARGET_NAME: &'static str = "webgpu";
    const OUTPUT_EXTENSION: &'static str = "wgsl";

    fn create_target_desc<'a>(_global_session: &'a GlobalSession) -> TargetDesc<'a> {
        TargetDesc::default().format(CompileTarget::Wgsl)
    }

    fn validate_output(path: &Path) -> anyhow::Result<()> {
        let source = fs::read_to_string(path).with_context(|| format!("cannot read {path:?}"))?;

        let module = wgsl::parse_str(&source)?;

        let flags = ValidationFlags::all();

        let capabilities = Capabilities::SHADER_FLOAT16
            | Capabilities::SHADER_FLOAT16_IN_FLOAT32
        // NOTE: subgroup stuff is only supported on chrome!!!!
            | Capabilities::SUBGROUP
            | Capabilities::SUBGROUP_BARRIER;

        Validator::new(flags, capabilities).validate(&module)?;

        Ok(())
    }

    fn bindgen_file(
        linked_component: &ComponentType,
        kernels: &[SlangKernel],
        gpu_type_map: &HashMap<String, String>,
        object_file: &Path,
    ) -> anyhow::Result<TokenStream> {
        bindgen_file::bindgen_file(linked_component, kernels, gpu_type_map, object_file)
    }

    fn bindgen_global(kernels: &[(impl AsRef<Path>, &[crate::common::kernel::Kernel])]) -> anyhow::Result<TokenStream> {
        bindgen_global::bindgen_global(kernels)
    }
}
