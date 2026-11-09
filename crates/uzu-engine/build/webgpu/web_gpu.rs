use std::{collections::HashMap, fs, iter::once, path::Path};

use anyhow::Context;
use itertools::Itertools;
use naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use proc_macro2::TokenStream;
use shader_slang::{CompileTarget, ComponentType, GlobalSession, ModifierID, Module, TargetDesc};

use crate::{
    common::{codegen::write_tokens, gpu_type_paths::GpuTypePaths, kernel::Kernel},
    slang::{SlangKernel, SlangTarget, kernel_wrappers},
    webgpu::{bindgen_file::bindgen_file, bindgen_global::bindgen_global},
};

#[derive(Clone)]
pub struct WebGPU;

impl WebGPU {
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
}

impl SlangTarget for WebGPU {
    const TARGET_NAME: &'static str = "webgpu";

    fn create_target_desc<'a>(_global_session: &'a GlobalSession) -> TargetDesc<'a> {
        TargetDesc::default().format(CompileTarget::Wgsl)
    }

    fn wrapper(
        source_path: &Path,
        kernels: &[SlangKernel],
    ) -> anyhow::Result<String> {
        let wrapper_blocks = kernels
            .iter()
            .map(|kernel| {
                kernel_wrappers(kernel)
                    .with_context(|| format!("cannot generate wrappers for kernel '{}'", kernel.name))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten();

        Ok(once(format!("#include {source_path:?}")).chain(wrapper_blocks).join("\n\n"))
    }

    fn emit(
        source_module: &Module,
        kernels: &[SlangKernel],
        gpu_type_map: &HashMap<String, String>,
        gpu_type_paths: &GpuTypePaths,
        linked_component: &ComponentType,
        output_base_path: &Path,
    ) -> anyhow::Result<()> {
        let constants = source_module
            .module_reflection()
            .children()
            .filter_map(|decl| decl.as_variable())
            .filter(|variable| variable.find_modifier(ModifierID::SlangModifierConst).is_some())
            .filter_map(|variable| Some((variable.name()?.to_owned(), variable.default_value_int()?.to_string())))
            .collect();

        let output_file = output_base_path.with_extension("wgsl");
        let linked_blob = linked_component.target_code(0).context("cannot get target code")?;

        fs::write(&output_file, linked_blob.as_slice()).context("cannot write output file")?;

        Self::validate_output(&output_file).with_context(|| format!("cannot validate {}", output_file.display()))?;

        let bindgen_tokens =
            bindgen_file(linked_component, kernels, gpu_type_map, gpu_type_paths, &constants, &output_file)
                .context("cannot generate bindings")?;

        write_tokens(bindgen_tokens, output_base_path.with_extension("rs")).context("cannot write bindgen file")
    }

    fn bindgen_global(kernels: &[(impl AsRef<Path>, &[Kernel])]) -> anyhow::Result<TokenStream> {
        bindgen_global(kernels)
    }
}
