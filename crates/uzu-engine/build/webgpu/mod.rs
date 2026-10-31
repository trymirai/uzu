use std::{
    collections::HashMap,
    fs,
    iter::once,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;
use naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use proc_macro2::TokenStream;
use shader_slang::{CompileTarget, ComponentType, GlobalSession, ModifierID, Module, TargetDesc};

use crate::{
    common::{codegen::write_tokens, identifiers::KernelName, kernel::Kernel},
    slang::{BlockingSlangCompiler, SlangCompilerFields, SlangGpuTypes, SlangKernel, SlangTarget, kernel_wrappers},
};

mod bindgen_file;
mod bindgen_global;

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
    const SOURCE_DIRECTORY: &'static str = "src/backends/slang/kernel";

    fn create_target_desc<'a>(_global_session: &'a GlobalSession) -> TargetDesc<'a> {
        TargetDesc::default().format(CompileTarget::Wgsl)
    }

    fn gpu_types_directory(fields: &SlangCompilerFields<Self>) -> PathBuf {
        fields.source_directory.join("generated")
    }

    fn wrapper(
        _fields: &SlangCompilerFields<Self>,
        source_path: &Path,
        _module: &Module,
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
        _compiler: &BlockingSlangCompiler<Self>,
        gpu_types: &SlangGpuTypes,
        module: &Module,
        kernels: &[SlangKernel],
        _descriptors: &[Kernel],
        linked: &ComponentType,
        output_base: &Path,
    ) -> anyhow::Result<()> {
        let constants = module
            .module_reflection()
            .children()
            .filter_map(|decl| decl.as_variable())
            .filter(|variable| variable.find_modifier(ModifierID::SlangModifierConst).is_some())
            .filter_map(|variable| Some((variable.name()?.to_owned(), variable.default_value_int()?.to_string())))
            .collect::<HashMap<_, _>>();

        let output_file = output_base.with_extension("wgsl");

        let linked_blob = linked.target_code(0).context("cannot get target code")?;

        fs::write(&output_file, linked_blob.as_slice()).context("cannot write output file")?;

        Self::validate_output(&output_file).with_context(|| format!("cannot validate {}", output_file.display()))?;

        let bindgen_tokens =
            bindgen_file::bindgen_file(linked, kernels, &gpu_types.map, &gpu_types.paths, &constants, &output_file)
                .context("cannot generate bindings")?;

        write_tokens(bindgen_tokens, output_base.with_extension("rs")).context("cannot write bindgen file")
    }

    fn bindgen_global(
        _fields: &SlangCompilerFields<Self>,
        modules: &[(PathBuf, &[Kernel], &[KernelName])],
    ) -> anyhow::Result<TokenStream> {
        let kernels = modules
            .iter()
            .filter(|(_, public, _)| !public.is_empty())
            .map(|(output_base, public, _)| (output_base.with_extension("rs"), *public))
            .collect::<Vec<_>>();

        bindgen_global::bindgen_global(&kernels)
    }
}
