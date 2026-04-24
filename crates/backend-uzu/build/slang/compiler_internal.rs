use std::{
    collections::HashMap,
    env,
    ffi::CString,
    fs,
    iter::once,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::TokenStream;
use serde::{Deserialize, Serialize};
use shader_slang::{
    CompilerOptions, ComponentType, GlobalSession, OptimizationLevel, Session, SessionDesc, TargetDesc,
    reflection::Shader,
};

use crate::{
    common::{caching, codegen::write_tokens, kernel::Kernel},
    debug_log,
    slang::{reflection::SlangKernel, slang_sys_ext::SessionLoadModuleWithDiagnosticsExt, wrapper},
};

pub trait SlangTarget: Send + Sync + Clone + 'static {
    const TARGET_NAME: &'static str;
    const OUTPUT_EXTENSION: &'static str;

    fn create_target_desc<'a>(global_session: &'a GlobalSession) -> TargetDesc<'a>;

    fn validate_output(path: &Path) -> anyhow::Result<()>;

    fn bindgen_file(
        shader: &Shader,
        kernels: &[SlangKernel],
        gpu_type_map: &HashMap<String, String>,
        object_file: &Path,
    ) -> anyhow::Result<TokenStream>;

    fn bindgen_global(kernels: &[(impl AsRef<Path>, &[Kernel])]) -> anyhow::Result<TokenStream>;
}

#[derive(Clone)]
pub struct SlangCompilerFields<T: SlangTarget> {
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    pub optimization_level: OptimizationLevel,
    _phantom: PhantomData<T>,
}

impl<T: SlangTarget> SlangCompilerFields<T> {
    pub fn new() -> anyhow::Result<Self> {
        let source_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/slang");

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join(T::TARGET_NAME);

        let optimization_level = match env::var("OPT_LEVEL").context("missing OPT_LEVEL")?.as_str() {
            "0" => OptimizationLevel::Default,
            _ => OptimizationLevel::High,
        };

        Ok(Self {
            source_directory,
            output_directory,
            optimization_level,
            _phantom: PhantomData,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Cached {
    buildsystem_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    public_kernels: Box<[Kernel]>,
}

pub struct BlockingSlangCompiler<T: SlangTarget> {
    session: Session,
    #[allow(unused)]
    global_session: GlobalSession,
    fields: SlangCompilerFields<T>,
}

unsafe impl<T: SlangTarget> Send for BlockingSlangCompiler<T> {}

impl<T: SlangTarget> BlockingSlangCompiler<T> {
    pub fn new(fields: SlangCompilerFields<T>) -> anyhow::Result<Self> {
        let global_session = GlobalSession::new().context("cannot create global session")?;

        let compiler_options = CompilerOptions::default()
            .optimization(fields.optimization_level)
            .emit_spirv_directly(true)
            .vulkan_use_entry_point_name(true);

        let search_path = CString::new(fields.source_directory.to_string_lossy().as_bytes())
            .context("cannot convert src_dir to CString")?;
        let search_paths = [search_path.as_ptr()];

        let target_desc = T::create_target_desc(&global_session);
        let targets = [target_desc];

        let session_desc =
            SessionDesc::default().options(&compiler_options).search_paths(&search_paths).targets(&targets);

        let session = global_session.create_session(&session_desc).context("cannot create session")?;

        Ok(Self {
            session,
            global_session,
            fields,
        })
    }

    pub fn compile(
        &self,
        source_path: PathBuf,
        gpu_type_map: Arc<HashMap<String, String>>,
    ) -> anyhow::Result<(Box<[Box<str>]>, Box<[Kernel]>)> {
        let source_path_relative =
            source_path.strip_prefix(&self.fields.source_directory).context("source is not in src_dir")?;
        let source_path_relative_str = source_path_relative.to_str().unwrap();
        debug_log!("compile start: {source_path_relative_str}");

        let source_path_relative_segments: Box<[Box<str>]> = source_path_relative
            .with_extension("")
            .components()
            .map(|s| s.as_os_str().to_str().unwrap().to_string().into_boxed_str())
            .collect();

        let output_base_path = self.fields.output_directory.join(&source_path_relative).with_extension("");
        fs::create_dir_all(output_base_path.parent().context("cannot get output directory")?)
            .context("cannot create output directory")?;

        let wrapper_file = output_base_path.with_extension("slang");
        let output_file = output_base_path.with_extension(T::OUTPUT_EXTENSION);
        let bindgen_file = output_base_path.with_extension("rs");
        let cached_file = output_base_path.with_extension("cached");

        let buildsystem_hash: &[u8; blake3::OUT_LEN] =
            caching::build_system_hash().context("cannot get build system hash")?.as_bytes();

        if let Ok(cached_contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&cached_contents)
            && &cached.buildsystem_hash == buildsystem_hash
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path)
                    .map(|dependency_contents| blake3::hash(&dependency_contents).as_bytes() == hash)
                    .unwrap_or(false)
            })
        {
            debug_log!("compile cached: {source_path_relative_str}");
            return Ok((source_path_relative_segments, cached.public_kernels));
        }

        let (source_module, source_diagnostics) =
            self.session.load_module_with_diagnostics(&source_path).context("cannot load source module")?;

        if let Some(source_diagnostics) = source_diagnostics {
            for source_diagnostics_line in source_diagnostics.lines() {
                println!("cargo::warning={source_diagnostics_line}");
            }
        }

        let dependency_hashes = source_module
            .dependency_file_paths()
            .map(|d| Ok((d.into(), blake3::hash(&fs::read(d).with_context(|| format!("cannot read {d}"))?).into())))
            .collect::<anyhow::Result<HashMap<String, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        let kernels = source_module
            .module_reflection()
            .children()
            .filter_map(|decl| SlangKernel::from_reflection(decl).transpose())
            .collect::<anyhow::Result<Vec<SlangKernel>>>()
            .context("cannot collect kernels")?;

        if !kernels.is_empty() {
            let wrapper_blocks = kernels
                .iter()
                .map(|kernel| {
                    wrapper::kernel_wrappers(kernel)
                        .with_context(|| format!("cannot generate wrappers for kernel '{}'", kernel.name))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .flatten();

            let wrapper_contents = once(format!("#include {source_path:?}")).chain(wrapper_blocks).join("\n\n");

            fs::write(&wrapper_file, wrapper_contents.as_bytes()).context("cannot write wrapper file")?;

            let (wrapper_module, wrapper_diagnostics) =
                self.session.load_module_with_diagnostics(&wrapper_file).context("cannot load wrapper module")?;

            if let Some(wrapper_diagnostics) = wrapper_diagnostics {
                for wrapper_diagnostics_line in wrapper_diagnostics.lines() {
                    println!("cargo::warning={wrapper_diagnostics_line}");
                }
            }

            let wrapper_component: ComponentType = wrapper_module.into();

            let linked_component = wrapper_component.link().context("cannot link wrapper module")?;

            let linked_blob = linked_component.target_code(0).context("cannot get target code")?;

            fs::write(&output_file, linked_blob.as_slice()).context("cannot write output file")?;

            T::validate_output(&output_file).with_context(|| format!("cannot validate {}", output_file.display()))?;

            let linked_layout = linked_component.layout(0).context("cannot get layout")?;

            let bindgen_tokens = T::bindgen_file(linked_layout, &kernels, &gpu_type_map, &output_file)
                .context("cannot generate bindings")?;

            write_tokens(bindgen_tokens, &bindgen_file).context("cannot write bindgen file")?;
        }

        let public_kernels = kernels
            .iter()
            .filter_map(|slang_kernel| {
                slang_kernel
                    .to_common(&gpu_type_map)
                    .with_context(|| format!("cannot convert {} to common kernel", slang_kernel.name))
                    .transpose()
            })
            .collect::<anyhow::Result<Box<[Kernel]>>>()?;

        fs::write(
            &cached_file,
            &serde_json::to_vec_pretty(&Cached {
                buildsystem_hash: *buildsystem_hash,
                dependency_hashes,
                public_kernels: public_kernels.clone(),
            })
            .context("cannot serialize cache")?,
        )
        .context("cannot write cache file")?;

        debug_log!("compile end: {source_path_relative_str}");

        Ok((source_path_relative_segments, public_kernels))
    }
}
