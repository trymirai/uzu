use std::{
    collections::HashMap,
    env,
    ffi::CString,
    fs,
    iter::once,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use anyhow::Context;
use proc_macro2::TokenStream;
use serde::{Deserialize, Serialize};
use shader_slang::{
    CompilerOptions, ComponentType, GlobalSession, Module, OptimizationLevel, Session, SessionDesc, TargetDesc,
};

use crate::{
    common::{caching, identifiers::KernelName, kernel::Kernel},
    debug_log,
    slang::{SlangGpuTypes, reflection::SlangKernel, slang_sys_ext::SessionLoadModuleWithDiagnosticsExt},
};

pub trait SlangTarget: Send + Sync + Clone + 'static {
    const TARGET_NAME: &'static str;
    /// The kernel sources, relative to the crate.
    const SOURCE_DIRECTORY: &'static str;
    /// The backend whose own GPU types the kernels name besides the common ones.
    const PRIVATE_GPU_TYPES: Option<&'static str> = None;
    /// The preprocessor definitions of each session; the first session links every wrapper.
    const SESSION_DEFINES: &'static [&'static [(&'static str, &'static str)]] = &[&[]];

    fn compiler_options(
        _global_session: &GlobalSession,
        options: CompilerOptions,
    ) -> CompilerOptions {
        options
    }

    fn create_target_desc<'a>(global_session: &'a GlobalSession) -> TargetDesc<'a>;

    /// The directory of the generated GPU type files, which the kernels import as `generated.<file>`.
    fn gpu_types_directory(fields: &SlangCompilerFields<Self>) -> PathBuf;

    /// The source of the wrapper that declares the entry points of `kernels`, defined by `module` at `source_path`.
    fn wrapper(
        fields: &SlangCompilerFields<Self>,
        source_path: &Path,
        module: &Module,
        kernels: &[SlangKernel],
    ) -> anyhow::Result<String>;

    /// Writes the target code and bindings of `kernels`, described in order by `descriptors`, from `linked`, their
    /// wrapper at `output_base`.slang linked with its entry points in the first session.
    fn emit(
        compiler: &BlockingSlangCompiler<Self>,
        gpu_types: &SlangGpuTypes,
        module: &Module,
        kernels: &[SlangKernel],
        descriptors: &[Kernel],
        linked: &ComponentType,
        output_base: &Path,
    ) -> anyhow::Result<()>;

    /// The bindings of every module with kernels, by output base: its public kernels and private kernel names.
    fn bindgen_global(
        fields: &SlangCompilerFields<Self>,
        modules: &[(PathBuf, &[Kernel], &[KernelName])],
    ) -> anyhow::Result<TokenStream>;
}

#[derive(Clone)]
pub struct SlangCompilerFields<T: SlangTarget> {
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    pub optimization_level: OptimizationLevel,
    /// The build system's hash with the optimization level, which the cache records check.
    pub buildsystem_hash: [u8; blake3::OUT_LEN],
    _phantom: PhantomData<T>,
}

impl<T: SlangTarget> SlangCompilerFields<T> {
    pub fn new() -> anyhow::Result<Self> {
        let source_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join(T::SOURCE_DIRECTORY);

        println!("cargo::rerun-if-changed={}", source_directory.display());

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join(T::TARGET_NAME);

        let opt_level = env::var("OPT_LEVEL").context("missing OPT_LEVEL")?;
        let optimization_level = match opt_level.as_str() {
            "0" => OptimizationLevel::Default,
            _ => OptimizationLevel::High,
        };

        let buildsystem_hash = *blake3::Hasher::new()
            .update(caching::build_system_hash().context("cannot get build system hash")?.as_bytes())
            .update(opt_level.as_bytes())
            .finalize()
            .as_bytes();

        Ok(Self {
            source_directory,
            output_directory,
            optimization_level,
            buildsystem_hash,
            _phantom: PhantomData,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Cached {
    buildsystem_hash: [u8; blake3::OUT_LEN],
    compiler_version: String,
    dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    public_kernels: Box<[Kernel]>,
    /// Kernels bound without a common trait.
    private_kernels: Box<[KernelName]>,
}

pub struct BlockingSlangCompiler<T: SlangTarget> {
    sessions: Vec<Session>,
    global_session: GlobalSession,
    fields: SlangCompilerFields<T>,
}

unsafe impl<T: SlangTarget> Send for BlockingSlangCompiler<T> {}

impl<T: SlangTarget> BlockingSlangCompiler<T> {
    pub fn new(fields: SlangCompilerFields<T>) -> anyhow::Result<Self> {
        let global_session = GlobalSession::new().context("cannot create global session")?;

        // The kernel sources, and the output when the target generates its GPU types there.
        let generated_in_output = T::gpu_types_directory(&fields).starts_with(&fields.output_directory);
        let search_path_strings = once(&fields.source_directory)
            .chain(generated_in_output.then_some(&fields.output_directory))
            .map(|path| {
                CString::new(path.to_string_lossy().as_bytes()).context("cannot convert search path to CString")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let search_paths = search_path_strings.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();

        let sessions = T::SESSION_DEFINES
            .iter()
            .map(|defines| {
                let compiler_options = CompilerOptions::default()
                    .optimization(fields.optimization_level)
                    .emit_spirv_directly(true)
                    .vulkan_use_entry_point_name(true);
                let compiler_options = defines
                    .iter()
                    .fold(T::compiler_options(&global_session, compiler_options), |compiler_options, (name, value)| {
                        compiler_options.macro_define(name, value)
                    });

                let targets = [T::create_target_desc(&global_session)];

                let session_desc =
                    SessionDesc::default().options(&compiler_options).search_paths(&search_paths).targets(&targets);

                global_session.create_session(&session_desc).context("cannot create session")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            sessions,
            global_session,
            fields,
        })
    }

    /// The wrapper module at `path` linked in session `session` with its entry points, which must be composed
    /// explicitly for the linked program to reflect them.
    pub fn link(
        &self,
        session: usize,
        path: &Path,
    ) -> anyhow::Result<ComponentType> {
        let session = &self.sessions[session];
        let (wrapper_module, wrapper_diagnostics) =
            session.load_module_with_diagnostics(path).context("cannot load wrapper module")?;

        for wrapper_diagnostics_line in wrapper_diagnostics.iter().flat_map(|diagnostics| diagnostics.lines()) {
            println!("cargo::warning={wrapper_diagnostics_line}");
        }

        let entry_points = wrapper_module.entry_points().map(ComponentType::from).collect::<Vec<_>>();

        let composite_component = session
            .create_composite_component_type(&once(wrapper_module.into()).chain(entry_points).collect::<Vec<_>>())
            .context("cannot create composite component")?;

        composite_component.link().context("cannot link wrapper module")
    }

    /// Reruns this script when a dependency outside the kernel sources, which it watches, and the output, which it
    /// writes, changes: a definition another target's sources share.
    fn rerun_if_changed<'a>(
        &self,
        dependencies: impl IntoIterator<Item = &'a String>,
    ) {
        for dependency in dependencies {
            let path = Path::new(dependency);
            if !path.starts_with(&self.fields.source_directory) && !path.starts_with(&self.fields.output_directory) {
                println!("cargo::rerun-if-changed={dependency}");
            }
        }
    }

    pub fn compile(
        &self,
        source_path: PathBuf,
        gpu_types: &SlangGpuTypes,
    ) -> anyhow::Result<(Box<[Box<str>]>, Box<[Kernel]>, Box<[KernelName]>)> {
        let source_path_relative =
            source_path.strip_prefix(&self.fields.source_directory).context("source is not in src_dir")?;
        let source_path_relative_str = source_path_relative.to_str().unwrap();
        debug_log!("compile start: {source_path_relative_str}");

        let source_path_relative_segments: Box<[Box<str>]> = source_path_relative
            .with_extension("")
            .components()
            .map(|s| s.as_os_str().to_str().unwrap().to_string().into_boxed_str())
            .collect();

        let output_base_path = self.fields.output_directory.join(source_path_relative).with_extension("");
        fs::create_dir_all(output_base_path.parent().context("cannot get output directory")?)
            .context("cannot create output directory")?;

        let wrapper_file = output_base_path.with_extension("slang");
        let cached_file = output_base_path.with_extension("cached");

        let buildsystem_hash = &self.fields.buildsystem_hash;

        if let Ok(cached_contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&cached_contents)
            && &cached.buildsystem_hash == buildsystem_hash
            && cached.compiler_version == self.global_session.build_tag_string()
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path)
                    .map(|dependency_contents| blake3::hash(&dependency_contents).as_bytes() == hash)
                    .unwrap_or(false)
            })
        {
            debug_log!("compile cached: {source_path_relative_str}");
            self.rerun_if_changed(cached.dependency_hashes.keys());
            return Ok((source_path_relative_segments, cached.public_kernels, cached.private_kernels));
        }

        let (source_module, source_diagnostics) =
            self.sessions[0].load_module_with_diagnostics(&source_path).context("cannot load source module")?;

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
        self.rerun_if_changed(dependency_hashes.keys());

        let common_gpu_type_map = gpu_types.map.keys().map(|name| (name.clone(), name.clone())).collect();
        let kernels = source_module
            .module_reflection()
            .children()
            .filter_map(|decl| SlangKernel::from_reflection(decl, &common_gpu_type_map).transpose())
            .collect::<anyhow::Result<Vec<SlangKernel>>>()
            .context("cannot collect kernels")?;
        let descriptors = kernels
            .iter()
            .map(|slang_kernel| {
                slang_kernel
                    .to_common(&common_gpu_type_map)
                    .with_context(|| format!("cannot convert {} to common kernel", slang_kernel.name))
            })
            .collect::<anyhow::Result<Vec<Kernel>>>()?;

        if !kernels.is_empty() {
            let wrapper_contents =
                T::wrapper(&self.fields, &source_path, &source_module, &kernels).context("cannot generate wrappers")?;

            fs::write(&wrapper_file, wrapper_contents.as_bytes()).context("cannot write wrapper file")?;

            let linked_component = self.link(0, &wrapper_file)?;

            T::emit(self, gpu_types, &source_module, &kernels, &descriptors, &linked_component, &output_base_path)
                .context("cannot emit kernels")?;
        }

        let public_kernels = kernels
            .iter()
            .zip(descriptors)
            .filter_map(|(slang_kernel, descriptor)| slang_kernel.public.then_some(descriptor))
            .collect::<Box<[Kernel]>>();
        let private_kernels = kernels
            .iter()
            .filter(|slang_kernel| !slang_kernel.public)
            .map(|slang_kernel| KernelName::from(slang_kernel.name.as_str()))
            .collect::<Box<[KernelName]>>();

        fs::write(
            &cached_file,
            &serde_json::to_vec_pretty(&Cached {
                buildsystem_hash: *buildsystem_hash,
                compiler_version: self.global_session.build_tag_string().to_owned(),
                dependency_hashes,
                public_kernels: public_kernels.clone(),
                private_kernels: private_kernels.clone(),
            })
            .context("cannot serialize cache")?,
        )
        .context("cannot write cache file")?;

        debug_log!("compile end: {source_path_relative_str}");

        Ok((source_path_relative_segments, public_kernels, private_kernels))
    }
}
