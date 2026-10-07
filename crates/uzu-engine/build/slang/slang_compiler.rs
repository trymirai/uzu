use std::{
    collections::HashMap,
    env,
    ffi::CString,
    fs,
    iter::once,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;
use shader_slang::{
    CompileTarget, CompilerOptions, GlobalSession, OptimizationLevel, Session, SessionDesc, TargetDesc,
};
use walkdir::WalkDir;

use super::{Dephashes, Error, SlangKernelInfo, slang_api, wrapper};
use crate::{
    common::{
        caching, compiler::Compiler, enum_paths::EnumPaths, gpu_types::GpuTypes, identifiers::KernelPath,
        kernel::Kernel,
    },
    debug_log,
};

pub struct SlangCompiler {
    session: Session,
    _global_session: GlobalSession,
    src_dir: PathBuf,
    out_dir: PathBuf,
}

impl SlangCompiler {
    pub fn new() -> Result<Self, Error> {
        let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src/backends/vulkan/kernel");
        let out_dir = PathBuf::from(env::var("OUT_DIR")?).join("vulkan");
        let optimization = match env::var("OPT_LEVEL")?.as_str() {
            "0" => OptimizationLevel::Default,
            _ => OptimizationLevel::High,
        };
        let global_session = GlobalSession::new().context("cannot create Slang global session")?;
        let options = CompilerOptions::default()
            .optimization(optimization)
            .emit_spirv_directly(true)
            .vulkan_use_entry_point_name(true);
        let search_path = CString::new(src_dir.to_string_lossy().as_bytes())?;
        let search_paths = [search_path.as_ptr()];
        let targets =
            [TargetDesc::default().format(CompileTarget::Spirv).profile(global_session.find_profile("glsl_450"))];
        let desc = SessionDesc::default().options(&options).search_paths(&search_paths).targets(&targets);
        let session = global_session.create_session(&desc).context("cannot create Slang session")?;

        Ok(Self {
            session,
            _global_session: global_session,
            src_dir,
            out_dir,
        })
    }

    fn compile(
        &self,
        source_file: &Path,
    ) -> Result<(KernelPath, Box<[Kernel]>), Error> {
        let source_relative = source_file.strip_prefix(&self.src_dir)?.with_extension("");
        let kernel_path = source_relative
            .components()
            .map(|component| component.as_os_str().to_str().map(str::to_owned))
            .collect::<Option<KernelPath>>()
            .context("Slang source path is not UTF-8")?;
        let output_base = self.out_dir.join(&source_relative);
        fs::create_dir_all(output_base.parent().context("Slang source has no parent")?)?;
        let wrapper_file = output_base.with_extension("slang");
        let object_file = output_base.with_extension("spv");
        let dephashes_file = output_base.with_extension("dephashes");
        let mut hasher = blake3::Hasher::new();
        hasher.update(caching::build_system_hash()?.as_bytes());
        hasher.update(self._global_session.build_tag_string().as_bytes());
        hasher.update(env::var("OPT_LEVEL")?.as_bytes());
        let buildsystem_hash = *hasher.finalize().as_bytes();

        if let Ok(contents) = fs::read(&dephashes_file)
            && let Ok(cached) = serde_json::from_slice::<Dephashes>(&contents)
            && cached.buildsystem_hash == buildsystem_hash
            && cached
                .dependency_hashes
                .iter()
                .chain(&cached.artifact_hashes)
                .all(|(path, hash)| fs::read(path).is_ok_and(|contents| blake3::hash(&contents).as_bytes() == hash))
        {
            for path in cached.artifact_hashes.keys() {
                println!("cargo::rerun-if-changed={path}");
            }
            debug_log!("Slang compile cached: {}", source_file.display());
            return Ok((kernel_path, cached.public_kernels));
        }

        let source_path = source_file.to_str().context("Slang source path is not UTF-8")?;
        let loaded = slang_api::load_module(&self.session, source_path)?;
        if let Some(diagnostics) = &loaded.diagnostics {
            for line in diagnostics.lines() {
                println!("cargo::warning={line}");
            }
        }
        let dependency_hashes = loaded
            .module
            .dependency_file_paths()
            .map(|path| {
                Ok((path.into(), blake3::hash(&fs::read(path).with_context(|| format!("cannot read {path}"))?).into()))
            })
            .collect::<Result<HashMap<String, [u8; blake3::OUT_LEN]>, Error>>()?;
        let kernels = loaded
            .module
            .module_reflection()
            .children()
            .map(SlangKernelInfo::from_reflection)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let public_kernels = kernels
            .iter()
            .map(SlangKernelInfo::to_kernel)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Box<[Kernel]>>();

        let mut artifact_hashes = HashMap::new();
        if !kernels.is_empty() {
            let blocks = kernels
                .iter()
                .map(|kernel| wrapper::generate_wrappers(kernel, &loaded.component))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten();
            let imports = format!("import definitions;\nimport {source_path:?};");
            let contents = once(imports).chain(blocks).join("\n\n");
            fs::write(&wrapper_file, contents)?;
            let wrapper_path = wrapper_file.to_str().context("Slang wrapper path is not UTF-8")?;
            let loaded = slang_api::load_module(&self.session, wrapper_path)?;
            if let Some(diagnostics) = &loaded.diagnostics {
                for line in diagnostics.lines() {
                    println!("cargo::warning={line}");
                }
            }
            let compiled = loaded.component.link().context("cannot link Slang wrapper module")?;
            let blob = compiled.target_code(0).context("cannot emit SPIR-V")?;
            fs::write(&object_file, blob.as_slice())?;
            for path in [&wrapper_file, &object_file] {
                artifact_hashes.insert(
                    path.to_str().context("Slang artifact path is not UTF-8")?.to_owned(),
                    blake3::hash(&fs::read(path)?).into(),
                );
                println!("cargo::rerun-if-changed={}", path.display());
            }
        } else {
            for path in [&wrapper_file, &object_file] {
                if path.exists() {
                    fs::remove_file(path)?;
                }
            }
        }
        fs::write(
            dephashes_file,
            serde_json::to_vec(&Dephashes {
                buildsystem_hash,
                dependency_hashes,
                artifact_hashes,
                public_kernels: public_kernels.clone(),
            })?,
        )?;
        debug_log!("Slang compile end: {}", source_file.display());
        Ok((kernel_path, public_kernels))
    }
}

impl Compiler for SlangCompiler {
    fn build(
        &self,
        _gpu_types: &GpuTypes,
        _enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        println!("cargo::rerun-if-changed={}", self.src_dir.display());
        println!("cargo::rerun-if-env-changed=SLANG_DIR");
        println!("cargo::rerun-if-env-changed=LD_LIBRARY_PATH");
        let mut sources = WalkDir::new(&self.src_dir)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|entry| {
                entry.file_type().is_file() && entry.path().extension().is_some_and(|extension| extension == "slang")
            })
            .map(|entry| entry.into_path())
            .collect::<Vec<_>>();
        sources.sort();
        let mut kernels = HashMap::new();
        for source in sources {
            if !fs::read(&source)?.starts_with(b"implementing") {
                let (path, file_kernels) =
                    self.compile(&source).with_context(|| format!("cannot compile {}", source.display()))?;
                kernels.insert(path, file_kernels);
            }
        }
        Ok(kernels)
    }
}
