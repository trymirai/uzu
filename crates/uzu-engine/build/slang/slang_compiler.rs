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
use crate::{common::caching, debug_log};

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

    pub fn build(&self) -> Result<(), Error> {
        println!("cargo::rerun-if-changed={}", self.src_dir.display());
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
        for source in sources {
            if !fs::read(&source)?.starts_with(b"implementing") {
                self.compile(&source).with_context(|| format!("cannot compile {}", source.display()))?;
            }
        }
        Ok(())
    }

    fn compile(
        &self,
        source_file: &Path,
    ) -> Result<(), Error> {
        let source_name = source_file.file_stem().context("Slang source has no file name")?;
        let source_dir = source_file.parent().context("Slang source has no parent")?;
        let out_dir = self.out_dir.join(source_dir.strip_prefix(&self.src_dir)?);
        fs::create_dir_all(&out_dir)?;
        let wrapper_file = out_dir.join(source_name).with_extension("slang");
        let object_file = out_dir.join(source_name).with_extension("spv");
        let dephashes_file = out_dir.join(source_name).with_extension("dephashes");
        let buildsystem_hash = *caching::build_system_hash()?.as_bytes();

        if let Ok(contents) = fs::read(&dephashes_file)
            && let Ok(cached) = serde_json::from_slice::<Dephashes>(&contents)
            && cached.buildsystem_hash == buildsystem_hash
            && cached
                .dependency_hashes
                .iter()
                .all(|(path, hash)| fs::read(path).is_ok_and(|contents| blake3::hash(&contents).as_bytes() == hash))
        {
            debug_log!("Slang compile cached: {}", source_file.display());
            return Ok(());
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

        if !kernels.is_empty() {
            let blocks = kernels
                .iter()
                .map(|kernel| wrapper::generate_wrappers(kernel, &loaded.component))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten();
            let contents = once(format!("import {source_path:?};")).chain(blocks).join("\n\n");
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
        }
        fs::write(
            dephashes_file,
            serde_json::to_vec(&Dephashes {
                buildsystem_hash,
                dependency_hashes,
            })?,
        )?;
        Ok(())
    }
}
