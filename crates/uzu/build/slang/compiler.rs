use std::{
    collections::HashMap, env, ffi::CString, fs, iter::once, path::PathBuf,
};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use shader_slang::{
    CompileTarget, CompilerOptions, Downcast, GlobalSession, OptimizationLevel,
    Session, SessionDesc, TargetDesc,
};
use tokio::{sync::Mutex, task::spawn_blocking};
use walkdir::WalkDir;

use crate::{
    common::{caching, compiler::Compiler},
    debug_log,
    slang::{
        bindgen::bindgen, reflection::SlangKernelInfo, slang_api, wrapper,
    },
};

#[derive(Serialize, Deserialize)]
struct Dephashes {
    buildsystem_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
}

#[derive(Clone)]
struct SlangCompilerFields {
    src_dir: PathBuf,
    out_dir: PathBuf,
    optimization_level: OptimizationLevel,
}

struct BlockingSlangCompiler {
    _global_session: GlobalSession,
    session: Session,
    fields: SlangCompilerFields,
}

unsafe impl Send for BlockingSlangCompiler {}

impl BlockingSlangCompiler {
    fn new(fields: SlangCompilerFields) -> anyhow::Result<Self> {
        let global_session =
            GlobalSession::new().context("cannot create global session")?;

        let session_options = CompilerOptions::default()
            .optimization(fields.optimization_level)
            .emit_spirv_directly(true)
            .vulkan_use_entry_point_name(true);

        let search_path =
            CString::new(fields.src_dir.to_string_lossy().as_bytes())
                .context("cannot convert src_dir to CString")?;
        let search_paths = [search_path.as_ptr()];

        let target_desc = TargetDesc::default()
            .format(CompileTarget::Spirv)
            .profile(global_session.find_profile("glsl_450"));
        let targets = [target_desc];

        let session_desc = SessionDesc::default()
            .options(&session_options)
            .search_paths(&search_paths)
            .targets(&targets);

        let session = global_session
            .create_session(&session_desc)
            .context("cannot create session")?;

        Ok(Self {
            _global_session: global_session,
            session,
            fields,
        })
    }

    fn compile(
        &self,
        source_file: PathBuf,
    ) -> anyhow::Result<()> {
        let source_name = source_file
            .with_extension("")
            .file_name()
            .context("source path has no file name")?
            .to_os_string();
        let source_dir = source_file
            .parent()
            .context("cannot get source file's directory")?;

        let source_relative_dir = source_dir
            .strip_prefix(&self.fields.src_dir)
            .context("source path is not a subpath of source directory")?;

        let out_dir = self.fields.out_dir.join(source_relative_dir);
        fs::create_dir_all(&out_dir).context("cannot create out dir")?;

        let wrapper_file = out_dir.join(&source_name).with_extension("slang");
        let object_file = out_dir.join(&source_name).with_extension("spv");
        let bindgen_file = out_dir.join(&source_name).with_extension("rs");
        let dephashes_file =
            out_dir.join(&source_name).with_extension("dephashes");

        let source_file_pretty =
            source_file.file_name().unwrap().to_str().unwrap();
        debug_log!("compile start: {source_file_pretty}");

        let buildsystem_hash: [u8; blake3::OUT_LEN] =
            caching::build_system_hash()
                .context("cannot get build system hash")?
                .as_bytes()
                .clone();

        if let Ok(dephashes_contents) = fs::read(&dephashes_file)
            && let Ok(cached) =
                serde_json::from_slice::<Dephashes>(&dephashes_contents)
            && cached.buildsystem_hash == buildsystem_hash
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path)
                    .map(|dependency_contents| {
                        blake3::hash(&dependency_contents).as_bytes() == hash
                    })
                    .unwrap_or(false)
            })
        {
            debug_log!("compile cached: {source_file_pretty}");
            return Ok(());
        }

        let source_file_str =
            source_file.to_str().context("cannot convert the path to &str")?;

        let shader_load_result =
            slang_api::load_module(&self.session, source_file_str)
                .context("cannot load shader module")?;

        if let Some(diag) = &shader_load_result.diagnostics {
            for line in diag.lines() {
                println!("cargo::warning={line}");
            }
        }

        let dependency_hashes = shader_load_result
            .module
            .dependency_file_paths()
            .map(|d| {
                Ok((
                    d.into(),
                    blake3::hash(
                        &fs::read(d)
                            .with_context(|| format!("cannot read {d}"))?,
                    )
                    .into(),
                ))
            })
            .collect::<anyhow::Result<HashMap<String, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        let kernels = shader_load_result
            .module
            .module_reflection()
            .children()
            .map(SlangKernelInfo::from_reflection)
            .collect::<anyhow::Result<Vec<_>>>()
            .context("cannot parse kernel info from reflection")?
            .into_iter()
            .flatten()
            .collect::<Vec<SlangKernelInfo>>();

        if !kernels.is_empty() {
            let wrapper_blocks = kernels
                .iter()
                .map(|kernel| {
                    wrapper::generate_wrappers(
                        kernel,
                        &shader_load_result.module,
                    )
                    .with_context(|| {
                        format!(
                            "cannot generate wrappers for kernel '{}'",
                            kernel.name()
                        )
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .flatten();

            let wrapper_contents = once(format!("import {source_file_str:?};"))
                .chain(wrapper_blocks)
                .join("\n\n");

            fs::write(&wrapper_file, wrapper_contents.as_bytes())
                .context("cannot write wrapper file")?;

            let wrapper_file_str = wrapper_file
                .to_str()
                .context("cannot convert the path to &str")?;

            let wrapper_load_result =
                slang_api::load_module(&self.session, wrapper_file_str)
                    .context("cannot load wrapper module")?;

            if let Some(diag) = wrapper_load_result.diagnostics {
                for line in diag.lines() {
                    println!("cargo::warning={line}");
                }
            }

            let compiled_module = wrapper_load_result
                .module
                .downcast()
                .link()
                .context("cannot link wrapper module")?;

            let compiled_blob = compiled_module
                .target_code(0)
                .context("cannot get target code")?;

            fs::write(&object_file, compiled_blob.as_slice())
                .context("cannot write object file")?;

            let bindgen_blocks = kernels
                .iter()
                .map(|kernel| {
                    bindgen(kernel).with_context(|| {
                        format!(
                            "cannot generate bindings for kernel '{}'",
                            kernel.name()
                        )
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let bindgen_contents = bindgen_blocks.join("\n\n");

            fs::write(&bindgen_file, bindgen_contents.as_bytes())
                .context("cannot write bindgen file")?;
        }

        fs::write(
            &dephashes_file,
            &serde_json::to_vec_pretty(&Dephashes {
                buildsystem_hash,
                dependency_hashes,
            })
            .context("cannot serialize dependency hashes")?,
        )
        .context("cannot write dephashes file")?;

        debug_log!("compile end: {source_file_pretty}");

        Ok(())
    }
}

pub struct SlangCompiler {
    fields: SlangCompilerFields,
    blocking_compiler_pool: Mutex<Vec<BlockingSlangCompiler>>,
}

impl SlangCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let src_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR")
                .context("missing CARGO_MANIFEST_DIR")?,
        )
        .join("src/backends/vulkan/kernel");

        let out_dir =
            PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?)
                .join("vulkan");

        let optimization_level = match env::var("OPT_LEVEL")
            .context("missing OPT_LEVEL")?
            .as_str()
        {
            "0" => OptimizationLevel::None,
            "1" => OptimizationLevel::Default,
            _ => OptimizationLevel::High,
        };

        Ok(Self {
            fields: SlangCompilerFields {
                src_dir,
                out_dir,
                optimization_level,
            },
            blocking_compiler_pool: Mutex::new(Vec::new()),
        })
    }

    async fn with_blocking_compiler<T, F>(
        &self,
        f: F,
    ) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut BlockingSlangCompiler) -> anyhow::Result<T>
            + Send
            + 'static,
    {
        let blocking_compiler =
            if let Some(bc) = self.blocking_compiler_pool.lock().await.pop() {
                bc
            } else {
                BlockingSlangCompiler::new(self.fields.clone())
                    .context("cannot create blocking compiler")?
            };

        let (blocking_compiler, result) = spawn_blocking(move || {
            let mut blocking_compiler = blocking_compiler;
            let result = f(&mut blocking_compiler);
            (blocking_compiler, result)
        })
        .await
        .context("spawn_blocking failed")?;

        self.blocking_compiler_pool.lock().await.push(blocking_compiler);

        result
    }
}

#[async_trait]
impl Compiler for SlangCompiler {
    async fn build(&self) -> anyhow::Result<()> {
        let slang_sources: Vec<PathBuf> = WalkDir::new(&self.fields.src_dir)
            .into_iter()
            .filter_map(|res| res.ok())
            .filter(|entry| {
                entry.path().is_file()
                    && entry.path().extension().and_then(|s| s.to_str())
                        == Some("slang")
                    && fs::read(entry.path())
                        .map(|p| !p.starts_with(b"implementing"))
                        .unwrap_or(true)
            })
            .map(|e| e.into_path())
            .collect();

        let num_concurrent_compiles =
            std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4)
                * 2;

        stream::iter(slang_sources)
            .map(|p| async move {
                let pc = p.clone();
                self.with_blocking_compiler(|bc| bc.compile(pc))
                    .await
                    .with_context(|| format!("cannot compile {}", p.display()))
            })
            .buffer_unordered(num_concurrent_compiles)
            .try_collect::<Vec<()>>()
            .await
            .context("cannot compile slang sources")?;

        Ok(())
    }
}
