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
    CompileTarget, CompilerOptions, ComponentType, GlobalSession, OptimizationLevel, Session, SessionDesc, TargetDesc,
};
use walkdir::WalkDir;

use super::{Dephashes, Error, SlangEntryPointAbi, SlangKernelInfo, bindgen, generate_constants, slang_api, wrapper};
use crate::{
    common::{
        caching,
        codegen::write_tokens,
        compiler::Compiler,
        enum_paths::EnumPaths,
        gpu_types::GpuTypes,
        identifiers::{KernelName, KernelPath},
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
        // Kernel sources and the generated GPU type modules.
        let search_path_strings =
            [CString::new(src_dir.to_string_lossy().as_bytes())?, CString::new(out_dir.to_string_lossy().as_bytes())?];
        let search_paths = search_path_strings.each_ref().map(|path| path.as_ptr());
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
    ) -> Result<(KernelPath, Box<[Kernel]>, Box<[KernelName]>), Error> {
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
            return Ok((kernel_path, cached.public_kernels, cached.test_bindings));
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
        let descriptors = kernels.iter().map(SlangKernelInfo::to_kernel).collect::<Result<Vec<_>, _>>()?;
        let (public, test): (Vec<_>, Vec<_>) =
            kernels.iter().zip(descriptors.iter().cloned()).partition(|(info, _)| info.is_public());
        let public_kernels = public.into_iter().filter_map(|(_, kernel)| kernel).collect::<Box<[Kernel]>>();
        let test_bindings =
            test.into_iter().filter_map(|(_, kernel)| Some(kernel?.name)).collect::<Box<[KernelName]>>();

        let mut artifact_hashes = HashMap::new();
        if !kernels.is_empty() {
            let wrappers = kernels
                .iter()
                .map(|kernel| wrapper::generate_wrappers(kernel, &loaded.component))
                .collect::<Result<Vec<_>, _>>()?;
            let imports = format!("import definitions;\nimport {source_path:?};");
            let contents =
                once(imports).chain(wrappers.iter().flat_map(|(blocks, _)| blocks.iter().cloned())).join("\n\n");
            fs::write(&wrapper_file, contents)?;
            let wrapper_path = wrapper_file.to_str().context("Slang wrapper path is not UTF-8")?;
            let loaded = slang_api::load_module(&self.session, wrapper_path)?;
            if let Some(diagnostics) = &loaded.diagnostics {
                for line in diagnostics.lines() {
                    println!("cargo::warning={line}");
                }
            }
            // Entry points must be composed explicitly for the linked program to reflect them.
            let components = once(loaded.module.clone().into())
                .chain(loaded.module.entry_points().map(ComponentType::from))
                .collect::<Vec<_>>();
            let compiled = self
                .session
                .create_composite_component_type(&components)
                .context("cannot compose Slang wrapper entry points")?
                .link()
                .context("cannot link Slang wrapper module")?;
            let blob = compiled.target_code(0).context("cannot emit SPIR-V")?;
            fs::write(&object_file, blob.as_slice())?;

            let program = compiled.layout(0).context("linked Slang program has no layout")?;
            let object_path = object_file.to_str().context("Slang artifact path is not UTF-8")?;
            let mut binding_files = Vec::new();
            for ((info, descriptor), (_, entry_points)) in kernels.iter().zip(&descriptors).zip(&wrappers) {
                let Some(descriptor) = descriptor else {
                    continue;
                };
                let variants = entry_points
                    .iter()
                    .map(|(name, types)| {
                        let entry_point = program.find_entry_point_by_name(name).context("entry point not linked")?;
                        Ok((types.clone(), SlangEntryPointAbi::from_reflection(program, entry_point)?))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let binding_file = bindgen::binding_file(&output_base, &descriptor.name);
                write_tokens(bindgen::bindgen(info, descriptor, &variants, object_path)?, &binding_file)
                    .with_context(|| format!("cannot write {} binding", descriptor.name))?;
                binding_files.push(binding_file);
            }
            for path in [&wrapper_file, &object_file].into_iter().chain(&binding_files) {
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
                test_bindings: test_bindings.clone(),
            })?,
        )?;
        debug_log!("Slang compile end: {}", source_file.display());
        Ok((kernel_path, public_kernels, test_bindings))
    }
}

impl Compiler for SlangCompiler {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        _enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        // Before any module loads or cache check, so cached dependency hashes see the current constants.
        generate_constants(gpu_types, &self.out_dir).context("cannot generate Slang GPU types")?;
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
        let mut bindings = Vec::new();
        for source in sources {
            if !fs::read(&source)?.starts_with(b"implementing") {
                let (path, file_kernels, test_bindings) =
                    self.compile(&source).with_context(|| format!("cannot compile {}", source.display()))?;
                let names = file_kernels.iter().map(|kernel| (&kernel.name, false));
                for (name, test) in names.chain(test_bindings.iter().map(|name| (name, true))) {
                    let file = bindgen::binding_file(&self.out_dir.join(path.join("/")), name);
                    bindings.push((
                        file.to_str().context("binding path is not UTF-8")?.to_owned(),
                        name.to_string(),
                        test,
                    ));
                }
                kernels.insert(path, file_kernels);
            }
        }
        write_tokens(bindgen::bindgen_umbrella(&bindings), self.out_dir.with_extension("rs"))
            .context("cannot write Vulkan bindings")?;
        Ok(kernels)
    }
}
