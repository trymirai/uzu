use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env,
    ffi::CString,
    fs,
    iter::once,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;
use shader_slang::{
    CompileTarget, CompilerOptions, ComponentType, DeclKind, GlobalSession, OptimizationLevel, ScalarType, Session,
    SessionDesc, TargetDesc,
};
use walkdir::WalkDir;

use super::{
    Dephashes, Error, SlangArgumentType, SlangEntryPointAbi, SlangKernelInfo, bindgen, generate_types, slang_api,
    wrapper,
};
use crate::{
    common::{
        caching,
        codegen::write_tokens,
        compiler::Compiler,
        gpu_type_paths::GpuTypePaths,
        gpu_types::GpuTypes,
        identifiers::{KernelName, KernelPath},
        kernel::{Kernel, KernelArgumentType, KernelParameterType},
    },
    debug_log,
};

pub struct SlangCompiler {
    session: Session,
    global_session: GlobalSession,
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
            .capability(global_session.find_capability("spvGroupNonUniform"))
            .capability(global_session.find_capability("spvGroupNonUniformArithmetic"))
            .capability(global_session.find_capability("spvGroupNonUniformVote"))
            .capability(global_session.find_capability("spvGroupNonUniformBallot"))
            .capability(global_session.find_capability("spvGroupNonUniformShuffle"))
            // Declared by the tensor-core kernels only, whose construction requires the device features.
            .capability(global_session.find_capability("spvCooperativeMatrixKHR"))
            .capability(global_session.find_capability("spvBFloat16KHR"))
            .vulkan_use_entry_point_name(true);
        // Kernel sources and the generated GPU type modules.
        let search_path_strings =
            [CString::new(src_dir.to_string_lossy().as_bytes())?, CString::new(out_dir.to_string_lossy().as_bytes())?];
        let search_paths = search_path_strings.each_ref().map(|path| path.as_ptr());
        let targets =
            [TargetDesc::default().format(CompileTarget::Spirv).profile(global_session.find_profile("spirv_1_6"))];
        let desc = SessionDesc::default().options(&options).search_paths(&search_paths).targets(&targets);
        let session = global_session.create_session(&desc).context("cannot create Slang session")?;

        Ok(Self {
            session,
            global_session,
            src_dir,
            out_dir,
        })
    }

    fn compile(
        &self,
        source_file: &Path,
        gpu_type_paths: &GpuTypePaths,
        gpu_types: &[&GpuTypes],
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
        let dephashes_file = output_base.with_extension("dephashes");
        let mut hasher = blake3::Hasher::new();
        hasher.update(caching::build_system_hash()?.as_bytes());
        hasher.update(self.global_session.build_tag_string().as_bytes());
        hasher.update(env::var("OPT_LEVEL")?.as_bytes());
        let buildsystem_hash = *hasher.finalize().as_bytes();

        if let Ok(contents) = fs::read(&dephashes_file)
            && let Ok(cached) = serde_json::from_slice::<Dephashes>(&contents)
            && cached.buildsystem_hash == buildsystem_hash
            && cached
                .dependency_hashes
                .iter()
                .all(|(path, hash)| fs::read(path).is_ok_and(|contents| blake3::hash(&contents).as_bytes() == hash))
        {
            debug_log!("Slang compile cached: {}", source_file.display());
            return Ok((kernel_path, cached.public_kernels, cached.private_kernels));
        }

        let source_path = source_file.to_str().context("Slang source path is not UTF-8")?;
        let module = slang_api::load_module(&self.session, source_path)?;
        let dependency_hashes = module
            .dependency_file_paths()
            .map(|path| {
                Ok((path.into(), blake3::hash(&fs::read(path).with_context(|| format!("cannot read {path}"))?).into()))
            })
            .collect::<Result<HashMap<String, [u8; blake3::OUT_LEN]>, Error>>()?;
        let kernels = module
            .module_reflection()
            .children()
            .map(|decl| SlangKernelInfo::from_reflection(decl, gpu_type_paths))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let descriptors = kernels.iter().map(SlangKernelInfo::to_kernel).collect::<Result<Vec<_>, _>>()?;
        let (public, private): (Vec<_>, Vec<_>) =
            kernels.iter().zip(descriptors.iter().cloned()).partition(|(info, _)| info.is_public());
        let public_kernels =
            public.into_iter().map(|(_, kernel)| shared_kernel(kernel, gpu_type_paths)).collect::<Box<[Kernel]>>();
        let private_kernels = private.into_iter().map(|(_, kernel)| kernel.name).collect::<Box<[_]>>();

        // A wrapper does not redeclare a specialization the module declares itself.
        let module_globals = module
            .module_reflection()
            .children()
            .filter(|decl| matches!(decl.kind(), DeclKind::Variable))
            .filter_map(|decl| decl.as_variable()?.name())
            .collect::<HashSet<_>>();
        if !kernels.is_empty() {
            let component = ComponentType::from(module.clone());
            let wrappers = kernels
                .iter()
                .map(|kernel| wrapper::generate_wrappers(kernel, &component, &module_globals))
                .collect::<Result<Vec<_>, _>>()?;
            // The wrapper names the generated types the kernels' signatures use, so it imports what the source does.
            let generated = self.out_dir.join("generated");
            let imports = once("definitions".to_owned())
                .chain(dependency_hashes.keys().filter_map(|path| {
                    let path = Path::new(path);
                    (path.parent() == Some(&generated))
                        .then(|| Some(format!("generated.{}", path.file_stem()?.to_str()?)))
                        .flatten()
                }))
                .sorted()
                .map(|module| format!("import {module};\n"))
                .chain(once(format!("import {source_path:?};")))
                .collect::<String>();
            let contents =
                once(imports).chain(wrappers.iter().flat_map(|(blocks, _)| blocks.iter().cloned())).join("\n\n");
            fs::write(&wrapper_file, contents)?;
            let wrapper_path = wrapper_file.to_str().context("Slang wrapper path is not UTF-8")?;
            let wrapper = slang_api::load_module(&self.session, wrapper_path)?;
            // Entry points must be composed explicitly for the linked program to reflect them.
            let components =
                once(wrapper.clone().into()).chain(wrapper.entry_points().map(ComponentType::from)).collect::<Vec<_>>();
            let compiled = self
                .session
                .create_composite_component_type(&components)
                .context("cannot compose Slang wrapper entry points")?
                .link()
                .context("cannot link Slang wrapper module")?;

            let program = compiled.layout(0).context("linked Slang program has no layout")?;
            // Specialization constants of the module's kernels with their wire types.
            let mut declared = BTreeMap::new();
            for info in &kernels {
                for argument in info.arguments() {
                    if matches!(argument.argument_type()?, SlangArgumentType::Specialize(_))
                        || argument.pipeline_variants()?
                    {
                        let wire = match wrapper::specialization_wire_type(&argument.slang_type()?) {
                            "bool" => ScalarType::Bool,
                            _ => ScalarType::Uint32,
                        };
                        declared.insert(wrapper::specialization_name(info.name(), argument.name()?), wire);
                    }
                }
            }
            for ((info, descriptor), (_, entry_points)) in kernels.iter().zip(&descriptors).zip(&wrappers) {
                let variants = entry_points
                    .iter()
                    .map(|(name, types)| {
                        let entry_point = program.find_entry_point_by_name(name).context("entry point not linked")?;
                        let index = program
                            .entry_points()
                            .position(|entry| entry.name() == Some(name.as_str()))
                            .context("entry point has no index")?;
                        let object_file = output_base.with_file_name(name).with_extension("spv");
                        let blob =
                            compiled.entry_point_code(index as i64, 0).context("cannot emit entry point SPIR-V")?;
                        fs::write(&object_file, blob.as_slice())?;
                        let object_path = object_file.to_str().context("Slang artifact path is not UTF-8")?.to_owned();
                        Ok((
                            types.clone(),
                            SlangEntryPointAbi::from_reflection(program, entry_point, &declared)?,
                            object_path,
                        ))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let binding_file = bindgen::binding_file(&output_base, &descriptor.name);
                write_tokens(bindgen::bindgen(info, descriptor, &variants, gpu_type_paths, gpu_types)?, &binding_file)
                    .with_context(|| format!("cannot write {} binding", descriptor.name))?;
            }
        } else if wrapper_file.exists() {
            fs::remove_file(&wrapper_file)?;
        }
        fs::write(
            dephashes_file,
            serde_json::to_vec(&Dephashes {
                buildsystem_hash,
                dependency_hashes,
                public_kernels: public_kernels.clone(),
                private_kernels: private_kernels.clone(),
            })?,
        )?;
        debug_log!("Slang compile end: {}", source_file.display());
        Ok((kernel_path, public_kernels, private_kernels))
    }
}

impl Compiler for SlangCompiler {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        let private_gpu_types = GpuTypes::scan("vulkan").context("cannot scan Vulkan-private gpu types")?;
        let mut gpu_type_paths = gpu_type_paths.clone();
        gpu_type_paths.extend(&private_gpu_types).context("cannot extend Vulkan GPU type paths")?;
        let gpu_types = [gpu_types, &private_gpu_types];
        // Before any module loads or cache check, so cached dependency hashes see the current constants.
        generate_types(&gpu_types, &self.out_dir).context("cannot generate Slang GPU types")?;
        println!("cargo::rerun-if-changed={}", self.src_dir.display());
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
                let (path, public, private) = self
                    .compile(&source, &gpu_type_paths, &gpu_types)
                    .with_context(|| format!("cannot compile {}", source.display()))?;
                let names =
                    public.iter().map(|kernel| (&kernel.name, true)).chain(private.iter().map(|name| (name, false)));
                for (name, is_public) in names {
                    let file = bindgen::binding_file(&self.out_dir.join(path.join("/")), name);
                    bindings.push((
                        file.to_str().context("binding path is not UTF-8")?.to_owned(),
                        name.to_string(),
                        is_public,
                    ));
                }
                kernels.insert(path, public);
            }
        }
        write_tokens(bindgen::bindgen_umbrella(&bindings), self.out_dir.with_extension("rs"))
            .context("cannot write Vulkan bindings")?;
        Ok(kernels)
    }
}

/// A public kernel with GPU types by their short names, as the CPU compiler records them.
fn shared_kernel(
    mut kernel: Kernel,
    gpu_type_paths: &GpuTypePaths,
) -> Kernel {
    let full_paths =
        gpu_type_paths.full_paths().sorted_unstable_by_key(|path| std::cmp::Reverse(path.len())).collect_vec();
    let shorten = |text: &mut Box<str>| {
        let mut shortened = text.to_string();
        for path in &full_paths {
            shortened = shortened.replace(path, path.rsplit_once("::").map_or(*path, |(_, name)| name));
        }
        *text = shortened.into();
    };
    for parameter in kernel.parameters.iter_mut() {
        if let KernelParameterType::Value(ty) = &mut parameter.ty {
            shorten(ty);
        }
    }
    for argument in kernel.arguments.iter_mut() {
        if let KernelArgumentType::Constant(ty) = &mut argument.ty {
            shorten(ty);
        }
    }
    kernel
}
