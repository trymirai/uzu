use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env,
    ffi::CString,
    fs,
    iter::once,
    path::{Path, PathBuf},
    sync::mpsc::{self, Sender},
    thread,
};

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::Ident;
use quote::ToTokens;
use shader_slang::{
    Blob, CompileTarget, CompilerOptions, ComponentType, DeclKind, GlobalSession, OptimizationLevel, ScalarType,
    Session, SessionDesc, TargetDesc,
};
use syn::{BinOp, Expr};
use walkdir::WalkDir;

use super::{Dephashes, SlangEntryPointAbi, SlangEntryPointVariant, bindgen, wrapper};
use crate::{
    common::{
        caching,
        codegen::write_tokens,
        compiler::Compiler,
        gpu_type_paths::GpuTypePaths,
        gpu_types::GpuTypes,
        identifiers::{KernelName, KernelPath},
        kernel::Kernel,
    },
    debug_log,
    slang::{SessionLoadModuleWithDiagnosticsExt, SlangArgumentType, SlangKernel, gpu_type_gen},
};

/// SPIR-V capabilities that device selection guarantees: Shader, Float16, Int16, Int8, GroupNonUniform and its
/// Arithmetic, Ballot and Shuffle, SignedZeroInfNanPreserve, RoundingModeRTE, VulkanMemoryModel and its device scope,
/// PhysicalStorageBufferAddresses, DotProductInput4x8BitPacked and DotProduct.
const DEVICE_CAPABILITIES: [u32; 15] = [1, 9, 22, 39, 61, 63, 64, 65, 4466, 4467, 5345, 5346, 5347, 6018, 6019];
/// Capabilities only some devices enable, with the device capabilities one of which a `new` precondition must require.
const GATED_CAPABILITIES: [(u32, &str, &[&str]); 4] = [
    (
        6022,
        "CooperativeMatrixKHR",
        &["BF16_TENSOR_CORES", "INT8_TENSOR_CORES_K32", "INT8_TENSOR_CORES_K16", "float_tensor_cores"],
    ),
    (5116, "BFloat16TypeKHR", &["BF16_TENSOR_CORES", "float_tensor_cores"]),
    (5118, "BFloat16CooperativeMatrixKHR", &["BF16_TENSOR_CORES", "float_tensor_cores"]),
    (5432, "CooperativeMatrixPerElementOperationsEXT", &["INT8_ELEMENT_FOLD"]),
];
/// The matmul selectors under the kernel sources, which may not name the device's or driver's identity.
const SELECTORS: [&str; 3] = ["matmul/vk_matmul_plan.rs", "matmul/vk_gemv_tile.rs", "matmul/vk_matmul_kernel.rs"];
/// Identity names a selector must not mention outside comments and strings: plans and their defaults read capabilities
/// and limits, never which device or driver this is.
const IDENTITY_NAMES: [&str; 13] = [
    "vendor_id",
    "device_id",
    "driver_id",
    "device_name",
    "driver_name",
    "driver_info",
    "driver_version",
    "driver_properties",
    "id_properties",
    "device_uuid",
    "driver_uuid",
    "pipeline_cache_uuid",
    "properties",
];

pub struct SlangCompiler {
    session: Session,
    global_session: GlobalSession,
    src_dir: PathBuf,
    out_dir: PathBuf,
}

impl SlangCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src/backends/vulkan/kernel");
        let out_dir = PathBuf::from(env::var("OUT_DIR")?).join("vulkan");
        let global_session = GlobalSession::new().context("cannot create Slang global session")?;
        let session = session(&global_session, &src_dir, &out_dir, false)?;

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
        gpu_type_map: &HashMap<String, String>,
        gpu_type_paths: &GpuTypePaths,
        gpu_types: &[&GpuTypes],
    ) -> anyhow::Result<(KernelPath, Box<[Kernel]>, Box<[KernelName]>)> {
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
        let (module, diagnostics) = self.session.load_module_with_diagnostics(source_file)?;
        for line in diagnostics.iter().flat_map(|diagnostics| diagnostics.lines()) {
            println!("cargo::warning={line}");
        }
        let dependency_hashes = module
            .dependency_file_paths()
            .map(|path| {
                Ok((path.into(), blake3::hash(&fs::read(path).with_context(|| format!("cannot read {path}"))?).into()))
            })
            .collect::<anyhow::Result<HashMap<String, [u8; blake3::OUT_LEN]>>>()?;
        // Kernel signatures name GPU types by their short names, as the CPU compiler records them.
        let short_names = gpu_type_map.keys().map(|name| (name.clone(), name.clone())).collect();
        let kernels = module
            .module_reflection()
            .children()
            .filter_map(|decl| SlangKernel::from_reflection(decl, &short_names).transpose())
            .collect::<anyhow::Result<Vec<_>>>()?;
        let descriptors = kernels.iter().map(|kernel| kernel.to_common(&short_names)).collect::<Result<Vec<_>, _>>()?;
        let (public, private): (Vec<_>, Vec<_>) =
            kernels.iter().zip(descriptors.iter().cloned()).partition(|(kernel, _)| kernel.public);
        let public_kernels = public.into_iter().map(|(_, kernel)| kernel).collect::<Box<[Kernel]>>();
        let private_kernels = private.into_iter().map(|(_, kernel)| kernel.name).collect::<Box<[_]>>();

        // A wrapper does not redeclare a specialization the module declares itself.
        let module_globals = module
            .module_reflection()
            .children()
            .filter(|decl| matches!(decl.kind(), DeclKind::Variable))
            .filter_map(|decl| decl.as_variable()?.name())
            .collect::<HashSet<_>>();
        if !kernels.is_empty() {
            let wrappers = kernels
                .iter()
                .map(|kernel| wrapper::generate_wrappers(kernel, &module_globals))
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
            // Specialization constants of the module's kernels with their wire types.
            let mut declared = BTreeMap::new();
            for kernel in &kernels {
                for argument in &kernel.arguments {
                    if matches!(argument.argument_type, SlangArgumentType::Specialize { .. })
                        || argument.pipeline_variants
                    {
                        let wire = match wrapper::specialization_wire_type(&argument.slang_type) {
                            "bool" => ScalarType::Bool,
                            _ => ScalarType::Uint32,
                        };
                        declared.insert(wrapper::specialization_name(&kernel.name, &argument.name), wire);
                    }
                }
            }
            let (src_dir, out_dir, wrapper_file, declared, wrappers) =
                (self.src_dir.as_path(), self.out_dir.as_path(), wrapper_file.as_path(), &declared, &wrappers);
            thread::scope(|scope| {
                // An entry point that stores BF16 also builds with the native type, alongside on another thread.
                let (sender, natives) = mpsc::channel();
                scope.spawn(move || {
                    let entry_points = wrappers.iter().flat_map(|(_, entry_points)| entry_points);
                    if let Err(error) =
                        send_native_bf16_variants(src_dir, out_dir, wrapper_file, entry_points, declared, &sender)
                    {
                        sender.send(Err(error)).ok();
                    }
                });
                let compiled = link(&self.session, wrapper_file)?;
                for ((kernel, descriptor), (_, entry_points)) in kernels.iter().zip(&descriptors).zip(wrappers) {
                    let required = kernel
                        .preconditions
                        .iter()
                        .filter(|(phase, _)| phase == "new")
                        .map(|(_, condition)| {
                            let condition = syn::parse_str(condition)
                                .with_context(|| format!("{}: malformed Precondition '{condition}'", kernel.name))?;
                            anyhow::Ok(required_capabilities(&condition, false))
                        })
                        .flatten_ok()
                        .collect::<anyhow::Result<HashSet<_>>>()?;
                    let variants = entry_points
                        .iter()
                        .map(|(name, types)| {
                            let (abi, spirv) = entry_point(&compiled, name, declared)?;
                            let spirv = spirv()?;
                            check_capabilities(name, &required, spirv.as_slice(), false)?;
                            let object_file = output_base.with_file_name(name).with_extension("spv");
                            fs::write(&object_file, spirv.as_slice())?;
                            // Kept where it differs.
                            let native_file = object_file.with_extension("bf16.spv");
                            let native = match natives.recv().context("native BF16 build stopped")?? {
                                Some((native_abi, native)) if native != spirv.as_slice() => {
                                    check_capabilities(name, &required, &native, true)?;
                                    fs::write(&native_file, native)?;
                                    Some((native_abi, artifact_path(&native_file)?))
                                },
                                _ => {
                                    if native_file.exists() {
                                        fs::remove_file(&native_file)?;
                                    }
                                    None
                                },
                            };
                            let spirv = artifact_path(&object_file)?;
                            Ok(SlangEntryPointVariant {
                                types: types.clone(),
                                abi,
                                spirv,
                                native_bf16: native,
                            })
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    let binding_file = bindgen::binding_file(&output_base, &descriptor.name);
                    write_tokens(
                        bindgen::bindgen(kernel, descriptor, &variants, gpu_type_paths, gpu_types)?,
                        &binding_file,
                    )
                    .with_context(|| format!("cannot write {} binding", descriptor.name))?;
                }
                Ok::<_, anyhow::Error>(())
            })?;
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
        let gpu_type_map =
            gpu_type_gen(&self.out_dir.join("generated"), &gpu_types).context("cannot generate Slang GPU types")?;
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
                    .compile(&source, &gpu_type_map, &gpu_type_paths, &gpu_types)
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
        // The selectors are under the kernel sources, whose changes already rerun this script.
        for selector in SELECTORS {
            let source = fs::read_to_string(self.src_dir.join(selector))?;
            let file = syn::parse_file(&source).with_context(|| format!("cannot parse {selector}"))?;
            let identity = |ident: &Ident| IDENTITY_NAMES.iter().any(|name| ident == name);
            if let Some(name) = bindgen::find_ident(file.into_token_stream(), &identity) {
                anyhow::bail!("{selector} names the device identity `{name}`; selectors read capabilities");
            }
        }
        Ok(kernels)
    }
}

/// Fails unless each capability `spirv` declares is in DEVICE_CAPABILITIES or gated: a GATED_CAPABILITIES entry one
/// of whose device capabilities is `required`; the native BF16 build, which only devices with the BF16 type load, may
/// also declare BFloat16TypeKHR.
fn check_capabilities(
    entry_point: &str,
    required: &HashSet<String>,
    spirv: &[u8],
    native_bf16: bool,
) -> anyhow::Result<()> {
    const OP_CAPABILITY: u32 = 17;
    const BFLOAT16_TYPE: u32 = 5116;
    let words = spirv.as_chunks::<4>().0.iter().map(|word| u32::from_le_bytes(*word)).collect::<Vec<_>>();
    let mut at = 5;
    while at < words.len() {
        let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xffff);
        anyhow::ensure!(count > 0 && at + count <= words.len(), "{entry_point}: malformed SPIR-V");
        if opcode == OP_CAPABILITY {
            let capability = words[at + 1];
            let gated = GATED_CAPABILITIES.iter().find(|(value, ..)| *value == capability);
            let allowed = DEVICE_CAPABILITIES.contains(&capability)
                || (native_bf16 && capability == BFLOAT16_TYPE)
                || gated.is_some_and(|(_, _, gates)| gates.iter().any(|gate| required.contains(*gate)));
            anyhow::ensure!(
                allowed,
                "{entry_point}: SPIR-V capability {} is neither enabled on every device nor gated by a precondition",
                gated.map_or(capability.to_string(), |(_, name, _)| (*name).to_owned())
            );
        }
        at += count;
    }
    Ok(())
}

/// The device capabilities a `new` precondition `condition` requires: each `float_tensor_cores` call and each
/// `VkDeviceCapabilities` flag a `contains` tests, among its `&&` terms; with `flags`, `condition` is `|`-joined flags.
fn required_capabilities(
    condition: &Expr,
    flags: bool,
) -> Vec<String> {
    match condition {
        Expr::Binary(binary) if matches!((&binary.op, flags), (BinOp::And(_), false) | (BinOp::BitOr(_), true)) => {
            [&binary.left, &binary.right].into_iter().flat_map(|term| required_capabilities(term, flags)).collect()
        },
        Expr::Paren(paren) => required_capabilities(&paren.expr, flags),
        Expr::MethodCall(call) if !flags && call.method == "float_tensor_cores" => vec![call.method.to_string()],
        Expr::MethodCall(call) if !flags && call.method == "contains" => {
            call.args.iter().flat_map(|flag| required_capabilities(flag, true)).collect()
        },
        Expr::Path(path)
            if flags && path.path.segments.len() == 2 && path.path.segments[0].ident == "VkDeviceCapabilities" =>
        {
            vec![path.path.segments[1].ident.to_string()]
        },
        _ => Vec::new(),
    }
}

/// A SPIR-V session over the kernel sources and the generated GPU type modules that compiles `bf16` as
/// VK_KHR_shader_bfloat16's type if `native_bf16`.
fn session(
    global_session: &GlobalSession,
    src_dir: &Path,
    out_dir: &Path,
    native_bf16: bool,
) -> anyhow::Result<Session> {
    let optimization = match env::var("OPT_LEVEL")?.as_str() {
        "0" => OptimizationLevel::Default,
        _ => OptimizationLevel::High,
    };
    let options = CompilerOptions::default()
        .optimization(optimization)
        .emit_spirv_directly(true)
        // The subgroup operations the device filter requires.
        .capability(global_session.find_capability("spvGroupNonUniform"))
        .capability(global_session.find_capability("spvGroupNonUniformArithmetic"))
        .capability(global_session.find_capability("spvGroupNonUniformBallot"))
        .capability(global_session.find_capability("spvGroupNonUniformShuffle"))
        // Declared by the tensor-core and native-BF16 kernels only, whose construction requires the device features.
        .capability(global_session.find_capability("spvCooperativeMatrixKHR"))
        .capability(global_session.find_capability("spvBFloat16KHR"))
        // Slang requires shader model 5.0 of BF16 conversions and cooperative-matrix operations; their SPIR-V needs none
        // of its capabilities.
        .capability(global_session.find_capability("sm_5_0"))
        .vulkan_use_entry_point_name(true);
    let options = match native_bf16 {
        true => options.macro_define("NATIVE_BF16", "1"),
        false => options,
    };
    // Kernel sources and the generated GPU type modules.
    let search_path_strings =
        [CString::new(src_dir.to_string_lossy().as_bytes())?, CString::new(out_dir.to_string_lossy().as_bytes())?];
    let search_paths = search_path_strings.each_ref().map(|path| path.as_ptr());
    let targets =
        [TargetDesc::default().format(CompileTarget::Spirv).profile(global_session.find_profile("spirv_1_6"))];
    let desc = SessionDesc::default().options(&options).search_paths(&search_paths).targets(&targets);
    global_session.create_session(&desc).context("cannot create Slang session")
}

/// Sends, in order, the ABI and SPIR-V with the native BF16 type of each entry point of the wrapper at `path` that
/// stores BF16, built in a Slang session of this thread's own.
fn send_native_bf16_variants<'a>(
    src_dir: &Path,
    out_dir: &Path,
    path: &Path,
    entry_points: impl Iterator<Item = &'a (String, Vec<String>)>,
    declared: &BTreeMap<String, ScalarType>,
    sender: &Sender<anyhow::Result<Option<(SlangEntryPointAbi, Vec<u8>)>>>,
) -> anyhow::Result<()> {
    let global_session = GlobalSession::new().context("cannot create Slang global session")?;
    let session = session(&global_session, src_dir, out_dir, true)?;
    let compiled = link(&session, path)?;
    for (name, types) in entry_points {
        let (abi, spirv) = entry_point(&compiled, name, declared)?;
        let stores_bf16 = types.iter().any(|ty| ty == "bf16")
            || abi.fields.iter().any(|field| field.layout.as_ref().is_some_and(|layout| layout.name == "bf16"));
        let native = stores_bf16.then(spirv).transpose()?.map(|spirv| (abi, spirv.as_slice().to_vec()));
        // The receiver stops only on an error of its own.
        if sender.send(Ok(native)).is_err() {
            break;
        }
    }
    Ok(())
}

/// The wrapper module at `path` linked with its entry points, which must be composed explicitly for the linked program
/// to reflect them.
fn link(
    session: &Session,
    path: &Path,
) -> anyhow::Result<ComponentType> {
    let (wrapper, diagnostics) = session.load_module_with_diagnostics(path)?;
    for line in diagnostics.iter().flat_map(|diagnostics| diagnostics.lines()) {
        println!("cargo::warning={line}");
    }
    let components =
        once(wrapper.clone().into()).chain(wrapper.entry_points().map(ComponentType::from)).collect::<Vec<_>>();
    session
        .create_composite_component_type(&components)
        .context("cannot compose Slang wrapper entry points")?
        .link()
        .context("cannot link Slang wrapper module")
}

/// The ABI of the entry point `name` of a linked program, and its SPIR-V on demand.
fn entry_point<'a>(
    compiled: &'a ComponentType,
    name: &str,
    declared: &BTreeMap<String, ScalarType>,
) -> anyhow::Result<(SlangEntryPointAbi, impl FnOnce() -> anyhow::Result<Blob> + 'a)> {
    let program = compiled.layout(0).context("linked Slang program has no layout")?;
    let entry_point = program.find_entry_point_by_name(name).context("entry point not linked")?;
    let index =
        program.entry_points().position(|entry| entry.name() == Some(name)).context("entry point has no index")?;
    let spirv = move || compiled.entry_point_code(index as i64, 0).context("cannot emit entry point SPIR-V");
    Ok((SlangEntryPointAbi::from_reflection(program, entry_point, declared)?, spirv))
}

fn artifact_path(file: &Path) -> anyhow::Result<String> {
    Ok(file.to_str().context("Slang artifact path is not UTF-8")?.to_owned())
}
