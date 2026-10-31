use std::{
    collections::{BTreeMap, HashSet},
    fs,
    iter::once,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use shader_slang::{
    Blob, CompileTarget, CompilerOptions, ComponentType, DeclKind, GlobalSession, Module, ScalarType, TargetDesc,
};
use syn::{BinOp, Expr};

use super::{SlangEntryPointAbi, SlangEntryPointVariant, bindgen, wrapper};
use crate::{
    common::{codegen::write_tokens, identifiers::KernelName, kernel::Kernel, mangling::static_mangle},
    slang::{BlockingSlangCompiler, SlangArgumentType, SlangCompilerFields, SlangGpuTypes, SlangKernel, SlangTarget},
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

#[derive(Clone)]
pub struct VulkanTarget;

impl SlangTarget for VulkanTarget {
    const TARGET_NAME: &'static str = "vulkan";
    const SOURCE_DIRECTORY: &'static str = "src/backends/vulkan/kernel";
    const PRIVATE_GPU_TYPES: Option<&'static str> = Some("vulkan");
    /// The second session compiles `bf16` as VK_KHR_shader_bfloat16's type.
    const SESSION_DEFINES: &'static [&'static [(&'static str, &'static str)]] = &[&[], &[("NATIVE_BF16", "1")]];

    fn compiler_options(
        global_session: &GlobalSession,
        options: CompilerOptions,
    ) -> CompilerOptions {
        options
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
    }

    fn create_target_desc<'a>(global_session: &'a GlobalSession) -> TargetDesc<'a> {
        TargetDesc::default().format(CompileTarget::Spirv).profile(global_session.find_profile("spirv_1_6"))
    }

    fn gpu_types_directory(fields: &SlangCompilerFields<Self>) -> PathBuf {
        fields.output_directory.join("generated")
    }

    fn wrapper(
        fields: &SlangCompilerFields<Self>,
        source_path: &Path,
        module: &Module,
        kernels: &[SlangKernel],
    ) -> anyhow::Result<String> {
        // A wrapper does not redeclare a specialization the module declares itself.
        let module_globals = module
            .module_reflection()
            .children()
            .filter(|decl| matches!(decl.kind(), DeclKind::Variable))
            .filter_map(|decl| decl.as_variable()?.name())
            .collect::<HashSet<_>>();
        let wrappers = kernels
            .iter()
            .map(|kernel| wrapper::generate_wrappers(kernel, &module_globals))
            .collect::<Result<Vec<_>, _>>()?;
        // The wrapper names the generated types the kernels' signatures use, so it imports what the source does.
        let generated = Self::gpu_types_directory(fields);
        let imports = once("definitions".to_owned())
            .chain(module.dependency_file_paths().filter_map(|path| {
                let path = Path::new(path);
                (path.parent() == Some(&generated))
                    .then(|| Some(format!("generated.{}", path.file_stem()?.to_str()?)))
                    .flatten()
            }))
            .sorted()
            .map(|module| format!("import {module};\n"))
            .chain(once(format!("import {source_path:?};")))
            .collect::<String>();
        Ok(once(imports).chain(wrappers.into_iter().flatten()).join("\n\n"))
    }

    fn emit(
        compiler: &BlockingSlangCompiler<Self>,
        gpu_types: &SlangGpuTypes,
        _module: &Module,
        kernels: &[SlangKernel],
        descriptors: &[Kernel],
        linked: &ComponentType,
        output_base: &Path,
    ) -> anyhow::Result<()> {
        // Specialization constants of the module's kernels with their wire types.
        let mut declared = BTreeMap::new();
        for kernel in kernels {
            for argument in &kernel.arguments {
                if matches!(argument.argument_type, SlangArgumentType::Specialize { .. }) || argument.pipeline_variants
                {
                    let wire = match wrapper::specialization_wire_type(&argument.slang_type) {
                        "bool" => ScalarType::Bool,
                        _ => ScalarType::Uint32,
                    };
                    declared.insert(wrapper::specialization_name(&kernel.name, &argument.name), wire);
                }
            }
        }
        // An entry point that stores BF16 also builds with the native type.
        let native = compiler.link(1, &output_base.with_extension("slang"))?;
        for (kernel, descriptor) in kernels.iter().zip(descriptors) {
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
            let variants = kernel
                .variants()
                .map(|variant| {
                    let types = variant.iter().map(|(_, ty)| ty.to_string()).collect::<Vec<_>>();
                    let name = static_mangle(&kernel.name, &types);
                    let (abi, spirv) = entry_point(linked, &name, &declared)?;
                    let spirv = spirv()?;
                    check_capabilities(&name, &required, spirv.as_slice(), false)?;
                    let object_file = output_base.with_file_name(&name).with_extension("spv");
                    fs::write(&object_file, spirv.as_slice())?;
                    let (native_abi, native_spirv) = entry_point(&native, &name, &declared)?;
                    let stores_bf16 = types.iter().any(|ty| ty == "bf16")
                        || native_abi
                            .fields
                            .iter()
                            .any(|field| field.layout.as_ref().is_some_and(|layout| layout.name == "bf16"));
                    // Kept where it differs.
                    let native_file = object_file.with_extension("bf16.spv");
                    let native = match stores_bf16.then(native_spirv).transpose()? {
                        Some(native) if native.as_slice() != spirv.as_slice() => {
                            check_capabilities(&name, &required, native.as_slice(), true)?;
                            fs::write(&native_file, native.as_slice())?;
                            Some((native_abi, artifact_path(&native_file)?))
                        },
                        _ => {
                            if native_file.exists() {
                                fs::remove_file(&native_file)?;
                            }
                            None
                        },
                    };
                    Ok(SlangEntryPointVariant {
                        types,
                        abi,
                        spirv: artifact_path(&object_file)?,
                        native_bf16: native,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            write_tokens(
                bindgen::bindgen(kernel, descriptor, &variants, &gpu_types.paths, &gpu_types.definitions)?,
                bindgen::binding_file(output_base, &kernel.name),
            )
            .with_context(|| format!("cannot write {} binding", kernel.name))?;
        }
        Ok(())
    }

    fn bindgen_global(
        fields: &SlangCompilerFields<Self>,
        modules: &[(PathBuf, &[Kernel], &[KernelName])],
    ) -> anyhow::Result<TokenStream> {
        // The selectors are under the kernel sources, whose changes already rerun this script.
        for selector in SELECTORS {
            let source = fs::read_to_string(fields.source_directory.join(selector))?;
            let file = syn::parse_file(&source).with_context(|| format!("cannot parse {selector}"))?;
            let identity = |ident: &Ident| IDENTITY_NAMES.iter().any(|name| ident == name);
            if let Some(name) = bindgen::find_ident(file.into_token_stream(), &identity) {
                anyhow::bail!("{selector} names the device identity `{name}`; selectors read capabilities");
            }
        }
        let mut bindings = Vec::new();
        for (output_base, public, private) in modules {
            let names =
                public.iter().map(|kernel| (&kernel.name, true)).chain(private.iter().map(|name| (name, false)));
            for (name, is_public) in names {
                let file = bindgen::binding_file(output_base, name);
                bindings.push((
                    file.to_str().context("binding path is not UTF-8")?.to_owned(),
                    name.to_string(),
                    is_public,
                ));
            }
        }
        Ok(bindgen::bindgen_umbrella(&bindings))
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
