use std::iter::once;

use anyhow::{Context, bail};
use itertools::Itertools;

use super::{
    ast::{MetalArgument, MetalArgumentType, MetalConstantType, MetalKernelInfo, shared_element_type},
    enum_path_rewrite::is_enum_c_type,
    variant_combinations::constrained_combinations,
};
use crate::common::{constraints::Constraints, enum_paths::EnumPaths, mangling::static_mangle};

pub struct VariantWrapper {
    pub name: Box<str>,
    pub source: Box<str>,
}

/// How a specialization value travels through the kernel arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecializeKind {
    Bool,
    U32,
    I32,
    F32,
    /// gpu_types enum or option set, passed as its u32 representation
    Gpu,
}

impl SpecializeKind {
    fn kernarg_c_type(self) -> &'static str {
        match self {
            SpecializeKind::Bool | SpecializeKind::U32 | SpecializeKind::Gpu => "uint",
            SpecializeKind::I32 => "int",
            SpecializeKind::F32 => "float",
        }
    }
}

/// One explicit kernel argument; the order of this list is the kernarg layout.
pub enum KernargParameter<'a> {
    /// buffer or constant: an 8-byte address
    Address(&'a MetalArgument),
    /// specialization value: 4 bytes
    Specialize(&'a MetalArgument, SpecializeKind),
    /// total threads of an AXIS dimension: 4 bytes
    AxisTotal(usize),
}

pub fn specialize_kind(
    argument: &MetalArgument,
    enum_paths: &EnumPaths,
) -> anyhow::Result<SpecializeKind> {
    if is_enum_c_type(enum_paths, &argument.c_type) {
        return Ok(SpecializeKind::Gpu);
    }
    let base = argument.c_type.trim_start_matches("const ").trim();
    Ok(match base {
        "bool" => SpecializeKind::Bool,
        "uint" | "uint32_t" | "unsigned int" => SpecializeKind::U32,
        "int" | "int32_t" => SpecializeKind::I32,
        "float" => SpecializeKind::F32,
        other => bail!("unsupported specialization type `{other}` for `{}`", argument.name),
    })
}

pub fn kernarg_parameters<'a>(
    kernel: &'a MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<KernargParameter<'a>>> {
    let mut parameters = Vec::new();
    for argument in kernel.arguments.iter() {
        match &argument.argument_type {
            MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_) => {
                parameters.push(KernargParameter::Address(argument))
            },
            MetalArgumentType::Groups(super::ast::MetalGroupsType::Indirect) => {
                bail!("indirect dispatch is not supported by the AMDGPU backend")
            },
            _ => {},
        }
    }
    for argument in kernel.arguments.iter() {
        if matches!(argument.argument_type, MetalArgumentType::Specialize(_)) {
            parameters.push(KernargParameter::Specialize(argument, specialize_kind(argument, enum_paths)?));
        }
    }
    let axis_count = kernel.arguments.iter().filter(|a| matches!(a.argument_type, MetalArgumentType::Axis(..))).count();
    parameters.extend((0..axis_count).map(KernargParameter::AxisTotal));
    Ok(parameters)
}

fn constant_pointer_name(argument: &MetalArgument) -> String {
    format!("__dsl_constant_{}", argument.name)
}

fn specialize_parameter_name(argument: &MetalArgument) -> String {
    format!("__dsl_specialize_{}", argument.name)
}

const AXIS_LETTERS: [&str; 3] = ["x", "y", "z"];

fn wrapper_parameter(parameter: &KernargParameter<'_>) -> anyhow::Result<String> {
    Ok(match parameter {
        KernargParameter::Address(argument) => match &argument.argument_type {
            MetalArgumentType::Buffer(_) => format!("{} {}", argument.c_type, argument.name),
            // `[const] constant T &` -> `constant T *`. The pointee is left non-const so that both
            // `const constant T &` and `constant T &` parameters bind (constant memory is read-only anyway).
            MetalArgumentType::Constant((_, MetalConstantType::Scalar)) => {
                let pointee = argument.c_type.trim_end().strip_suffix('&').context("constant is not a reference")?;
                format!("{}* {}", pointee.trim().trim_start_matches("const "), constant_pointer_name(argument))
            },
            MetalArgumentType::Constant((_, MetalConstantType::Array(_))) => {
                format!("{} {}", argument.c_type.trim().trim_start_matches("const "), argument.name)
            },
            _ => unreachable!(),
        },
        KernargParameter::Specialize(argument, kind) => {
            format!("{} {}", kind.kernarg_c_type(), specialize_parameter_name(argument))
        },
        KernargParameter::AxisTotal(dimension) => format!("uint __dsl_axis_total_{}", AXIS_LETTERS[*dimension]),
    })
}

fn specialize_local(
    argument: &MetalArgument,
    kind: SpecializeKind,
) -> String {
    let ty = argument.c_type.trim_start_matches("const ").trim();
    let source = specialize_parameter_name(argument);
    let value = match kind {
        SpecializeKind::Bool => format!("({source} != 0u)"),
        SpecializeKind::U32 | SpecializeKind::I32 | SpecializeKind::F32 => source,
        SpecializeKind::Gpu => format!("static_cast<{ty}>({source})"),
    };
    format!("const {ty} {} = {value};", argument.name)
}

fn kernel_body(
    kernel: &MetalKernelInfo,
    kernarg: &[KernargParameter<'_>],
    underlying_name: &str,
) -> anyhow::Result<String> {
    let mut lines: Vec<String> = Vec::new();

    for parameter in kernarg {
        if let KernargParameter::Specialize(argument, kind) = parameter {
            lines.push(specialize_local(argument, *kind));
        }
    }

    lines.push(
        "const uint3 __dsl_group_idx = uint3(__builtin_amdgcn_workgroup_id_x(), __builtin_amdgcn_workgroup_id_y(), __builtin_amdgcn_workgroup_id_z());"
            .into(),
    );
    lines.push(
        "const uint3 __dsl_thread_idx = uint3(__builtin_amdgcn_workitem_id_x(), __builtin_amdgcn_workitem_id_y(), __builtin_amdgcn_workitem_id_z());"
            .into(),
    );
    lines.push(
        "const uint3 __dsl_group_size = uint3(__builtin_amdgcn_workgroup_size_x(), __builtin_amdgcn_workgroup_size_y(), __builtin_amdgcn_workgroup_size_z());"
            .into(),
    );

    if kernel.has_axis() {
        lines.push("const uint3 __dsl_axis_idx = __dsl_group_idx * __dsl_group_size + __dsl_thread_idx;".into());
        let axis_count =
            kernel.arguments.iter().filter(|a| matches!(a.argument_type, MetalArgumentType::Axis(..))).count();
        let out_of_bounds = (0..axis_count)
            .map(|d| format!("__dsl_axis_idx.{l} >= __dsl_axis_total_{l}", l = AXIS_LETTERS[d]))
            .join(" || ");
        // Metal dispatches exactly the requested threads (non-uniform threadgroups); the grid here is
        // rounded up to whole workgroups, so the extra threads leave immediately.
        lines.push(format!("if ({out_of_bounds}) return;"));
    }

    if kernel.has_thread_context() {
        lines.push(
            "const uint __dsl_flat_thread = __dsl_thread_idx.x + __dsl_group_size.x * (__dsl_thread_idx.y + __dsl_group_size.y * __dsl_thread_idx.z);"
                .into(),
        );
        lines.push("ThreadContext __dsl_thread_context;".into());
        lines.push("__dsl_thread_context.simd_lane_id = metal::__simd_lane_id();".into());
        lines.push("__dsl_thread_context.simdgroup_index = __dsl_flat_thread / 32u;".into());
        lines.push("__dsl_thread_context.simdgroup_size = 32u;".into());
        lines.push(
            "__dsl_thread_context.simdgroups_per_threadgroup = (__dsl_group_size.x * __dsl_group_size.y * __dsl_group_size.z + 31u) / 32u;"
                .into(),
        );
        lines.push("__dsl_thread_context.threadgroup_position = __dsl_group_idx;".into());
        lines.push("__dsl_thread_context.threadgroup_size = __dsl_group_size;".into());
    }

    for argument in kernel.arguments.iter() {
        if let MetalArgumentType::Shared(dimensions) = &argument.argument_type {
            let element_type = shared_element_type(&argument.c_type);
            let dimensions = dimensions.as_deref().map(|d| format!("[{d}]")).unwrap_or_default();
            // 16-byte alignment lets staged tiles move through LDS as b128 accesses
            lines.push(format!("{element_type} {}{dimensions} __attribute__((aligned(16)));", argument.name));
        }
    }

    let mut group_axis_letters = AXIS_LETTERS.iter();
    let mut thread_axis_letters = AXIS_LETTERS.iter();
    let call_arguments = kernel
        .arguments
        .iter()
        .map(|argument| {
            Ok(match &argument.argument_type {
                MetalArgumentType::Buffer(_)
                | MetalArgumentType::Constant((_, MetalConstantType::Array(_)))
                | MetalArgumentType::Shared(_)
                | MetalArgumentType::Specialize(_) => argument.name.to_string(),
                MetalArgumentType::Constant((_, MetalConstantType::Scalar)) => {
                    format!("*{}", constant_pointer_name(argument))
                },
                MetalArgumentType::Axis(..) => {
                    format!("__dsl_axis_idx.{}", group_axis_letters.next().context("more than 3 axes")?)
                },
                MetalArgumentType::Groups(_) => {
                    format!("__dsl_group_idx.{}", group_axis_letters.next().context("more than 3 group axes")?)
                },
                MetalArgumentType::Threads(_) => {
                    format!("__dsl_thread_idx.{}", thread_axis_letters.next().context("more than 3 thread axes")?)
                },
                MetalArgumentType::ThreadContext => "__dsl_thread_context".into(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .join(", ");

    let body =
        lines.into_iter().chain(once(format!("{underlying_name}({call_arguments});"))).map(|l| format!("  {l}\n"));
    Ok(body.collect())
}

/// Maximum workgroup size: the product of the THREADS / AXIS threads-per-group expressions.
/// AMDGPU OpenCL kernels default to 256, while uzu dispatches up to 1024.
fn max_workgroup_size(kernel: &MetalKernelInfo) -> String {
    let factors = kernel
        .arguments
        .iter()
        .filter_map(|a| match &a.argument_type {
            MetalArgumentType::Axis(_, per_group) | MetalArgumentType::Threads(per_group) => {
                Some(format!("({per_group})"))
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    if factors.is_empty() {
        "1".into()
    } else {
        factors.join(" * ")
    }
}

/// Variants that are never compiled: the MXU path runs on the M5 neural accelerators through
/// MetalPerformancePrimitives, and AMD composites do not request it.
const EXCLUDED_VARIANTS: &[(&str, &str)] = &[("use_mxu", "true")];

/// Per-kernel allowlists `(kernel, parameter, values)`: only the listed values of the parameter are
/// compiled. GEMM keeps the non-MXU tilings the AMD policy uses, the group sizes of uzu's models and
/// full-precision activations (int8 activations need MXU); its full variant set is ~1500 kernels.
const VARIANT_ALLOWLIST: &[(&str, &str, &[&str])] = &[
    ("Gemm", "GEMM_TILING", &["GemmTiling::Tile32x32x32_Simdgroups2x2", "GemmTiling::Tile64x64x32_Simdgroups2x2"]),
    ("Gemm", "AT", &["bfloat"]),
    ("Gemm", "BT", &["bfloat"]),
    ("Gemm", "DT", &["bfloat"]),
    ("Gemm", "GROUP_SIZE", &["0", "32", "64"]),
    ("Gemm", "A_PROLOGUE", &["GemmAPrologueKind::FullPrecision"]),
];

/// Per-kernel combinations `(kernel, [(parameter, value)])` that are not compiled. Attention GEMM with
/// 32-key blocks at head size 256 is Metal's MXU tiling: it needs 66.5 KB of LDS (the workgroup limit is
/// 64 KB), and the AMD policy takes 16-key blocks from head size 128 on.
const EXCLUDED_COMBINATIONS: &[(&str, &[(&str, &str)])] = &[("AttentionGemm", &[("BK", "32"), ("BD", "256")])];

pub fn kernel_wrappers(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<VariantWrapper>> {
    if kernel.has_axis() && (kernel.has_groups() || kernel.has_threads()) {
        bail!("mixing groups/threads and axis is not supported");
    }

    let kernarg = kernarg_parameters(kernel, enum_paths)?;
    let wrapper_parameters = kernarg.iter().map(wrapper_parameter).collect::<anyhow::Result<Vec<_>>>()?.join(", ");
    let max_workgroup_size = max_workgroup_size(kernel);

    let parameters = kernel.variants.as_deref();
    let selections: Vec<Box<[usize]>> = if let Some(parameters) = parameters {
        let constraints = Constraints::new(
            parameters.iter().flat_map(|tp| tp.variants.iter().map(|v| v.as_ref())),
            &kernel.constraints,
        );
        let domain_lengths = parameters.iter().map(|parameter| parameter.variants.len()).collect::<Vec<_>>();
        constrained_combinations(&domain_lengths, |selection, complete| {
            let bindings = parameters.iter().zip(selection).filter_map(|(parameter, value)| {
                value.map(|value| (parameter.name.as_ref(), parameter.variants[value].as_ref()))
            });
            if complete {
                constraints.satisfied(bindings)
            } else {
                constraints.could_satisfy(bindings)
            }
        })
    } else {
        vec![Box::new([])]
    };

    let mut wrappers = Vec::new();
    for selection in selections {
        let values: Vec<&str> = parameters
            .into_iter()
            .flatten()
            .zip(selection.iter())
            .map(|(parameter, &value)| parameter.variants[value].as_ref())
            .collect();
        let excluded = parameters.into_iter().flatten().zip(values.iter()).any(|(parameter, value)| {
            EXCLUDED_VARIANTS.iter().any(|(name, excluded_value)| {
                parameter.name.eq_ignore_ascii_case(name) && value.trim() == *excluded_value
            })
        });
        let not_allowed = parameters.into_iter().flatten().zip(values.iter()).any(|(parameter, value)| {
            VARIANT_ALLOWLIST.iter().any(|(kernel_name, name, allowed)| {
                kernel.name.as_ref() == *kernel_name
                    && parameter.name.as_ref() == *name
                    && !allowed.contains(&value.trim())
            })
        });
        let excluded_combination =
            EXCLUDED_COMBINATIONS.iter().any(|(kernel_name, combination)| {
                kernel.name.as_ref() == *kernel_name
                    && combination.iter().all(|(name, excluded_value)| {
                        parameters.into_iter().flatten().zip(values.iter()).any(|(parameter, value)| {
                            parameter.name.as_ref() == *name && value.trim() == *excluded_value
                        })
                    })
            });
        if excluded || not_allowed || excluded_combination {
            continue;
        }
        let wrapper_name = static_mangle(kernel.name.as_ref(), values.iter());
        let underlying_name = if parameters.is_some() {
            format!("{}<{}>", kernel.name, values.iter().join(", "))
        } else {
            kernel.name.to_string()
        };

        let (defines, undefines): (Vec<_>, Vec<_>) = parameters
            .into_iter()
            .flatten()
            .zip(values.iter())
            .map(|(parameter, value)| {
                (format!("#define {} {value}\n", parameter.name), format!("#undef {}\n", parameter.name))
            })
            .unzip();

        let body = kernel_body(kernel, &kernarg, &underlying_name)?;
        let source = format!(
            "{defines}__kernel __attribute__((amdgpu_flat_work_group_size(1, {max_workgroup_size}))) void {wrapper_name}({wrapper_parameters}) {{\n{body}}}\n{undefines}\n",
            defines = defines.join(""),
            undefines = undefines.join(""),
        );
        wrappers.push(VariantWrapper {
            name: wrapper_name.into_boxed_str(),
            source: source.into_boxed_str(),
        });
    }
    Ok(wrappers)
}
