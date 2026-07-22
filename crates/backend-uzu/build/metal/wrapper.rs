use std::{
    collections::{BTreeMap, HashMap},
    iter::once,
};

use anyhow::{Context, bail};
use igata::{
    constraint_expr::AxisType,
    enum_paths::EnumPaths,
    mangling::static_mangle,
    variants::{AcceptedVariants, AxisSpec, KernelSpace, flattening_impl},
};
use itertools::Itertools;
use proc_macro2::TokenStream;

use super::{
    ast::{
        MetalArgument, MetalArgumentType, MetalKernelInfo, MetalTemplateParameter, MetalTemplateParameterType,
        shared_element_type,
    },
    enum_path_rewrite::is_enum_c_type,
};
use crate::common::identifiers::KernelName;

pub type SpecializeBaseIndices = HashMap<KernelName, usize>;

pub fn wrappers(
    kernels: &[MetalKernelInfo],
    enum_paths: &EnumPaths,
) -> anyhow::Result<(Box<[Box<str>]>, SpecializeBaseIndices)> {
    let mut all_wrappers = Vec::new();
    let mut base_indices = SpecializeBaseIndices::new();
    let mut next_index = 0usize;

    for kernel in kernels {
        let specialize_count =
            kernel.arguments.iter().filter(|a| matches!(&a.argument_type, MetalArgumentType::Specialize(_))).count();

        if specialize_count > 0 {
            base_indices.insert(kernel.name.clone(), next_index);
            next_index += specialize_count;
        }

        let kernel_wrappers = kernel_wrappers(kernel, base_indices.get(&kernel.name), enum_paths)?;
        all_wrappers.extend(kernel_wrappers.into_vec());
    }

    Ok((all_wrappers.into_boxed_slice(), base_indices))
}

struct SpecializeBinding<'a> {
    argument: &'a MetalArgument,
    name_suffix: String,
    enum_type: Option<String>,
}

impl SpecializeBinding<'_> {
    fn slot_name(&self) -> String {
        format!("__dsl_specialize_{}", self.name_suffix)
    }

    fn typed_name(&self) -> String {
        format!("__dsl_typed_{}", self.name_suffix)
    }

    fn slot_type(&self) -> &str {
        match self.enum_type {
            Some(_) => "uint32_t",
            None => self.argument.c_type.trim_start_matches("const ").trim(),
        }
    }

    fn alias_target(&self) -> String {
        if self.enum_type.is_some() {
            self.typed_name()
        } else {
            self.slot_name()
        }
    }
}

fn specialize_bindings<'a>(
    kernel: &'a MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> Vec<SpecializeBinding<'a>> {
    kernel
        .arguments
        .iter()
        .filter(|a| matches!(&a.argument_type, MetalArgumentType::Specialize(_)))
        .map(|argument| {
            let name_suffix = format!("{}_{}", kernel.name, argument.name);
            let enum_type = is_enum_c_type(enum_paths, &argument.c_type)
                .then(|| argument.c_type.trim_start_matches("const ").trim().to_string());
            SpecializeBinding {
                argument,
                name_suffix,
                enum_type,
            }
        })
        .collect()
}

fn kernel_header(
    bindings: &[SpecializeBinding<'_>],
    base_index: usize,
) -> String {
    let mut lines = Vec::new();

    for (offset, binding) in bindings.iter().enumerate() {
        lines.push(format!(
            "constant {} {} [[function_constant({})]];",
            binding.slot_type(),
            binding.slot_name(),
            base_index + offset,
        ));
    }

    for binding in bindings.iter() {
        if let Some(enum_type) = &binding.enum_type {
            lines.push(format!(
                "constant {ty} {name} = {ty}({slot});",
                ty = enum_type,
                name = binding.typed_name(),
                slot = binding.slot_name(),
            ));
        }
    }

    for binding in bindings.iter() {
        lines.push(format!("#define {} {}", binding.argument.name, binding.alias_target()));
    }

    lines.push(String::new());
    lines.join("\n")
}

fn kernel_footer(bindings: &[SpecializeBinding<'_>]) -> String {
    bindings.iter().map(|b| format!("#undef {}\n", b.argument.name)).collect()
}

/// The axis type a constraint expression sees for a template parameter.
fn axis_type(parameter: &MetalTemplateParameter) -> anyhow::Result<AxisType> {
    Ok(match &parameter.ty {
        MetalTemplateParameterType::Type => AxisType::DType,
        MetalTemplateParameterType::Value(rust_type) => match rust_type.as_ref() {
            "bool" => AxisType::Bool,
            "u32" => AxisType::Int,
            // A negative template argument would be mangled and emitted as a `u32`
            // anyway; absence is the honest answer until a kernel needs one.
            "i32" => bail!("template parameter `{}` is `i32`: signed template axes are not supported", parameter.name),
            path => match path.rsplit_once("::") {
                Some((_, short_name)) => AxisType::Enum(short_name.into()),
                None => bail!("template parameter `{}` has unsupported type `{path}`", parameter.name),
            },
        },
    })
}

/// The kernel's variant space in igata's terms: the whole adapter between the Metal AST
/// and the enumeration.
pub fn kernel_space(kernel: &MetalKernelInfo) -> anyhow::Result<KernelSpace<'_>> {
    let axes = kernel
        .variants
        .as_deref()
        .map(|parameters| {
            parameters
                .iter()
                .map(|parameter| {
                    Ok(AxisSpec {
                        name: parameter.name.clone(),
                        ty: axis_type(parameter)?,
                        values: parameter.variants.clone(),
                    })
                })
                .collect::<anyhow::Result<Box<[AxisSpec]>>>()
        })
        .transpose()?;

    Ok(KernelSpace {
        name: kernel.name.as_ref(),
        axes,
        constraints: &kernel.constraints,
    })
}

/// Template-argument tuples this kernel is instantiated for; see
/// [`KernelSpace::accepted_variants`].
pub fn accepted_variants(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> anyhow::Result<AcceptedVariants> {
    kernel_space(kernel)?.accepted_variants(enum_paths)
}

/// One `to_template_args()` per variant group, emitted once for the whole library.
///
/// A group's arms only name field enums; which template axis a field flattens to, and
/// what type that axis has, is the shader's half of the story, so the flattening is
/// generated here rather than beside the type. Every kernel that declares the group must
/// agree on it -- they are the same axes.
pub fn group_flattenings<'a>(
    kernels: impl IntoIterator<Item = &'a MetalKernelInfo>,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<TokenStream>> {
    let mut flattenings: BTreeMap<Box<str>, (&KernelName, String, TokenStream)> = BTreeMap::new();

    for kernel in kernels {
        let space = kernel_space(kernel)?;
        for group in space.groups(enum_paths) {
            let tokens = flattening_impl(group, space.axes(), enum_paths)
                .with_context(|| format!("in variant group `{}` of kernel `{}`", group.name, kernel.name))?;
            let rendered = tokens.to_string();
            if let Some((first, previous, _)) = flattenings.get(&group.name) {
                anyhow::ensure!(
                    *previous == rendered,
                    "kernels `{first}` and `{}` disagree about how `{}` flattens onto their axes",
                    kernel.name,
                    group.name,
                );
            } else {
                flattenings.insert(group.name.clone(), (&kernel.name, rendered, tokens));
            }
        }
    }

    Ok(flattenings.into_values().map(|(_, _, tokens)| tokens).collect())
}

fn kernel_wrappers(
    kernel: &MetalKernelInfo,
    base_index: Option<&usize>,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();
    let bindings = specialize_bindings(kernel, enum_paths);

    if let Some(&base) = base_index {
        kernel_wrappers.push(kernel_header(&bindings, base).into_boxed_str());
    }

    for type_variant in accepted_variants(kernel, enum_paths)? {
        let (wrapper_name, underlying_name) = if let Some(type_variant) = &type_variant {
            (
                static_mangle(kernel.name.as_ref(), type_variant.iter().map(|(_k, v)| v.as_str())),
                format!("{}<{}>", kernel.name, type_variant.iter().map(|(_k, v)| v).join(", ")),
            )
        } else {
            (static_mangle(kernel.name.as_ref(), [] as [&str; 0]), kernel.name.to_string())
        };

        let max_total_threads_per_threadgroup = kernel
            .arguments
            .iter()
            .filter_map(|a| match &a.argument_type {
                MetalArgumentType::Axis(_, l) | MetalArgumentType::Threads(l) => Some(format!("({l})")),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_total_threads_per_threadgroup = if !max_total_threads_per_threadgroup.is_empty() {
            max_total_threads_per_threadgroup.join(" * ")
        } else {
            "1".to_string()
        };

        let (mut wrapper_arguments, condition_definitions): (Vec<_>, Vec<_>) = kernel
            .arguments
            .iter()
            .filter(|a| matches!(&a.argument_type, MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_)))
            .enumerate()
            .map(|(i, a)| {
                let condition_field = format!("__dsl_buffer_condition_{}_{}", wrapper_name, a.name);
                match a.condition.as_deref() {
                    Some(condition) => (
                        format!("{} {} [[buffer({}), function_constant({})]]", a.c_type, a.name, i, condition_field,),
                        Some(format!("constant bool {} = ({});", condition_field, condition)),
                    ),
                    None => (format!("{} {} [[buffer({})]]", a.c_type, a.name, i), None),
                }
            })
            .unzip();

        for (threadgroup_index, argument) in optional_shared_arguments(kernel).enumerate() {
            let element_type = shared_element_type(&argument.c_type);
            wrapper_arguments.push(format!("{element_type}* {} [[threadgroup({threadgroup_index})]]", argument.name));
        }

        if kernel.has_axis() {
            if kernel.has_groups() || kernel.has_threads() {
                bail!("mixing groups/threads and axis is not supported");
            }

            wrapper_arguments.push("uint3 __dsl_axis_idx [[thread_position_in_grid]]".into());
        }

        if kernel.has_groups() || kernel.has_thread_context() {
            wrapper_arguments.push("uint3 __dsl_group_idx [[threadgroup_position_in_grid]]".into());
        }

        if kernel.has_threads() {
            wrapper_arguments.push("uint3 __dsl_thread_idx [[thread_position_in_threadgroup]]".into());
        }

        if kernel.has_thread_context() {
            wrapper_arguments.push("uint __dsl_simd_lane_idx [[thread_index_in_simdgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_idx [[simdgroup_index_in_threadgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_size [[threads_per_simdgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_per_threadgroup [[simdgroups_per_threadgroup]]".into());
        }

        let wrapper_arguments = wrapper_arguments.join(", ");

        let shared_definitions = kernel.arguments.iter().filter_map(|a| match &a.argument_type {
            MetalArgumentType::Shared(dimensions) if a.condition.is_none() => {
                let element_type = shared_element_type(&a.c_type);
                let dimensions = dimensions.as_deref().map(|d| format!("[{d}]")).unwrap_or_default();
                Some(format!("{element_type} {}{dimensions}", a.name))
            },
            _ => None,
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match &a.argument_type {
                    MetalArgumentType::Shared(_) if a.condition.is_some() => {
                        format!("({}){}", a.c_type, a.name)
                    },
                    MetalArgumentType::Buffer(_)
                    | MetalArgumentType::Constant(_)
                    | MetalArgumentType::Shared(_)
                    | MetalArgumentType::Specialize(_) => a.name.to_string(),
                    MetalArgumentType::Axis(..) => {
                        format!("__dsl_axis_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Groups(_) => {
                        format!("__dsl_group_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Threads(_) => {
                        format!("__dsl_thread_idx.{}", thread_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::ThreadContext => "ThreadContext { .simd_lane_id = __dsl_simd_lane_idx, .simdgroup_index = __dsl_simd_group_idx, .simdgroup_size = __dsl_simd_group_size, .simdgroups_per_threadgroup = __dsl_simd_group_per_threadgroup, .threadgroup_position = __dsl_group_idx }".into(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call = format!("{underlying_name}({underlying_arguments})");

        let wrapper_body =
            shared_definitions.chain(once(underlying_call)).map(|l| format!("  {l};\n")).collect::<Vec<_>>().join("");

        let (defs, undefs): (Vec<_>, Vec<_>) = type_variant
            .unwrap_or_default()
            .iter()
            .map(|(k, v)| (format!("\n#define {k} {v}"), format!("#undef {k}\n")))
            .unzip();

        let defs = defs.join("");
        let condition_definitions = condition_definitions.into_iter().flatten().join("\n");
        let undefs = undefs.join("");

        kernel_wrappers.push(
            format!(
                "{defs}\n{condition_definitions}\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n{undefs}"
            )
            .into(),
        );
    }

    if base_index.is_some() {
        kernel_wrappers.push(kernel_footer(&bindings).into_boxed_str());
    }

    Ok(kernel_wrappers.into())
}

fn optional_shared_arguments(kernel: &MetalKernelInfo) -> impl Iterator<Item = &MetalArgument> {
    kernel.arguments.iter().filter(|argument| argument.is_optional_shared())
}
