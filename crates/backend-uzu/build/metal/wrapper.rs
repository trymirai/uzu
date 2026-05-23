use std::{collections::HashMap, iter::once};

use anyhow::bail;
use itertools::Itertools;

use super::{
    ast::{MetalArgument, MetalArgumentType, MetalKernelInfo},
    enum_path_rewrite::is_enum_c_type,
};
use crate::common::{enum_paths::EnumPaths, identifiers::KernelName, mangling::static_mangle};

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

    let engine = rhai::Engine::new();
    for type_variant in if let Some(variants) = &kernel.variants {
        variants
            .iter()
            .map(|type_parameter| type_parameter.variants.iter())
            .multi_cartesian_product()
            .map(|values| {
                Some(
                    variants
                        .iter()
                        .map(|tp| tp.name.to_string())
                        .zip(values.iter().map(|v| v.to_string()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    } else {
        vec![None]
    } {
        if let Some(ref tv) = type_variant {
            if !crate::common::constraints::satisfied(&engine, tv, &kernel.constraints) {
                continue;
            }
        }

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
                        format!("{} {} [[buffer({}), function_constant({})]]", &a.c_type, a.name, i, condition_field,),
                        Some(format!("constant bool {} = ({});", condition_field, condition)),
                    ),
                    None => (format!("{} {} [[buffer({})]]", &a.c_type, a.name, i), None),
                }
            })
            .unzip();

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
            MetalArgumentType::Shared(Some(len)) => {
                Some(format!("{} {}[{}]", &a.c_type.replace('*', ""), a.name, len.as_ref(),))
            },
            MetalArgumentType::Shared(None) => Some(format!("{} {}", &a.c_type.replace('&', ""), a.name)),
            _ => None,
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match &a.argument_type {
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
