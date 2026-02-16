use std::{collections::HashMap, iter::once};

use anyhow::bail;
use itertools::Itertools;

use super::ast::{MetalArgumentType, MetalKernelInfo};
use crate::common::mangling::static_mangle;

pub type SpecializeBaseIndices = HashMap<Box<str>, usize>;

pub fn wrappers(kernels: &[MetalKernelInfo]) -> anyhow::Result<(Box<[Box<str>]>, SpecializeBaseIndices)> {
    let mut all_wrappers = Vec::new();
    let mut base_indices = SpecializeBaseIndices::new();
    let mut next_index = 0usize;

    for kernel in kernels {
        let specialize_count = kernel
            .arguments
            .iter()
            .filter(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_))))
            .count();

        if specialize_count > 0 {
            base_indices.insert(kernel.name.clone(), next_index);
            next_index += specialize_count;
        }

        let kernel_wrappers = kernel_wrappers(kernel, base_indices.get(&kernel.name))?;
        all_wrappers.extend(kernel_wrappers.into_vec());
    }

    Ok((all_wrappers.into_boxed_slice(), base_indices))
}

fn kernel_wrappers(
    kernel: &MetalKernelInfo,
    base_index: Option<&usize>,
) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();

    let specialize_constants = if let Some(&base) = base_index {
        kernel
            .arguments
            .iter()
            .filter(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_))))
            .enumerate()
            .map(|(i, a)| {
                let c_type = a.c_type.trim_start_matches("const ");
                let idx = base + i;
                format!("constant {c_type} __dsl_specialize_{}_{} [[function_constant({idx})]];\n", kernel.name, a.name)
            })
            .collect::<String>()
    } else {
        String::new()
    };

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
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Axis(_, l)) | Ok(MetalArgumentType::Threads(l)) => Some(format!("({l})")),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_total_threads_per_threadgroup = if !max_total_threads_per_threadgroup.is_empty() {
            max_total_threads_per_threadgroup.join(" * ")
        } else {
            "1".to_string()
        };

        let mut wrapper_arguments = kernel
            .arguments
            .iter()
            .filter(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Buffer) | Ok(MetalArgumentType::Constant(_))))
            .enumerate()
            .map(|(i, a)| match a.argument_type() {
                Ok(MetalArgumentType::Buffer) | Ok(MetalArgumentType::Constant(_)) => {
                    let condition = a.argument_condition().unwrap();

                    if let Some(condition) = condition {
                        format!(
                            "{} {} [[buffer({}), function_constant(__dsl_specialize_{}_{})]]",
                            &a.c_type, a.name, i, kernel.name, condition
                        )
                    } else {
                        format!("{} {} [[buffer({})]]", &a.c_type, a.name, i)
                    }
                },
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();

        if kernel.has_axis() {
            if kernel.has_groups() || kernel.has_threads() {
                bail!("mixing groups/threads and axis is not supported");
            }

            wrapper_arguments.push("uint3 __dsl_axis_idx [[thread_position_in_grid]]".into());
        }

        if kernel.has_groups() {
            wrapper_arguments.push("uint3 __dsl_group_idx [[threadgroup_position_in_grid]]".into());
        }

        if kernel.has_threads() {
            wrapper_arguments.push("uint3 __dsl_thread_idx [[thread_position_in_threadgroup]]".into());
        }

        let wrapper_arguments = wrapper_arguments.join(", ");

        let shared_definitions = kernel.arguments.iter().filter_map(|a| match a.argument_type() {
            Ok(MetalArgumentType::Shared(Some(len))) => {
                Some(format!("{} {}[{}]", &a.c_type.replace('*', ""), a.name, len.as_ref(),))
            },
            Ok(MetalArgumentType::Shared(None)) => Some(format!("{} {}", &a.c_type.replace('&', ""), a.name)),
            _ => None,
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match a.argument_type().unwrap() {
                    MetalArgumentType::Buffer | MetalArgumentType::Constant(_) | MetalArgumentType::Shared(_) => {
                        a.name.to_string()
                    },
                    MetalArgumentType::Specialize(_) => {
                        format!("__dsl_specialize_{}_{}", kernel.name, a.name)
                    },
                    MetalArgumentType::Axis(..) => {
                        format!("__dsl_axis_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Groups(_) => {
                        format!("__dsl_group_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Threads(_) => {
                        format!("__dsl_thread_idx.{}", thread_axis_letters.next().unwrap())
                    },
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call = format!("{underlying_name}({underlying_arguments})");

        let wrapper_body =
            shared_definitions.chain(once(underlying_call)).map(|l| format!("  {l};\n")).collect::<Vec<_>>().join("");

        let (defs, undefs) = type_variant
            .unwrap_or_default()
            .iter()
            .map(|(k, v)| (format!("\n#define {k} {v}"), format!("#undef {k}\n")))
            .collect::<(Vec<_>, Vec<_>)>();

        let defs = defs.join("");
        let undefs = undefs.join("");

        kernel_wrappers.push(
            format!(
                "{defs}\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n{undefs}"
            )
            .into(),
        );
    }

    if !specialize_constants.is_empty() {
        kernel_wrappers.insert(0, specialize_constants.into());
    }

    Ok(kernel_wrappers.into())
}
