use std::{collections::HashMap, iter::once};

use anyhow::bail;
use itertools::Itertools;

use super::ast::{MetalArgumentType, MetalKernelInfo};

pub fn wrappers(kernel: &MetalKernelInfo) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();

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
        let (wrapper_name, underlying_name) =
            if let Some(type_variant) = &type_variant {
                (
                    format!(
                        "{}_{}",
                        kernel.name,
                        type_variant.iter().map(|(_k, v)| v).join("_")
                    ),
                    format!(
                        "{}<{}>",
                        kernel.name,
                        type_variant.iter().map(|(_k, v)| v).join(", ")
                    ),
                )
            } else {
                (kernel.name.to_string(), kernel.name.to_string())
            };

        let max_total_threads_per_threadgroup = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Axis(_, l))
                | Ok(MetalArgumentType::Threads(l)) => Some(format!("({l})")),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_total_threads_per_threadgroup =
            if !max_total_threads_per_threadgroup.is_empty() {
                max_total_threads_per_threadgroup.join(" * ")
            } else {
                "1".to_string()
            };

        let wrapper_argument_replace = type_variant
            .as_ref()
            .map(|tv| tv.iter().cloned().collect::<HashMap<_, _>>());

        let mut wrapper_arguments = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Buffer)
                | Ok(MetalArgumentType::Constant(_)) => Some(format!(
                    "{} {}",
                    a.c_type
                        .split_whitespace()
                        .map(|token| {
                            if let Some(wrapper_argument_replace) =
                                &wrapper_argument_replace
                                && let Some(replacement) =
                                    wrapper_argument_replace.get(token)
                            {
                                replacement
                            } else {
                                token
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                    a.name
                )),
                _ => None,
            })
            .collect::<Vec<_>>();

        if kernel.has_axis() {
            if kernel.has_groups() || kernel.has_threads() {
                bail!("mixing groups/threads and axis is not supported");
            }

            wrapper_arguments.push(
                "uint3 __dsl_axis_idx [[thread_position_in_grid]]".into(),
            );
        }

        if kernel.has_groups() {
            wrapper_arguments.push(
                "uint3 __dsl_group_idx [[threadgroup_position_in_grid]]".into(),
            );
        }

        if kernel.has_threads() {
            wrapper_arguments.push(
                "uint3 __dsl_thread_idx [[thread_position_in_threadgroup]]"
                    .into(),
            );
        }

        let wrapper_arguments = wrapper_arguments.join(", ");

        let shared_definitions = kernel.arguments.iter().filter_map(|a| {
            if let Ok(MetalArgumentType::Shared(len)) = a.argument_type() {
                Some(format!(
                    "{} {}[{}]",
                    a.c_type.replace('*', ""),
                    a.name,
                    len.as_ref(),
                ))
            } else {
                None
            }
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match a.argument_type().unwrap() {
                    MetalArgumentType::Buffer
                    | MetalArgumentType::Constant(_)
                    | MetalArgumentType::Shared(_) => a.name.to_string(),
                    MetalArgumentType::Axis(..) => {
                        format!(
                            "__dsl_axis_idx.{}",
                            group_axis_letters.next().unwrap()
                        )
                    },
                    MetalArgumentType::Groups(_) => {
                        format!(
                            "__dsl_group_idx.{}",
                            group_axis_letters.next().unwrap()
                        )
                    },
                    MetalArgumentType::Threads(_) => {
                        format!(
                            "__dsl_thread_idx.{}",
                            thread_axis_letters.next().unwrap()
                        )
                    },
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call =
            format!("{underlying_name}({underlying_arguments})");

        let wrapper_body = shared_definitions
            .chain(once(underlying_call))
            .map(|l| format!("  {l};\n"))
            .collect::<Vec<_>>()
            .join("");

        kernel_wrappers.push(
            format!(
                "\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n"
            )
            .into(),
        );
    }

    Ok(kernel_wrappers.into())
}
