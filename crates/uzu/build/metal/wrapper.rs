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
            .filter(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Specialize(_))))
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

    let specialize_constants = if let Some(&base_function_constant_index) = base_index {
        kernel
            .arguments
            .iter()
            .filter(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Specialize(_))))
            .enumerate()
            .map(|(specialization_argument_offset, argument)| {
                use crate::metal::type_parser::TypeParser;

                let specialization_type_name = if let Ok(parsed_type) = TypeParser::parse_type(&argument.c_type) {
                    match &parsed_type.base {
                        crate::metal::type_info::BaseType::Bool => "bool".to_string(),
                        crate::metal::type_info::BaseType::Int => "int32_t".to_string(),
                        crate::metal::type_info::BaseType::UInt => "uint32_t".to_string(),
                        crate::metal::type_info::BaseType::Float => "float".to_string(),
                        crate::metal::type_info::BaseType::Simd => "Simd".to_string(),
                        crate::metal::type_info::BaseType::Named(name) => {
                            if let Some(namespace_segments) = &parsed_type.namespace {
                                format!("{}::{}", namespace_segments.join("::"), name)
                            } else {
                                name.clone()
                            }
                        },
                    }
                } else {
                    argument.c_type.trim_start_matches("const ").to_string()
                };

                let function_constant_index = base_function_constant_index + specialization_argument_offset;
                format!(
                    "constant {specialization_type_name} __dsl_specialize_{}_{} [[function_constant({function_constant_index})]];\n",
                    kernel.name, argument.name
                )
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
            .map(|variant_values| {
                Some(
                    variants
                        .iter()
                        .map(|template_parameter| template_parameter.name.to_string())
                        .zip(variant_values.iter().map(|variant_value| variant_value.to_string()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    } else {
        vec![None]
    } {
        let (wrapper_name, underlying_name) = if let Some(type_variant) = &type_variant {
            (
                static_mangle(
                    kernel.name.as_ref(),
                    type_variant.iter().map(|(_parameter_name, parameter_value)| parameter_value.as_str()),
                ),
                format!(
                    "{}<{}>",
                    kernel.name,
                    type_variant.iter().map(|(_parameter_name, parameter_value)| parameter_value).join(", ")
                ),
            )
        } else {
            (static_mangle(kernel.name.as_ref(), [] as [&str; 0]), kernel.name.to_string())
        };

        let max_total_threads_per_threadgroup = kernel
            .arguments
            .iter()
            .filter_map(|argument| match argument.argument_type() {
                Ok(MetalArgumentType::Axis(_, axis_extent_expression))
                | Ok(MetalArgumentType::Threads(axis_extent_expression)) => Some(format!("({axis_extent_expression})")),
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
            .filter(|argument| {
                matches!(argument.argument_type(), Ok(MetalArgumentType::Buffer) | Ok(MetalArgumentType::Constant(_)))
            })
            .enumerate()
            .map(|(buffer_index, argument)| match argument.argument_type() {
                Ok(MetalArgumentType::Buffer) | Ok(MetalArgumentType::Constant(_)) => {
                    let condition = argument.argument_condition().unwrap();

                    if let Some(condition_parameter_name) = condition {
                        format!(
                            "{} {} [[buffer({}), function_constant(__dsl_specialize_{}_{})]]",
                            &argument.c_type, argument.name, buffer_index, kernel.name, condition_parameter_name
                        )
                    } else {
                        format!("{} {} [[buffer({})]]", &argument.c_type, argument.name, buffer_index)
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

        if kernel.has_simd() {
            wrapper_arguments.push("uint __dsl_simd_lane_idx [[thread_index_in_simdgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_idx [[simdgroup_index_in_threadgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_size [[threads_per_simdgroup]]".into());
        }

        let wrapper_arguments = wrapper_arguments.join(", ");

        let shared_definitions = kernel.arguments.iter().filter_map(|argument| match argument.argument_type() {
            Ok(MetalArgumentType::Shared(Some(length_expression))) => {
                Some(format!("{} {}[{}]", argument.c_type.replace('*', ""), argument.name, length_expression.as_ref()))
            },
            Ok(MetalArgumentType::Shared(None)) => {
                Some(format!("{} {}", argument.c_type.replace('&', ""), argument.name))
            },
            _ => None,
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|argument| match argument.argument_type().unwrap() {
                    MetalArgumentType::Buffer | MetalArgumentType::Constant(_) | MetalArgumentType::Shared(_) => {
                        argument.name.to_string()
                    },
                    MetalArgumentType::Specialize(_) => {
                        format!("__dsl_specialize_{}_{}", kernel.name, argument.name)
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
                    MetalArgumentType::Simd => "Simd { .lane_idx = __dsl_simd_lane_idx, .group_idx = __dsl_simd_group_idx, .group_size = __dsl_simd_group_size }".into(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call = format!("{underlying_name}({underlying_arguments})");

        let wrapper_body = shared_definitions
            .chain(once(underlying_call))
            .map(|line| format!("  {line};\n"))
            .collect::<Vec<_>>()
            .join("");

        let (macro_definitions, macro_undefinitions) = type_variant
            .unwrap_or_default()
            .iter()
            .map(|(parameter_name, parameter_value)| {
                (format!("\n#define {parameter_name} {parameter_value}"), format!("#undef {parameter_name}\n"))
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let macro_definitions = macro_definitions.join("");
        let macro_undefinitions = macro_undefinitions.join("");

        kernel_wrappers.push(
            format!(
                "{macro_definitions}\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n{macro_undefinitions}"
            )
            .into(),
        );
    }

    if !specialize_constants.is_empty() {
        kernel_wrappers.insert(0, specialize_constants.into());
    }

    Ok(kernel_wrappers.into())
}
