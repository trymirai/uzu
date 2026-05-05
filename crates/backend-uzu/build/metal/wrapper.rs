use std::{
    collections::{HashMap, HashSet},
    iter::once,
};

use anyhow::bail;
use itertools::Itertools;

use super::{
    ast::{MetalArgumentType, MetalKernelInfo},
    enum_path_rewrite::EnumPathRewriter,
};
use crate::common::{
    gpu_types::{GpuType, GpuTypes},
    mangling::static_mangle,
};

pub type SpecializeBaseIndices = HashMap<Box<str>, usize>;

pub fn wrappers(
    kernels: &[MetalKernelInfo],
    rewriter: &EnumPathRewriter,
    gpu_types: &GpuTypes,
) -> anyhow::Result<(Box<[Box<str>]>, SpecializeBaseIndices)> {
    let mut all_wrappers = Vec::new();
    let mut base_indices = SpecializeBaseIndices::new();
    let mut next_index = 0usize;

    let uint_packed_names = collect_uint_packed_type_names(gpu_types);

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

        let kernel_wrappers = kernel_wrappers(kernel, base_indices.get(&kernel.name), rewriter, &uint_packed_names)?;
        all_wrappers.extend(kernel_wrappers.into_vec());
    }

    Ok((all_wrappers.into_boxed_slice(), base_indices))
}

fn collect_uint_packed_type_names(gpu_types: &GpuTypes) -> HashSet<Box<str>> {
    gpu_types
        .files
        .iter()
        .flat_map(|file| file.types.iter())
        .filter_map(|ty| match ty {
            GpuType::Enum(e) => Some(e.name.clone()),
            GpuType::Struct(s) if s.is_uint_compatible() => Some(s.name.clone()),
            _ => None,
        })
        .collect()
}

fn short_type_name(c_type: &str) -> &str {
    let trimmed = c_type.trim_start_matches("const ").trim();
    trimmed.rsplit("::").next().unwrap_or(trimmed)
}

fn kernel_wrappers(
    kernel: &MetalKernelInfo,
    base_index: Option<&usize>,
    rewriter: &EnumPathRewriter,
    uint_packed_names: &HashSet<Box<str>>,
) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();

    let is_uint_packed = |c_type: &str| uint_packed_names.contains(short_type_name(c_type));

    let specialize_constant_type = |c_type: &str| {
        let trimmed = c_type.trim_start_matches("const ");
        if is_uint_packed(c_type) {
            "uint".to_string()
        } else {
            trimmed.to_string()
        }
    };

    let specialize_argument = |kernel_name: &str, argument_name: &str, c_type: &str| {
        let trimmed = c_type.trim_start_matches("const ");
        let constant_name = format!("__dsl_specialize_{}_{}", kernel_name, argument_name);
        if is_uint_packed(c_type) {
            format!("{trimmed}({constant_name})")
        } else {
            constant_name
        }
    };

    let specialize_constant_names = if let Some(&base) = base_index {
        kernel_wrappers.push(
            kernel
                .arguments
                .iter()
                .filter(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_))))
                .enumerate()
                .map(|(i, a)| {
                    let c_type = specialize_constant_type(&a.c_type);
                    let idx = base + i;
                    format!(
                        "constant {c_type} __dsl_specialize_{}_{} [[function_constant({idx})]];\n",
                        kernel.name, a.name
                    )
                })
                .collect::<String>()
                .into_boxed_str(),
        );

        let specialize_constant_names = kernel
            .arguments
            .iter()
            .filter_map(|a| {
                if matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_))) {
                    Some(a.name.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>();

        kernel_wrappers.push(
            specialize_constant_names
                .iter()
                .map(|n| format!("#define {} __dsl_specialize_{}_{}\n", n, kernel.name, n))
                .collect::<String>()
                .into(),
        );

        specialize_constant_names
    } else {
        Vec::new()
    };

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

        let (mut wrapper_arguments, condition_definitions) = kernel
            .arguments
            .iter()
            .filter(|a| {
                matches!(a.argument_type(), Ok(MetalArgumentType::Buffer(_)) | Ok(MetalArgumentType::Constant(_)))
            })
            .enumerate()
            .map(|(i, a)| match a.argument_type() {
                Ok(MetalArgumentType::Buffer(_)) | Ok(MetalArgumentType::Constant(_)) => {
                    let condition = a.argument_condition().unwrap();

                    if let Some(condition) = condition {
                        let metal_condition = rewriter.rewrite_for_metal(condition);
                        (
                            format!(
                                "{} {} [[buffer({}), function_constant(__dsl_buffer_condition_{}_{})]]",
                                &a.c_type, a.name, i, wrapper_name, a.name
                            ),
                            Some(format!(
                                "constant bool __dsl_buffer_condition_{}_{} = {};",
                                wrapper_name, a.name, metal_condition
                            )),
                        )
                    } else {
                        (format!("{} {} [[buffer({})]]", &a.c_type, a.name, i), None)
                    }
                },
                _ => unreachable!(),
            })
            .collect::<(Vec<_>, Vec<_>)>();

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

        if kernel.has_thread_context() {
            wrapper_arguments.push("uint __dsl_simd_lane_idx [[thread_index_in_simdgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_idx [[simdgroup_index_in_threadgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_size [[threads_per_simdgroup]]".into());
            wrapper_arguments.push("uint __dsl_simd_group_per_threadgroup [[simdgroups_per_threadgroup]]".into());
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
                    MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_) | MetalArgumentType::Shared(_) => {
                        a.name.to_string()
                    },
                    MetalArgumentType::Specialize(_) => {
                        specialize_argument(&kernel.name, &a.name, &a.c_type)
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
                    MetalArgumentType::ThreadContext => "ThreadContext { .simdgroup_index = __dsl_simd_lane_idx, .threadgroup_index = __dsl_simd_group_idx, .simdgroup_size = __dsl_simd_group_size, .simdgroups_per_threadgroup = __dsl_simd_group_per_threadgroup }".into(),
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
        let condition_definitions = condition_definitions.into_iter().flatten().join("\n");
        let undefs = undefs.join("");

        kernel_wrappers.push(
            format!(
                "{defs}\n{condition_definitions}\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n{undefs}"
            )
            .into(),
        );
    }

    kernel_wrappers
        .push(specialize_constant_names.iter().map(|n| format!("#undef {}\n", n)).collect::<String>().into());

    Ok(kernel_wrappers.into())
}
