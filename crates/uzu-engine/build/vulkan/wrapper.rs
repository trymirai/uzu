use std::collections::HashSet;

use anyhow::{Context, bail};
use itertools::Itertools;
use quote::{ToTokens, format_ident};
use syn::{Expr, parse_quote};

use crate::{
    common::{expr_rewrite::rewrite_paths_with, mangling::static_mangle},
    slang::{SlangArgumentType, SlangKernel, SlangParameterType, Specializer},
};

/// Workgroups per row of a folded one-axis grid: Vulkan's minimum maxComputeWorkGroupCount, which wider grids
/// exceed on devices that report only the minimum.
pub const AXIS_FOLD_GROUPS: u32 = 65_535;

pub fn generate_wrappers(
    kernel: &SlangKernel,
    module_globals: &HashSet<&str>,
) -> anyhow::Result<Vec<String>> {
    if let Some(parameter) = kernel.parameters.iter().find_map(|parameter| match &parameter.ty {
        SlangParameterType::GroupShared {
            value_type,
            length,
        } => Some(format!("{}: {value_type}[{length}]", parameter.name)),
        _ => None,
    }) {
        bail!("kernel '{}': GroupShared {parameter} has no Vulkan binding", kernel.name);
    }
    // A `[[PipelineVariants]]` argument travels as a specialization constant instead of a push constant.
    let variant_arguments =
        kernel.arguments.iter().filter(|argument| argument.pipeline_variants).map(|argument| argument.name.as_str());
    let variant_arguments = variant_arguments.collect::<Vec<_>>();
    if variant_arguments.len() > 1 {
        bail!("kernel '{}' marks {variant_arguments:?} PipelineVariants; at most one argument may be", kernel.name);
    }
    let variant_argument = variant_arguments.first().copied();
    let mut size_parameters = HashSet::new();
    for argument in &kernel.arguments {
        let size = match &argument.argument_type {
            SlangArgumentType::Axis {
                threads_in_group: size,
                ..
            }
            | SlangArgumentType::Threads {
                threads: size,
            } => size,
            _ => continue,
        };
        let mut expression = syn::parse_str::<Expr>(size)?;
        rewrite_paths_with(&mut expression, |path| {
            if let Some(name) = path.get_ident() {
                size_parameters.insert(name.to_string());
            }
            None
        });
    }
    let mut wrappers = kernel
        .arguments
        .iter()
        .filter(|a| matches!(a.argument_type, SlangArgumentType::Specialize { .. }) || a.pipeline_variants)
        .map(|a| (a, specialization_name(&kernel.name, &a.name)))
        .filter(|(_, name)| !module_globals.contains(name.as_str()))
        .map(|(a, name)| {
            format!(
                "[[SpecializationConstant]] const {} {name}{};",
                specialization_wire_type(&a.slang_type),
                if size_parameters.contains(&a.name) {
                    " = 1"
                } else {
                    ""
                }
            )
        })
        .collect::<Vec<_>>();

    for variant in kernel.variants() {
        let types = variant.iter().map(|(_, ty)| ty.to_string()).collect::<Vec<_>>();
        let wrapper_name = static_mangle(&kernel.name, &types);
        let underlying_call = match types.is_empty() {
            true => kernel.name.clone(),
            false => format!("{}<{}>", kernel.name, types.join(", ")),
        };
        let specializer = Specializer::new(&variant);
        let arguments = kernel
            .arguments
            .iter()
            .map(|a| (a.name.as_str(), &a.argument_type, specializer.specialize(&a.slang_type)))
            .collect::<Vec<_>>();

        let has_axis = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Axis { .. }));
        // A single axis may arrive folded into rows of AXIS_FOLD_GROUPS workgroups.
        let folded_axis = match arguments
            .iter()
            .filter_map(|(_, kind, _)| match kind {
                SlangArgumentType::Axis {
                    threads_in_group: size,
                    ..
                } => Some(size),
                _ => None,
            })
            .collect::<Vec<_>>()
            .as_slice()
        {
            [size] => Some(format!("__dsl_axis_idx.x + __dsl_axis_idx.y * ({AXIS_FOLD_GROUPS}u * ({size}))")),
            _ => None,
        };
        let has_groups = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Groups { .. }));
        let has_threads = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Threads { .. }));

        if has_axis && (has_groups || has_threads) {
            bail!("mixing groups/threads and axis is not supported");
        }
        if has_groups != has_threads || !(has_axis || has_groups) {
            bail!("kernel '{}' needs either Axis or both Groups and Threads", kernel.name);
        }

        let mut wrapper_arguments: Vec<String> = arguments
            .iter()
            .filter_map(|(name, arg_type, slang_type)| match arg_type {
                SlangArgumentType::Buffer {
                    ..
                } => Some(format!("{} {}", slang_type, name)),
                SlangArgumentType::Uniform {
                    ..
                } if Some(*name) != variant_argument => Some(format!("uniform {} {}", slang_type, name)),
                _ => None,
            })
            .collect();

        if has_axis {
            wrapper_arguments.push("uint3 __dsl_axis_idx : SV_DispatchThreadID".into());
        }
        if has_groups {
            wrapper_arguments.push("uint3 __dsl_group_idx : SV_GroupID".into());
        }
        if has_threads {
            wrapper_arguments.push("uint3 __dsl_thread_idx : SV_GroupThreadID".into());
        }

        let wrapper_arguments_str = wrapper_arguments.join(", ");

        let underlying_arguments = {
            let mut axis_letters = ["x", "y", "z"].iter();
            let mut group_letters = ["x", "y", "z"].iter();
            let mut thread_letters = ["x", "y", "z"].iter();

            arguments
                .iter()
                .map(|(name, arg_type, slang_type)| {
                    let variant = Some(*name) == variant_argument;
                    Ok(match arg_type {
                        SlangArgumentType::Buffer {
                            ..
                        } => name.to_string(),
                        SlangArgumentType::Uniform {
                            ..
                        } if !variant => name.to_string(),
                        // Enum specializations travel as `uint` and convert back at the call.
                        SlangArgumentType::Specialize {
                            ..
                        }
                        | SlangArgumentType::Uniform {
                            ..
                        } => match specialization_wire_type(slang_type) {
                            wire if wire == &**slang_type => specialization_name(&kernel.name, name),
                            _ => format!("{slang_type}({})", specialization_name(&kernel.name, name)),
                        },
                        SlangArgumentType::Axis {
                            ..
                        } => match &folded_axis {
                            Some(_) => "__dsl_axis".to_owned(),
                            None => format!(
                                "__dsl_axis_idx.{}",
                                axis_letters.next().context("more than three Axis arguments")?
                            ),
                        },
                        SlangArgumentType::Groups {
                            ..
                        } => {
                            format!(
                                "__dsl_group_idx.{}",
                                group_letters.next().context("more than three Groups arguments")?
                            )
                        },
                        SlangArgumentType::Threads {
                            ..
                        } => {
                            format!(
                                "__dsl_thread_idx.{}",
                                thread_letters.next().context("more than three Threads arguments")?
                            )
                        },
                        SlangArgumentType::ThreadContext => {
                            bail!("kernel '{}': ThreadContext '{name}' has no Vulkan binding", kernel.name)
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .join(", ")
        };

        let numthreads = calculate_numthreads(&kernel.name, &arguments)?;

        let guards = arguments
            .iter()
            .filter_map(|(_, kind, _)| match kind {
                SlangArgumentType::Axis {
                    threads: total,
                    ..
                } => Some(total),
                _ => None,
            })
            .zip(["x", "y", "z"])
            .map(|(total, axis)| match &folded_axis {
                Some(index) => format!("let __dsl_axis = {index};\n  if (__dsl_axis >= ({total})) return;"),
                None => format!("if (__dsl_axis_idx.{axis} >= ({total})) return;"),
            })
            .join("\n  ");
        // Round to nearest even and preserve signed zeros, infinities and NaNs like the CPU; Vulkan leaves both open.
        let rounding = format!(
            "spirv_asm {{\n    OpCapability RoundingModeRTE;\n    OpCapability SignedZeroInfNanPreserve;\n    \
             OpExtension \"SPV_KHR_float_controls\";\n    \
             OpExecutionMode ${wrapper_name} RoundingModeRTE 16;\n    \
             OpExecutionMode ${wrapper_name} RoundingModeRTE 32;\n    \
             OpExecutionMode ${wrapper_name} SignedZeroInfNanPreserve 16;\n    \
             OpExecutionMode ${wrapper_name} SignedZeroInfNanPreserve 32;\n  }};"
        );
        let body = format!("{rounding}\n  {guards}\n  {underlying_call}({underlying_arguments});");

        let wrapper = format!(
            "[shader(\"compute\")]\n[numthreads({})]\nvoid {wrapper_name}({wrapper_arguments_str}) {{\n  {body}\n}}",
            numthreads
        );

        wrappers.push(wrapper);
    }

    Ok(wrappers)
}

/// Slang type of a specialization constant: Vulkan specializes only scalars, so enums are `uint`.
pub fn specialization_wire_type(slang_type: &str) -> &str {
    match slang_type {
        "bool" => "bool",
        _ => "uint",
    }
}

pub fn specialization_name(
    kernel_name: &str,
    argument_name: &str,
) -> String {
    format!("__dsl_{kernel_name}_{argument_name}")
}

fn calculate_numthreads(
    kernel_name: &str,
    arguments: &[(&str, &SlangArgumentType, impl AsRef<str>)],
) -> anyhow::Result<String> {
    let mut threads = arguments
        .iter()
        .filter_map(|(_, arg_type, _)| match arg_type {
            SlangArgumentType::Axis {
                threads_in_group: threads,
                ..
            }
            | SlangArgumentType::Threads {
                threads,
            } => Some(threads.as_str()),
            _ => None,
        })
        .map(|text| {
            let mut expression = syn::parse_str::<Expr>(text)?;
            rewrite_paths_with(&mut expression, |path| {
                let name = path.get_ident()?;
                arguments
                    .iter()
                    .any(|(argument, kind, _)| name == argument && matches!(kind, SlangArgumentType::Specialize { .. }))
                    .then(|| {
                        let name = format_ident!("{}", specialization_name(kernel_name, &name.to_string()));
                        parse_quote! { #name }
                    })
            });
            Ok::<_, anyhow::Error>(expression.into_token_stream().to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    while threads.len() < 3 {
        threads.push("1".to_owned());
    }
    Ok(threads.join(", "))
}
