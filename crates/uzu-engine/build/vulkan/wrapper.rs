use std::collections::HashSet;

use anyhow::{Context, bail};
use itertools::Itertools;
use quote::{ToTokens, format_ident};
use shader_slang::ComponentType;
use syn::{Expr, parse_quote};

use super::{Error, SlangArgumentType, SlangKernelInfo, slang_api};
use crate::common::{expr_rewrite::rewrite_paths_with, mangling::static_mangle};

/// Workgroups per row of a folded one-axis grid: Vulkan's minimum maxComputeWorkGroupCount, which wider grids
/// exceed on devices that report only the minimum.
pub const AXIS_FOLD_GROUPS: u32 = 65_535;

pub fn generate_wrappers(
    kernel: &SlangKernelInfo,
    component: &ComponentType,
    module_globals: &HashSet<&str>,
) -> Result<(Vec<String>, Vec<(String, Vec<&'static str>)>), Error> {
    // A `[[PipelineVariants]]` argument travels as a specialization constant instead of a push constant.
    let mut variant_arguments = Vec::new();
    for argument in kernel.arguments() {
        if argument.pipeline_variants()? {
            variant_arguments.push(argument.name()?.to_owned());
        }
    }
    if variant_arguments.len() > 1 {
        bail!("kernel '{}' marks {variant_arguments:?} PipelineVariants; at most one argument may be", kernel.name());
    }
    let variant_argument = variant_arguments.first().map(String::as_str);
    let mut size_parameters = HashSet::new();
    for argument in kernel.arguments() {
        let kind = argument.argument_type()?;
        let size = match &kind {
            SlangArgumentType::Axis(_, size) | SlangArgumentType::Threads(size) => size,
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
        .arguments()
        .map(|a| {
            let specialized =
                matches!(a.argument_type()?, SlangArgumentType::Specialize(_)) || Some(a.name()?) == variant_argument;
            let name = specialization_name(kernel.name(), a.name()?);
            Ok(match specialized && !module_globals.contains(name.as_str()) {
                true => Some(format!(
                    "[[SpecializationConstant]] const {} {name}{};",
                    specialization_wire_type(&a.slang_type()?),
                    if size_parameters.contains(a.name()?) {
                        " = 1"
                    } else {
                        ""
                    }
                )),
                false => None,
            })
        })
        .filter_map(Result::transpose)
        .collect::<Result<Vec<_>, Error>>()?;

    let type_params = kernel.type_parameters().collect::<Vec<_>>();

    let mut entry_points = Vec::new();
    let specialization_variants: Vec<Option<Vec<&'static str>>> = if type_params.is_empty() {
        vec![None]
    } else {
        type_params.iter().map(|p| p.iter().copied()).multi_cartesian_product().map(Some).collect()
    };

    for specialization_variant in specialization_variants {
        let wrapper_name = static_mangle(kernel.name(), specialization_variant.iter().flatten());
        let (underlying_call, specialized_generic) = if let Some(ref type_args) = specialization_variant {
            let type_args_str = type_args.join(", ");
            let specialized_generic = kernel
                .generic_decl()
                .map(|gd| slang_api::create_specialized_generic(component, gd, type_args))
                .transpose()?;
            (format!("{}<{}>", kernel.name(), type_args_str), specialized_generic)
        } else {
            (kernel.name().to_string(), None)
        };

        let arguments: Vec<_> = kernel
            .arguments()
            .map(|a| {
                let arg_type = a.argument_type()?;
                let specialized_type = if let Some(sg) = specialized_generic {
                    a.specialized_slang_type(sg)?
                } else {
                    a.slang_type()?
                };
                Ok((a.name()?.to_string(), arg_type, specialized_type))
            })
            .collect::<Result<_, Error>>()?;

        let has_axis = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Axis(_, _)));
        // A single axis may arrive folded into rows of AXIS_FOLD_GROUPS workgroups.
        let folded_axis = match arguments
            .iter()
            .filter_map(|(_, kind, _)| match kind {
                SlangArgumentType::Axis(_, size) => Some(size),
                _ => None,
            })
            .collect::<Vec<_>>()
            .as_slice()
        {
            [size] => Some(format!("__dsl_axis_idx.x + __dsl_axis_idx.y * ({AXIS_FOLD_GROUPS}u * ({size}))")),
            _ => None,
        };
        let has_groups = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Groups(_)));
        let has_threads = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Threads(_)));

        if has_axis && (has_groups || has_threads) {
            bail!("mixing groups/threads and axis is not supported");
        }
        if has_groups != has_threads || !(has_axis || has_groups) {
            bail!("kernel '{}' needs either Axis or both Groups and Threads", kernel.name());
        }

        let mut wrapper_arguments: Vec<String> = arguments
            .iter()
            .filter_map(|(name, arg_type, slang_type)| match arg_type {
                SlangArgumentType::Ptr(_) => Some(format!("{} {}", slang_type, name)),
                SlangArgumentType::Constant(_) if Some(name.as_str()) != variant_argument => {
                    Some(format!("uniform {} {}", slang_type, name))
                },
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
                    let variant = Some(name.as_str()) == variant_argument;
                    Ok(match arg_type {
                        SlangArgumentType::Ptr(_) => name.clone(),
                        SlangArgumentType::Constant(_) if !variant => name.clone(),
                        // Enum specializations travel as `uint` and convert back at the call.
                        SlangArgumentType::Specialize(_) | SlangArgumentType::Constant(_) => {
                            match specialization_wire_type(slang_type) {
                                wire if wire == slang_type => specialization_name(kernel.name(), name),
                                _ => format!("{slang_type}({})", specialization_name(kernel.name(), name)),
                            }
                        },
                        SlangArgumentType::Axis(_, _) => match &folded_axis {
                            Some(_) => "__dsl_axis".to_owned(),
                            None => format!(
                                "__dsl_axis_idx.{}",
                                axis_letters.next().context("more than three Axis arguments")?
                            ),
                        },
                        SlangArgumentType::Groups(_) => {
                            format!(
                                "__dsl_group_idx.{}",
                                group_letters.next().context("more than three Groups arguments")?
                            )
                        },
                        SlangArgumentType::Threads(_) => {
                            format!(
                                "__dsl_thread_idx.{}",
                                thread_letters.next().context("more than three Threads arguments")?
                            )
                        },
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?
                .join(", ")
        };

        let numthreads = calculate_numthreads(kernel.name(), &arguments)?;

        let guards = arguments
            .iter()
            .filter_map(|(_, kind, _)| match kind {
                SlangArgumentType::Axis(total, _) => Some(total),
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
        entry_points.push((wrapper_name, specialization_variant.unwrap_or_default()));
    }

    Ok((wrappers, entry_points))
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
    arguments: &[(String, SlangArgumentType, String)],
) -> Result<String, Error> {
    let mut threads = arguments
        .iter()
        .filter_map(|(_, arg_type, _)| match arg_type {
            SlangArgumentType::Axis(_, threads_per_group) => Some(threads_per_group.as_ref()),
            SlangArgumentType::Threads(threads) => Some(threads.as_ref()),
            _ => None,
        })
        .map(|text| {
            let mut expression = syn::parse_str::<Expr>(text)?;
            rewrite_paths_with(&mut expression, |path| {
                let name = path.get_ident()?;
                arguments
                    .iter()
                    .any(|(argument, kind, _)| {
                        argument == &name.to_string() && matches!(kind, SlangArgumentType::Specialize(_))
                    })
                    .then(|| {
                        let name = format_ident!("{}", specialization_name(kernel_name, &name.to_string()));
                        parse_quote! { #name }
                    })
            });
            Ok::<_, Error>(expression.into_token_stream().to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    while threads.len() < 3 {
        threads.push("1".to_owned());
    }
    Ok(threads.join(", "))
}
