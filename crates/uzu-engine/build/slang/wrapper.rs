use anyhow::{Context, bail};
use itertools::Itertools;
use shader_slang::ComponentType;

use super::{Error, SlangArgumentType, SlangKernelInfo, slang_api};

pub fn generate_wrappers(
    kernel: &SlangKernelInfo,
    component: &ComponentType,
) -> Result<(Vec<String>, Vec<(String, Vec<&'static str>)>), Error> {
    let mut wrappers = kernel
        .arguments()
        .map(|a| {
            Ok(match a.argument_type()? {
                SlangArgumentType::Specialize(_) => Some(format!(
                    "[[SpecializationConstant]] const {} {};",
                    specialization_wire_type(&a.slang_type()?),
                    specialization_name(kernel.name(), a.name()?)
                )),
                _ => None,
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
        let (wrapper_name, underlying_call, specialized_generic) = if let Some(ref type_args) = specialization_variant {
            let type_args_str = type_args.join(", ");
            let specialized_generic = kernel
                .generic_decl()
                .map(|gd| slang_api::create_specialized_generic(component, gd, type_args))
                .transpose()?;
            (
                mangle_name(kernel.name(), type_args),
                format!("{}<{}>", kernel.name(), type_args_str),
                specialized_generic,
            )
        } else {
            (mangle_name(kernel.name(), &[]), kernel.name().to_string(), None)
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
                SlangArgumentType::Constant(_) => Some(format!("uniform {} {}", slang_type, name)),
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
                    Ok(match arg_type {
                        SlangArgumentType::Ptr(_) | SlangArgumentType::Constant(_) => name.clone(),
                        // Enum specializations travel as `uint` and convert back at the call.
                        SlangArgumentType::Specialize(_) => match specialization_wire_type(slang_type) {
                            wire if wire == slang_type => specialization_name(kernel.name(), name),
                            _ => format!("{slang_type}({})", specialization_name(kernel.name(), name)),
                        },
                        SlangArgumentType::Axis(_, _) => {
                            format!("__dsl_axis_idx.{}", axis_letters.next().context("more than three Axis arguments")?)
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

        let numthreads = calculate_numthreads(&arguments)?;

        let guards = arguments
            .iter()
            .filter_map(|(_, kind, _)| match kind {
                SlangArgumentType::Axis(total, _) => Some(total),
                _ => None,
            })
            .zip(["x", "y", "z"])
            .map(|(total, axis)| format!("if (__dsl_axis_idx.{axis} >= ({total})) return;"))
            .join("\n  ");
        // Every entry point rounds 16- and 32-bit float results to nearest even, the same as the CPU backend; without
        // an explicit mode, Vulkan leaves the rounding implementation-defined.
        let rounding = format!(
            "spirv_asm {{\n    OpCapability RoundingModeRTE;\n    OpExtension \"SPV_KHR_float_controls\";\n    \
             OpExecutionMode ${wrapper_name} RoundingModeRTE 16;\n    \
             OpExecutionMode ${wrapper_name} RoundingModeRTE 32;\n  }};"
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

fn mangle_name(
    kernel_name: &str,
    type_args: &[&str],
) -> String {
    let mut result = format!("__dsl_{}{}", kernel_name.len(), kernel_name);
    for ty in type_args {
        result.push_str(&format!("_{}{}", ty.len(), ty));
    }
    result
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

fn calculate_numthreads(arguments: &[(String, SlangArgumentType, String)]) -> Result<String, Error> {
    let threads: Vec<&str> = arguments
        .iter()
        .filter_map(|(_, arg_type, _)| match arg_type {
            SlangArgumentType::Axis(_, threads_per_group) => Some(threads_per_group.as_ref()),
            SlangArgumentType::Threads(threads) => Some(threads.as_ref()),
            _ => None,
        })
        .collect();

    if threads.is_empty() {
        Ok("1, 1, 1".to_string())
    } else {
        let mut result = threads.to_vec();
        while result.len() < 3 {
            result.push("1");
        }
        Ok(result.join(", "))
    }
}
