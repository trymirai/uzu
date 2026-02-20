use anyhow::bail;
use itertools::Itertools;
use shader_slang::Module;

use super::reflection::{SlangArgumentType, SlangKernelInfo};
use super::slang_api;

pub fn generate_wrappers(
    kernel: &SlangKernelInfo,
    module: &Module,
) -> anyhow::Result<Vec<String>> {
    let mut wrappers = Vec::new();

    let type_params: Vec<_> = kernel.type_parameters().collect();

    let specialization_variants: Vec<Option<Vec<&str>>> = if type_params.is_empty() {
        vec![None]
    } else {
        type_params.iter().map(|p| p.variants.iter().copied()).multi_cartesian_product().map(Some).collect()
    };

    for specialization_variant in specialization_variants {
        let (wrapper_name, underlying_call, specialized_generic) = if let Some(ref type_args) = specialization_variant {
            let type_args_str = type_args.join(", ");
            let specialized_generic = kernel
                .generic_decl()
                .map(|gd| slang_api::create_specialized_generic(module, gd, type_args))
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
                    a.specialized_slang_type(sg)
                } else {
                    a.slang_type()
                };
                Ok((a.name().to_string(), arg_type, specialized_type))
            })
            .collect::<anyhow::Result<_>>()?;

        let has_axis = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Axis(_, _)));
        let has_groups = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Groups(_)));
        let has_threads = arguments.iter().any(|(_, t, _)| matches!(t, SlangArgumentType::Threads(_)));

        if has_axis && (has_groups || has_threads) {
            bail!("mixing groups/threads and axis is not supported");
        }

        let mut wrapper_arguments: Vec<String> = arguments
            .iter()
            .filter_map(|(name, arg_type, slang_type)| match arg_type {
                SlangArgumentType::Ptr => Some(format!("{} {}", slang_type, name)),
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
                .map(|(name, arg_type, _)| match arg_type {
                    SlangArgumentType::Ptr | SlangArgumentType::Constant(_) => name.clone(),
                    SlangArgumentType::Axis(_, _) => {
                        format!("__dsl_axis_idx.{}", axis_letters.next().unwrap())
                    },
                    SlangArgumentType::Groups(_) => {
                        format!("__dsl_group_idx.{}", group_letters.next().unwrap())
                    },
                    SlangArgumentType::Threads(_) => {
                        format!("__dsl_thread_idx.{}", thread_letters.next().unwrap())
                    },
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let numthreads = calculate_numthreads(&arguments)?;

        let body = format!("{}({});", underlying_call, underlying_arguments);

        let wrapper = format!(
            "[shader(\"compute\")]\n[numthreads({})]\nvoid {wrapper_name}({wrapper_arguments_str}) {{\n  {body}\n}}",
            numthreads
        );

        wrappers.push(wrapper);
    }

    Ok(wrappers.into())
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

fn calculate_numthreads(arguments: &[(String, SlangArgumentType, String)]) -> anyhow::Result<String> {
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
