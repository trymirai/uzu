use std::{borrow::Cow, iter::repeat_n};

use anyhow::bail;
use itertools::Itertools;

use crate::{
    common::mangling,
    slang::{
        reflection::{SlangArgumentType, SlangKernel, SlangParameterType},
        types::Specializer,
    },
};

pub fn kernel_wrappers(kernel: &SlangKernel) -> anyhow::Result<Vec<String>> {
    let inner_function_name = kernel.name.as_str();

    let mut wrappers = Vec::new();
    for variant in kernel.variants() {
        let outer_function_name =
            mangling::static_mangle(inner_function_name, variant.iter().map(|(_key, value)| *value));
        let specializer = Specializer::new(&variant);

        let mut wrapper = Vec::new();

        for specialize in &kernel.arguments {
            let SlangArgumentType::Specialize = &specialize.argument_type else {
                continue;
            };

            wrapper.push(format!("[SpecializationConstant]"));
            wrapper.push(format!(
                "const {} {}_specialize_{};",
                specializer.specialize(&specialize.slang_type),
                outer_function_name,
                specialize.name
            ));
            wrapper.push(format!(""));
        }

        for parameter in &kernel.parameters {
            let SlangParameterType::GroupShared {
                value_type,
                length,
            } = &parameter.ty
            else {
                continue;
            };

            let base_name = format!("{}_groupshared_{}", outer_function_name, parameter.name);
            let value_type = specializer.specialize(value_type);
            let length = specializer.specialize(length);

            wrapper.push(format!("groupshared {value_type} {base_name}_var[{length}];"));
            wrapper.push(format!(""));
            wrapper.push(format!("struct {base_name}_Struct: IGroupShared<{value_type}> {{"));
            wrapper.push(format!("  static func load(uint idx) -> {value_type} {{"));
            wrapper.push(format!("    return {base_name}_var[idx];"));
            wrapper.push(format!("  }}"));
            wrapper.push(format!("  static func store(uint idx, {value_type} value) {{"));
            wrapper.push(format!("    {base_name}_var[idx] = value;"));
            wrapper.push(format!("  }}"));
            wrapper.push(format!("}};"));
            wrapper.push(format!(""));
        }

        let mut numthreads = kernel
            .arguments
            .iter()
            .filter_map(|argument| match &argument.argument_type {
                SlangArgumentType::Axis {
                    threads: _,
                    threads_in_group,
                } => Some(specializer.specialize(threads_in_group)),
                SlangArgumentType::Threads {
                    threads,
                } => Some(specializer.specialize(threads)),
                _ => None,
            })
            .collect::<Vec<_>>();

        if numthreads.len() > 3 {
            bail!("numthreads cannot be more than 3d: {numthreads:?}");
        }

        numthreads.extend(repeat_n(Cow::Borrowed("1"), 3 - numthreads.len()));

        wrapper.push(format!("[shader(\"compute\")]"));
        wrapper.push(format!("[numthreads({})]", numthreads.join(", ")));
        wrapper.push(format!("func {outer_function_name}("));

        let mut has_axis = false;
        let mut has_groups = false;
        let mut has_threads = false;

        for argument in &kernel.arguments {
            match &argument.argument_type {
                SlangArgumentType::Buffer {
                    ..
                } => {
                    wrapper.push(format!(
                        "  {} __dsl_arg_{},",
                        specializer.specialize(&argument.slang_type),
                        argument.name
                    ));
                },
                SlangArgumentType::Constant {
                    ..
                } => {
                    wrapper.push(format!(
                        "  uniform {} __dsl_arg_{},",
                        specializer.specialize(&argument.slang_type),
                        argument.name
                    ));
                },
                SlangArgumentType::Axis {
                    ..
                } if !has_axis => {
                    has_axis = true;
                    wrapper.push(format!("  uint3 __dsl_axis_idx : SV_DispatchThreadID,"));
                },
                SlangArgumentType::Groups {
                    ..
                } if !has_groups => {
                    has_groups = true;
                    wrapper.push(format!("  uint3 __dsl_groups_idx : SV_GroupID,"));
                },
                SlangArgumentType::Threads {
                    ..
                } if !has_threads => {
                    has_threads = true;
                    wrapper.push(format!("  uint3 __dsl_threads_idx : SV_GroupThreadID,"));
                },
                _ => (),
            };
        }

        if has_axis && (has_groups || has_threads) {
            bail!("Cannot mix axis and groups/threads");
        }

        wrapper.push(format!(") {{"));

        let inner_function_parameters = kernel
            .parameters
            .iter()
            .map(|parameter| match &parameter.ty {
                SlangParameterType::Type {
                    ..
                }
                | SlangParameterType::Value {
                    ..
                } => specializer.specialize(&parameter.name),
                SlangParameterType::GroupShared {
                    ..
                } => Cow::Owned(format!("{}_groupshared_{}_Struct", outer_function_name, parameter.name)),
            })
            .collect::<Vec<_>>();

        let mut group_axis_letters = ["x", "y", "z"].iter();
        let mut thread_axis_letters = ["x", "y", "z"].iter();
        let inner_function_arguments = kernel
            .arguments
            .iter()
            .map(|argument| match &argument.argument_type {
                SlangArgumentType::Buffer {
                    ..
                }
                | SlangArgumentType::Constant {
                    ..
                } => format!("__dsl_arg_{}", argument.name),
                SlangArgumentType::Specialize => {
                    format!("{}_specialize_{}", outer_function_name, argument.name)
                },
                SlangArgumentType::Axis {
                    ..
                } => format!("__dsl_axis_idx.{}", group_axis_letters.next().unwrap()),
                SlangArgumentType::Groups {
                    ..
                } => format!("__dsl_groups_idx.{}", group_axis_letters.next().unwrap()),
                SlangArgumentType::Threads {
                    ..
                } => format!("__dsl_threads_idx.{}", thread_axis_letters.next().unwrap()),
                SlangArgumentType::ThreadContext {
                    ..
                } => format!("ThreadContext()"),
            })
            .collect::<Vec<_>>();

        if inner_function_parameters.is_empty() {
            wrapper.push(format!("  {inner_function_name}({});", inner_function_arguments.into_iter().join(", ")));
        } else {
            wrapper.push(format!(
                "  {inner_function_name}<{}>({});",
                inner_function_parameters.into_iter().join(", "),
                inner_function_arguments.into_iter().join(", ")
            ));
        }

        wrapper.push(format!("}}"));

        wrappers.push(wrapper.into_iter().join("\n"));
    }

    Ok(wrappers)
}
