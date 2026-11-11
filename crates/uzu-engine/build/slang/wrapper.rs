use std::{borrow::Cow, collections::HashSet, iter::repeat_n};

use anyhow::{Context, bail};
use itertools::Itertools;
use regex::Regex;
use shader_slang::{DeclKind, Module};

use crate::{
    common::mangling,
    slang::{
        SlangTarget, Specializer,
        reflection::{SlangArgumentType, SlangKernel, SlangParameterType},
        slang_axis_expr,
    },
};

/// The specialization constant of a kernel argument.
pub fn specialization_name(
    kernel: &str,
    argument: &str,
) -> String {
    format!("{kernel}_specialize_{argument}")
}

/// The entry points of the kernels of `module`, a block of declarations then one entry point per variant of each.
pub fn kernel_wrappers<T: SlangTarget>(
    module: &Module,
    kernels: &[SlangKernel],
) -> anyhow::Result<Vec<String>> {
    // A wrapper does not redeclare a specialization constant the module declares itself.
    let module_globals = module
        .module_reflection()
        .children()
        .filter(|decl| matches!(decl.kind(), DeclKind::Variable))
        .filter_map(|decl| decl.as_variable()?.name())
        .collect::<HashSet<_>>();
    let wrappers = kernels
        .iter()
        .map(|kernel| {
            kernel_wrapper::<T>(kernel, &module_globals)
                .with_context(|| format!("cannot generate wrappers for kernel '{}'", kernel.name))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(wrappers.into_iter().flatten().collect())
}

fn kernel_wrapper<T: SlangTarget>(
    kernel: &SlangKernel,
    module_globals: &HashSet<&str>,
) -> anyhow::Result<Vec<String>> {
    let inner_function_name = kernel.name.as_str();
    // A `[[PipelineVariants]]` uniform travels as a specialization constant too.
    let specialized = |argument_type: &SlangArgumentType, pipeline_variants: bool| {
        matches!(argument_type, SlangArgumentType::Specialize { .. }) || pipeline_variants
    };
    // A specialization a group size names defaults to 1, so the module compiles before it is specialized.
    let sizes = kernel
        .arguments
        .iter()
        .filter_map(|argument| match &argument.argument_type {
            SlangArgumentType::Axis {
                threads_in_group: size,
                ..
            }
            | SlangArgumentType::Threads {
                threads: size,
            } => Some(size.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut wrappers = Vec::new();

    let mut wrapper = Vec::new();
    for argument in &kernel.arguments {
        let name = specialization_name(&kernel.name, &argument.name);
        if !specialized(&argument.argument_type, argument.pipeline_variants) || module_globals.contains(name.as_str()) {
            continue;
        }
        let mentioned = Regex::new(&format!(r"\b{}\b", regex::escape(&argument.name)))?;
        let default = if sizes.iter().any(|size| mentioned.is_match(size)) {
            " = 1"
        } else {
            ""
        };
        wrapper.push("[SpecializationConstant]".to_string());
        wrapper.push(format!("const {} {name}{default};", T::specialization_type(&argument.slang_type)));
        wrapper.push(String::new());
    }
    wrappers.push(wrapper.into_iter().join("\n"));

    for variant in kernel.variants() {
        let outer_function_name =
            mangling::static_mangle(inner_function_name, variant.iter().map(|(_key, value)| *value));
        let specialization_names = kernel
            .arguments
            .iter()
            .filter(|argument| matches!(argument.argument_type, SlangArgumentType::Specialize { .. }))
            .map(|argument| (argument.name.as_str(), specialization_name(&kernel.name, &argument.name)))
            .collect::<Vec<_>>();
        let substitutions = variant
            .iter()
            .copied()
            .chain(specialization_names.iter().map(|(name, symbol)| (*name, symbol.as_str())))
            .collect::<Vec<_>>();
        let specializer = Specializer::new(&substitutions);

        let mut wrapper = Vec::new();

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
            wrapper.push(String::new());
            wrapper.push(format!("struct {base_name}_Struct: IGroupShared<{value_type}> {{"));
            wrapper.push(format!("  static func load(uint idx) -> {value_type} {{"));
            wrapper.push(format!("    return {base_name}_var[idx];"));
            wrapper.push("  }".to_string());
            wrapper.push(format!("  static func store(uint idx, {value_type} value) {{"));
            wrapper.push(format!("    {base_name}_var[idx] = value;"));
            wrapper.push("  }".to_string());
            wrapper.push("};".to_string());
            wrapper.push(String::new());
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

        wrapper.push("[shader(\"compute\")]".to_string());
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
                    wrapper.push(format!("  {} {},", specializer.specialize(&argument.slang_type), argument.name));
                },
                SlangArgumentType::Uniform {
                    ..
                } if !argument.pipeline_variants => {
                    wrapper.push(format!(
                        "  uniform {} {},",
                        specializer.specialize(&argument.slang_type),
                        argument.name
                    ));
                },
                SlangArgumentType::Axis {
                    ..
                } if !has_axis => {
                    has_axis = true;
                    wrapper.push("  uint3 __dsl_axis_idx : SV_DispatchThreadID,".to_string());
                },
                SlangArgumentType::Groups {
                    ..
                } if !has_groups => {
                    has_groups = true;
                    wrapper.push("  uint3 __dsl_groups_idx : SV_GroupID,".to_string());
                },
                SlangArgumentType::Threads {
                    ..
                } if !has_threads => {
                    has_threads = true;
                    wrapper.push("  uint3 __dsl_threads_idx : SV_GroupThreadID,".to_string());
                },
                _ => (),
            };
        }

        if has_axis && (has_groups || has_threads) {
            bail!("Cannot mix axis and groups/threads");
        }

        wrapper.push(") {".to_string());
        let prologue = T::entry_point_prologue(&outer_function_name);
        if !prologue.is_empty() {
            wrapper.push(prologue);
        }
        let axes = kernel
            .arguments
            .iter()
            .filter_map(|argument| match &argument.argument_type {
                SlangArgumentType::Axis {
                    threads,
                    threads_in_group,
                } => Some((
                    specializer.specialize(&slang_axis_expr(threads, kernel)).into_owned(),
                    specializer.specialize(threads_in_group).into_owned(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut axis_indices = Vec::new().into_iter();
        if has_axis {
            let (guard, indices) = T::axis_guard(&axes);
            wrapper.push(guard);
            axis_indices = indices.into_iter();
        }

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

        let mut group_letters = ["x", "y", "z"].iter();
        let mut thread_letters = ["x", "y", "z"].iter();
        let inner_function_arguments = kernel
            .arguments
            .iter()
            .map(|argument| {
                Ok(match &argument.argument_type {
                    SlangArgumentType::Buffer {
                        ..
                    } => argument.name.to_string(),
                    SlangArgumentType::Uniform {
                        ..
                    } if !argument.pipeline_variants => argument.name.to_string(),
                    SlangArgumentType::Specialize {
                        ..
                    }
                    | SlangArgumentType::Uniform {
                        ..
                    } => {
                        let constant = specialization_name(&kernel.name, &argument.name);
                        let slang_type = specializer.specialize(&argument.slang_type);
                        // A constant of another type converts back at the call.
                        if T::specialization_type(&slang_type) == slang_type {
                            constant
                        } else {
                            format!("{slang_type}({constant})")
                        }
                    },
                    SlangArgumentType::Axis {
                        ..
                    } => axis_indices.next().context("more Axis arguments than guarded axes")?,
                    SlangArgumentType::Groups {
                        ..
                    } => format!("__dsl_groups_idx.{}", group_letters.next().context("more than three Groups")?),
                    SlangArgumentType::Threads {
                        ..
                    } => format!("__dsl_threads_idx.{}", thread_letters.next().context("more than three Threads")?),
                    SlangArgumentType::ThreadContext => "ThreadContext()".to_string(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        if inner_function_parameters.is_empty() {
            wrapper.push(format!("  {inner_function_name}({});", inner_function_arguments.into_iter().join(", ")));
        } else {
            wrapper.push(format!(
                "  {inner_function_name}<{}>({});",
                inner_function_parameters.into_iter().join(", "),
                inner_function_arguments.into_iter().join(", ")
            ));
        }

        wrapper.push("}".to_string());

        wrappers.push(wrapper.into_iter().join("\n"));
    }

    Ok(wrappers)
}
