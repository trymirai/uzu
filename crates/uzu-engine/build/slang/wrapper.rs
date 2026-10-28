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

    let mut wrapper = Vec::new();
    for specialize in &kernel.arguments {
        let SlangArgumentType::Specialize {
            ..
        } = &specialize.argument_type
        else {
            continue;
        };

        wrapper.push("[SpecializationConstant]".to_string());
        wrapper.push(format!("const {} {}_specialize_{};", specialize.slang_type, kernel.name, specialize.name));
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
            .map(|argument| (argument.name.as_str(), format!("{}_specialize_{}", kernel.name, argument.name)))
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
                SlangArgumentType::Constant {
                    ..
                } => {
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
        if has_axis {
            let axis_guards = kernel
                .arguments
                .iter()
                .filter_map(|argument| match &argument.argument_type {
                    SlangArgumentType::Axis {
                        threads,
                        threads_in_group: _,
                    } => Some(specializer.specialize(&slang_axis_expr(threads, kernel)).into_owned()),
                    _ => None,
                })
                .zip(["x", "y", "z"])
                .map(|(threads, axis)| format!("__dsl_axis_idx.{axis} >= uint({threads})"))
                .join(" || ");
            wrapper.push(format!("  if ({axis_guards}) {{ return; }}"));
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
                } => argument.name.to_string(),
                SlangArgumentType::Specialize {
                    ..
                } => {
                    format!("{}_specialize_{}", kernel.name, argument.name)
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
                SlangArgumentType::ThreadContext => "ThreadContext()".to_string(),
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

        wrapper.push("}".to_string());

        wrappers.push(wrapper.into_iter().join("\n"));
    }

    Ok(wrappers)
}

fn slang_axis_expr(
    expr: &str,
    kernel: &SlangKernel,
) -> String {
    let expr = rewrite_slang_method(expr, "unwrap_or", |receiver, fallback| {
        let condition = kernel
            .arguments
            .iter()
            .find_map(|argument| {
                if argument.name != receiver {
                    return None;
                }
                match &argument.argument_type {
                    SlangArgumentType::Constant {
                        condition,
                    } => condition.as_deref(),
                    _ => None,
                }
            })
            .expect("unwrap_or receiver must be an optional constant");
        format!("(({condition}) ? ({receiver}) : ({fallback}))")
    });
    let expr = rewrite_slang_method(&expr, "div_ceil", |receiver, argument| {
        format!("(({receiver}) / ({argument}) + uint(({receiver}) % ({argument}) != 0))")
    });
    let expr =
        rewrite_slang_method(&expr, "saturating_mul", |receiver, argument| format!("(({receiver}) * ({argument}))"));

    rewrite_slang_method(&expr, "saturating_sub", |receiver, argument| {
        format!("(({receiver}) > ({argument}) ? ({receiver}) - ({argument}) : 0)")
    })
}

fn rewrite_slang_method(
    expr: &str,
    method: &str,
    rewrite: impl Fn(&str, &str) -> String,
) -> String {
    let pattern = format!(".{method}(");
    let Some(dot) = expr.find(&pattern) else {
        return expr.to_string();
    };

    let receiver_start = expr[..dot].rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |idx| idx + 1);
    let argument_start = dot + pattern.len();
    let mut depth = 1;
    let mut argument_end = argument_start;
    for (offset, ch) in expr[argument_start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    argument_end = argument_start + offset;
                    break;
                }
            },
            _ => (),
        }
    }

    let receiver = &expr[receiver_start..dot];
    let argument = &expr[argument_start..argument_end];
    let rewritten = rewrite(receiver, argument);
    format!("{}{}{}", &expr[..receiver_start], rewritten, &expr[argument_end + 1..])
}
