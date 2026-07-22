use std::{collections::HashMap, iter::once};

use anyhow::{Context, bail};
use itertools::Itertools;

use super::{
    ast::{
        MetalArgument, MetalArgumentType, MetalKernelInfo, MetalTemplateParameter, MetalTemplateParameterType,
        shared_element_type,
    },
    enum_path_rewrite::is_enum_c_type,
};
use crate::common::{
    constraint_expr::{self, ConstraintSet},
    enum_paths::EnumPaths,
    gpu_types::{GpuTypeVariantGroup, VariantGroupArm},
    identifiers::KernelName,
    mangling::static_mangle,
};

pub type SpecializeBaseIndices = HashMap<KernelName, usize>;

pub fn wrappers(
    kernels: &[MetalKernelInfo],
    enum_paths: &EnumPaths,
) -> anyhow::Result<(Box<[Box<str>]>, SpecializeBaseIndices)> {
    let mut all_wrappers = Vec::new();
    let mut base_indices = SpecializeBaseIndices::new();
    let mut next_index = 0usize;

    for kernel in kernels {
        let specialize_count =
            kernel.arguments.iter().filter(|a| matches!(&a.argument_type, MetalArgumentType::Specialize(_))).count();

        if specialize_count > 0 {
            base_indices.insert(kernel.name.clone(), next_index);
            next_index += specialize_count;
        }

        let kernel_wrappers = kernel_wrappers(kernel, base_indices.get(&kernel.name), enum_paths)?;
        all_wrappers.extend(kernel_wrappers.into_vec());
    }

    Ok((all_wrappers.into_boxed_slice(), base_indices))
}

struct SpecializeBinding<'a> {
    argument: &'a MetalArgument,
    name_suffix: String,
    enum_type: Option<String>,
}

impl SpecializeBinding<'_> {
    fn slot_name(&self) -> String {
        format!("__dsl_specialize_{}", self.name_suffix)
    }

    fn typed_name(&self) -> String {
        format!("__dsl_typed_{}", self.name_suffix)
    }

    fn slot_type(&self) -> &str {
        match self.enum_type {
            Some(_) => "uint32_t",
            None => self.argument.c_type.trim_start_matches("const ").trim(),
        }
    }

    fn alias_target(&self) -> String {
        if self.enum_type.is_some() {
            self.typed_name()
        } else {
            self.slot_name()
        }
    }
}

fn specialize_bindings<'a>(
    kernel: &'a MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> Vec<SpecializeBinding<'a>> {
    kernel
        .arguments
        .iter()
        .filter(|a| matches!(&a.argument_type, MetalArgumentType::Specialize(_)))
        .map(|argument| {
            let name_suffix = format!("{}_{}", kernel.name, argument.name);
            let enum_type = is_enum_c_type(enum_paths, &argument.c_type)
                .then(|| argument.c_type.trim_start_matches("const ").trim().to_string());
            SpecializeBinding {
                argument,
                name_suffix,
                enum_type,
            }
        })
        .collect()
}

fn kernel_header(
    bindings: &[SpecializeBinding<'_>],
    base_index: usize,
) -> String {
    let mut lines = Vec::new();

    for (offset, binding) in bindings.iter().enumerate() {
        lines.push(format!(
            "constant {} {} [[function_constant({})]];",
            binding.slot_type(),
            binding.slot_name(),
            base_index + offset,
        ));
    }

    for binding in bindings.iter() {
        if let Some(enum_type) = &binding.enum_type {
            lines.push(format!(
                "constant {ty} {name} = {ty}({slot});",
                ty = enum_type,
                name = binding.typed_name(),
                slot = binding.slot_name(),
            ));
        }
    }

    for binding in bindings.iter() {
        lines.push(format!("#define {} {}", binding.argument.name, binding.alias_target()));
    }

    lines.push(String::new());
    lines.join("\n")
}

fn kernel_footer(bindings: &[SpecializeBinding<'_>]) -> String {
    bindings.iter().map(|b| format!("#undef {}\n", b.argument.name)).collect()
}

/// The axis type a constraint expression sees for a template parameter.
fn axis_type(parameter: &MetalTemplateParameter) -> anyhow::Result<constraint_expr::Type> {
    Ok(match &parameter.ty {
        MetalTemplateParameterType::Type => constraint_expr::Type::DType,
        MetalTemplateParameterType::Value(rust_type) => match rust_type.as_ref() {
            "bool" => constraint_expr::Type::Bool,
            "u32" | "i32" => constraint_expr::Type::Int,
            path => match path.rsplit_once("::") {
                Some((_, short_name)) => constraint_expr::Type::Enum(short_name.into()),
                None => bail!("template parameter `{}` has unsupported type `{path}`", parameter.name),
            },
        },
    })
}

pub fn constraint_set(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> anyhow::Result<ConstraintSet> {
    let axes = kernel
        .variants
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|parameter| {
            let ty = axis_type(parameter)?;
            let values = parameter
                .variants
                .iter()
                .map(|text| {
                    constraint_expr::AxisDecl::parse_value(&ty, text)
                        .with_context(|| format!("in VARIANTS({}, ...) of kernel `{}`", parameter.name, kernel.name))
                })
                .collect::<anyhow::Result<Box<[_]>>>()?;
            Ok(constraint_expr::AxisDecl {
                name: parameter.name.clone(),
                ty,
                values,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let helpers = enum_paths
        .helpers()
        .iter()
        .map(|(name, helper)| {
            (
                name.clone(),
                constraint_expr::Helper {
                    parameter: constraint_expr::Type::Enum(helper.parameter.clone()),
                    values: helper.values.iter().map(|(variant, value)| (variant.clone(), i64::from(*value))).collect(),
                },
            )
        })
        .collect();

    ConstraintSet::compile(kernel.name.as_ref(), axes, helpers, kernel.constraints.iter())
}

/// One independently varying dimension of a kernel's variant space: usually a single
/// axis, but a `#[variant_group]` binds several axes into one dimension whose values are
/// whole tuples, so combinations its Rust sum type cannot represent are never generated.
struct Dimension {
    axes: Vec<Box<str>>,
    tuples: Vec<Vec<Box<str>>>,
}

/// How a kernel's axes map onto fields of its runtime key: grouped axes collapse into
/// the one sum type that represents them, the rest keep a field each.
pub enum KeyField<'a> {
    Axis(&'a MetalTemplateParameter),
    Group {
        type_name: &'a str,
        axes: &'a [Box<str>],
    },
}

/// The key fields for a kernel, in template parameter order.
pub fn key_layout<'a>(
    kernel: &'a MetalKernelInfo,
    enum_paths: &'a EnumPaths,
) -> Vec<KeyField<'a>> {
    let Some(parameters) = &kernel.variants else {
        return Vec::new();
    };

    let declares = |axis: &str| parameters.iter().any(|p| p.name.as_ref() == axis);
    let groups = enum_paths
        .variant_groups()
        .iter()
        .filter(|group| group.axes.iter().all(|axis| declares(axis)))
        .collect::<Vec<_>>();

    parameters
        .iter()
        .filter_map(|parameter| match groups.iter().find(|group| group.axes.contains(&parameter.name)) {
            Some(group) if group.axes.first() == Some(&parameter.name) => Some(KeyField::Group {
                type_name: &group.name,
                axes: &group.axes,
            }),
            Some(_) => None,
            None => Some(KeyField::Axis(parameter)),
        })
        .collect()
}

/// Splits a kernel's axes into dimensions, collapsing any variant group whose axes the
/// kernel declares in full.
fn dimensions(
    kernel: &MetalKernelInfo,
    parameters: &[MetalTemplateParameter],
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<Dimension>> {
    let declares = |axis: &str| parameters.iter().any(|p| p.name.as_ref() == axis);

    let groups = enum_paths
        .variant_groups()
        .iter()
        .filter(|group| group.axes.iter().all(|axis| declares(axis)))
        .collect::<Vec<_>>();

    if let Some((first, second)) =
        groups.iter().tuple_combinations().find(|(a, b)| a.axes.iter().any(|axis| b.axes.contains(axis)))
    {
        bail!("variant groups `{}` and `{}` share an axis of kernel `{}`", first.name, second.name, kernel.name);
    }

    let mut dimensions: Vec<Dimension> = Vec::new();
    for parameter in parameters {
        match groups.iter().find(|group| group.axes.contains(&parameter.name)) {
            // A group is emitted once, at its first axis.
            Some(group) if group.axes.first() == Some(&parameter.name) => {
                let tuples = group_tuples(group, parameters, enum_paths)
                    .with_context(|| format!("in variant group `{}` of kernel `{}`", group.name, kernel.name))?;
                dimensions.push(Dimension {
                    axes: group.axes.to_vec(),
                    tuples,
                });
            },
            Some(_) => (),
            None => dimensions.push(Dimension {
                axes: vec![parameter.name.clone()],
                tuples: parameter.variants.iter().map(|value| vec![value.clone()]).collect(),
            }),
        }
    }

    Ok(dimensions)
}

/// The axis-value tuples a variant group's Rust definition admits.
///
/// Each field of a struct arm names an enum whose members select values of the
/// corresponding axis — by variant name for an enum axis, by discriminant for a numeric
/// one. A unit arm takes each axis's remaining value, which is what makes
/// `FullPrecision` mean "zero bits, zero group size" without anyone writing that down.
fn group_tuples(
    group: &GpuTypeVariantGroup,
    parameters: &[MetalTemplateParameter],
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<Vec<Box<str>>>> {
    let declared = group
        .axes
        .iter()
        .map(|axis| {
            let parameter = parameters.iter().find(|p| &p.name == axis).context("axis is not declared")?;
            Ok(&*parameter.variants)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut tuples: Vec<Vec<Box<str>>> = Vec::new();
    let mut covered: Vec<Vec<Box<str>>> = vec![Vec::new(); group.axes.len()];

    for arm in group.arms.iter() {
        let VariantGroupArm::Product {
            fields,
            ..
        } = arm
        else {
            continue;
        };

        let per_axis = fields
            .iter()
            .map(|(_, field_type)| field_type)
            .zip(group.axes.iter())
            .zip(declared.iter())
            .map(|((field_type, axis), axis_values)| {
                let members = enum_paths
                    .variants_for(field_type)
                    .with_context(|| format!("`{field_type}` is not an enum gpu type"))?;

                members
                    .iter()
                    .map(|(member, discriminant)| {
                        axis_value_for(axis_values, member, *discriminant).cloned().with_context(|| {
                            format!("`{field_type}::{member}` does not match any declared value of axis `{axis}`")
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        for (axis_index, values) in per_axis.iter().enumerate() {
            covered[axis_index].extend(values.iter().cloned());
        }

        tuples.extend(per_axis.into_iter().multi_cartesian_product());
    }

    if group.arms.iter().any(|arm| matches!(arm, VariantGroupArm::Unit { .. })) {
        let unit_tuple = group
            .axes
            .iter()
            .zip(declared.iter())
            .zip(covered.iter())
            .map(|((axis, axis_values), covered)| {
                let mut remaining = axis_values.iter().filter(|value| !covered.contains(value));
                let (Some(value), None) = (remaining.next(), remaining.next()) else {
                    bail!(
                        "axis `{axis}` must have exactly one value left over for the unit variant, but its declared \
                         values are {:?} and the struct variants cover {covered:?}",
                        axis_values,
                    );
                };
                Ok(value.clone())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        tuples.push(unit_tuple);
    }

    Ok(tuples)
}

/// Matches an enum member against an axis's declared values: by variant name when the
/// axis is enum-typed, by discriminant when it is numeric.
fn axis_value_for<'a>(
    axis_values: &'a [Box<str>],
    member: &str,
    discriminant: u32,
) -> Option<&'a Box<str>> {
    axis_values.iter().find(|value| match value.rsplit_once("::") {
        Some((_, variant)) => variant == member,
        None => value.parse::<u32>() == Ok(discriminant),
    })
}

/// Template-argument tuples this kernel is instantiated for: the declared cross-product
/// minus everything its CONSTRAINTs reject. Non-templated kernels yield a single `None`.
///
/// The single source of truth for the shipped kernel set — both the wrapper emission
/// below and the build manifest enumerate variants through here, so the manifest cannot
/// drift from what actually gets compiled.
pub fn accepted_variants(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<Option<Vec<(String, String)>>>> {
    let Some(parameters) = &kernel.variants else {
        return Ok(vec![None]);
    };

    let constraints = constraint_set(kernel, enum_paths)?;
    let dimensions = dimensions(kernel, parameters, enum_paths)?;

    dimensions
        .iter()
        .map(|dimension| dimension.tuples.iter())
        .multi_cartesian_product()
        .map(|choice| {
            // Back to template parameter order: mangled entry point names depend on it.
            let bound: HashMap<&str, &str> = dimensions
                .iter()
                .zip(choice)
                .flat_map(|(dimension, tuple)| {
                    dimension.axes.iter().zip(tuple).map(|(axis, value)| (axis.as_ref(), value.as_ref()))
                })
                .collect();

            parameters.iter().map(|p| (p.name.to_string(), bound[p.name.as_ref()].to_string())).collect::<Vec<_>>()
        })
        .filter_map(|type_variant| match constraints.satisfied(&type_variant) {
            Ok(true) => Some(Ok(Some(type_variant))),
            Ok(false) => None,
            Err(error) => Some(Err(error.context(format!("in kernel `{}`", kernel.name)))),
        })
        .collect()
}

fn kernel_wrappers(
    kernel: &MetalKernelInfo,
    base_index: Option<&usize>,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();
    let bindings = specialize_bindings(kernel, enum_paths);

    if let Some(&base) = base_index {
        kernel_wrappers.push(kernel_header(&bindings, base).into_boxed_str());
    }

    for type_variant in accepted_variants(kernel, enum_paths)? {
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
            .filter_map(|a| match &a.argument_type {
                MetalArgumentType::Axis(_, l) | MetalArgumentType::Threads(l) => Some(format!("({l})")),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_total_threads_per_threadgroup = if !max_total_threads_per_threadgroup.is_empty() {
            max_total_threads_per_threadgroup.join(" * ")
        } else {
            "1".to_string()
        };

        let (mut wrapper_arguments, condition_definitions): (Vec<_>, Vec<_>) = kernel
            .arguments
            .iter()
            .filter(|a| matches!(&a.argument_type, MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_)))
            .enumerate()
            .map(|(i, a)| {
                let condition_field = format!("__dsl_buffer_condition_{}_{}", wrapper_name, a.name);
                match a.condition.as_deref() {
                    Some(condition) => (
                        format!("{} {} [[buffer({}), function_constant({})]]", a.c_type, a.name, i, condition_field,),
                        Some(format!("constant bool {} = ({});", condition_field, condition)),
                    ),
                    None => (format!("{} {} [[buffer({})]]", a.c_type, a.name, i), None),
                }
            })
            .unzip();

        for (threadgroup_index, argument) in optional_shared_arguments(kernel).enumerate() {
            let element_type = shared_element_type(&argument.c_type);
            wrapper_arguments.push(format!("{element_type}* {} [[threadgroup({threadgroup_index})]]", argument.name));
        }

        if kernel.has_axis() {
            if kernel.has_groups() || kernel.has_threads() {
                bail!("mixing groups/threads and axis is not supported");
            }

            wrapper_arguments.push("uint3 __dsl_axis_idx [[thread_position_in_grid]]".into());
        }

        if kernel.has_groups() || kernel.has_thread_context() {
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

        let shared_definitions = kernel.arguments.iter().filter_map(|a| match &a.argument_type {
            MetalArgumentType::Shared(dimensions) if a.condition.is_none() => {
                let element_type = shared_element_type(&a.c_type);
                let dimensions = dimensions.as_deref().map(|d| format!("[{d}]")).unwrap_or_default();
                Some(format!("{element_type} {}{dimensions}", a.name))
            },
            _ => None,
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match &a.argument_type {
                    MetalArgumentType::Shared(_) if a.condition.is_some() => {
                        format!("({}){}", a.c_type, a.name)
                    },
                    MetalArgumentType::Buffer(_)
                    | MetalArgumentType::Constant(_)
                    | MetalArgumentType::Shared(_)
                    | MetalArgumentType::Specialize(_) => a.name.to_string(),
                    MetalArgumentType::Axis(..) => {
                        format!("__dsl_axis_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Groups(_) => {
                        format!("__dsl_group_idx.{}", group_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::Threads(_) => {
                        format!("__dsl_thread_idx.{}", thread_axis_letters.next().unwrap())
                    },
                    MetalArgumentType::ThreadContext => "ThreadContext { .simd_lane_id = __dsl_simd_lane_idx, .simdgroup_index = __dsl_simd_group_idx, .simdgroup_size = __dsl_simd_group_size, .simdgroups_per_threadgroup = __dsl_simd_group_per_threadgroup, .threadgroup_position = __dsl_group_idx }".into(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call = format!("{underlying_name}({underlying_arguments})");

        let wrapper_body =
            shared_definitions.chain(once(underlying_call)).map(|l| format!("  {l};\n")).collect::<Vec<_>>().join("");

        let (defs, undefs): (Vec<_>, Vec<_>) = type_variant
            .unwrap_or_default()
            .iter()
            .map(|(k, v)| (format!("\n#define {k} {v}"), format!("#undef {k}\n")))
            .unzip();

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

    if base_index.is_some() {
        kernel_wrappers.push(kernel_footer(&bindings).into_boxed_str());
    }

    Ok(kernel_wrappers.into())
}

fn optional_shared_arguments(kernel: &MetalKernelInfo) -> impl Iterator<Item = &MetalArgument> {
    kernel.arguments.iter().filter(|argument| argument.is_optional_shared())
}
