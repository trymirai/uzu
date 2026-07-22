//! Enumeration of a kernel's variant space.
//!
//! A templated kernel declares axes; a `#[variant_group]` binds some of them into one
//! sum type; `CONSTRAINT`s prune the rest. What survives is the exact set of
//! template-argument tuples the build instantiates, so this module is the single source
//! of truth for which kernels ship.

use std::collections::HashMap;

use anyhow::{Context, bail};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{
    constraint_expr::{self, AxisDecl, ConstraintSet},
    enum_paths::EnumPaths,
    gpu_types::{GpuTypeVariantGroup, VariantGroupArm},
    mangling::static_mangle,
};

/// One template axis as igata sees it: its name, its type, and the values the shader
/// declared for it, spelled exactly as they will be substituted into the template.
#[derive(Clone, Debug)]
pub struct AxisSpec {
    pub name: Box<str>,
    pub ty: constraint_expr::Type,
    pub values: Box<[Box<str>]>,
}

/// A kernel's variant space: the axes it declares and the constraints pruning them.
pub struct KernelSpace<'a> {
    pub name: &'a str,
    /// `None` for a kernel that is not a template at all.
    pub axes: Option<&'a [AxisSpec]>,
    pub constraints: &'a [Box<str>],
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
    Axis(&'a AxisSpec),
    Group {
        type_name: &'a str,
        axes: &'a [Box<str>],
    },
}

impl<'a> KernelSpace<'a> {
    fn axes(&self) -> &'a [AxisSpec] {
        self.axes.unwrap_or_default()
    }

    /// The kernel's constraints, type-checked against its axes.
    pub fn constraint_set(
        &self,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<ConstraintSet> {
        let axes = self
            .axes()
            .iter()
            .map(|axis| {
                let values = axis
                    .values
                    .iter()
                    .map(|text| {
                        AxisDecl::parse_value(&axis.ty, text)
                            .with_context(|| format!("in VARIANTS({}, ...) of kernel `{}`", axis.name, self.name))
                    })
                    .collect::<anyhow::Result<Box<[_]>>>()?;
                Ok(AxisDecl {
                    name: axis.name.clone(),
                    ty: axis.ty.clone(),
                    values,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        ConstraintSet::compile(self.name, axes, enum_paths.helpers().clone(), self.constraints.iter())
    }

    /// The key fields for this kernel, in template parameter order.
    pub fn key_layout(
        &self,
        enum_paths: &'a EnumPaths,
    ) -> Vec<KeyField<'a>> {
        let Some(axes) = self.axes else {
            return Vec::new();
        };
        let groups = groups_of(axes, enum_paths);

        axes.iter()
            .filter_map(|axis| match groups.iter().find(|group| group.axes.contains(&axis.name)) {
                Some(group) if group.axes.first() == Some(&axis.name) => Some(KeyField::Group {
                    type_name: &group.name,
                    axes: &group.axes,
                }),
                Some(_) => None,
                None => Some(KeyField::Axis(axis)),
            })
            .collect()
    }

    /// Template-argument tuples this kernel is instantiated for: the declared
    /// cross-product minus everything its `CONSTRAINT`s reject. A non-templated kernel
    /// yields a single `None`.
    ///
    /// Wrapper emission, bindings and the build manifest all enumerate variants through
    /// here, so none of them can drift from what actually gets compiled.
    pub fn accepted_variants(
        &self,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<Vec<Option<Vec<(String, String)>>>> {
        let Some(axes) = self.axes else {
            return Ok(vec![None]);
        };

        let constraints = self.constraint_set(enum_paths)?;
        let dimensions = self.dimensions(axes, enum_paths)?;

        let variants = dimensions
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

                axes.iter()
                    .map(|axis| (axis.name.to_string(), bound[axis.name.as_ref()].to_string()))
                    .collect::<Vec<_>>()
            })
            .filter_map(|type_variant| match constraints.satisfied(&type_variant) {
                Ok(true) => Some(Ok(Some(type_variant))),
                Ok(false) => None,
                Err(error) => Some(Err(error.context(format!("in kernel `{}`", self.name)))),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Two tuples that mangle alike would emit the same entry point twice: an
        // overlapping pair of variant-group arms, say. The Metal compiler reports that as
        // a redefinition deep in generated code, so name the tuples instead.
        let mut seen: HashMap<String, &Vec<(String, String)>> = HashMap::new();
        for type_variant in variants.iter().flatten() {
            let name = static_mangle(self.name, type_variant.iter().map(|(_, value)| value.as_str()));
            if let Some(previous) = seen.insert(name, type_variant) {
                let render =
                    |variant: &Vec<(String, String)>| variant.iter().map(|(a, v)| format!("{a}={v}")).join(" ");
                bail!(
                    "kernel `{}` instantiates the same entry point twice: `{}` and `{}`",
                    self.name,
                    render(previous),
                    render(type_variant),
                );
            }
        }

        Ok(variants)
    }

    /// Splits the kernel's axes into dimensions, collapsing any variant group whose axes
    /// the kernel declares in full.
    fn dimensions(
        &self,
        axes: &[AxisSpec],
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<Vec<Dimension>> {
        let groups = groups_of(axes, enum_paths);

        if let Some((first, second)) =
            groups.iter().tuple_combinations().find(|(a, b)| a.axes.iter().any(|axis| b.axes.contains(axis)))
        {
            bail!("variant groups `{}` and `{}` share an axis of kernel `{}`", first.name, second.name, self.name);
        }

        let mut dimensions: Vec<Dimension> = Vec::new();
        for axis in axes {
            match groups.iter().find(|group| group.axes.contains(&axis.name)) {
                // A group is emitted once, at its first axis.
                Some(group) if group.axes.first() == Some(&axis.name) => {
                    let tuples = group_tuples(group, axes, enum_paths)
                        .with_context(|| format!("in variant group `{}` of kernel `{}`", group.name, self.name))?;
                    dimensions.push(Dimension {
                        axes: group.axes.to_vec(),
                        tuples,
                    });
                },
                Some(_) => (),
                None => dimensions.push(Dimension {
                    axes: vec![axis.name.clone()],
                    tuples: axis.values.iter().map(|value| vec![value.clone()]).collect(),
                }),
            }
        }

        Ok(dimensions)
    }
}

/// The variant groups every one of whose axes this kernel declares.
fn groups_of<'a>(
    axes: &[AxisSpec],
    enum_paths: &'a EnumPaths,
) -> Vec<&'a GpuTypeVariantGroup> {
    let declares = |axis: &str| axes.iter().any(|a| a.name.as_ref() == axis);
    enum_paths.variant_groups().iter().filter(|group| group.axes.iter().all(|axis| declares(axis))).collect()
}

/// What a variant group's arms say about the axes, resolved once: every struct arm's
/// fields, and for each of them the axis value that each member of the field's enum
/// selects — by variant name for an enum axis, by discriminant for a numeric one.
struct GroupSelections<'a> {
    arms: Vec<ArmSelection<'a>>,
    /// The unit arm's name and the value it leaves each axis at, if the group has one.
    unit: Option<(&'a str, Vec<&'a Box<str>>)>,
}

struct ArmSelection<'a> {
    name: &'a str,
    /// One per axis, in axis order.
    fields: Vec<FieldSelection<'a>>,
}

struct FieldSelection<'a> {
    field: &'a str,
    field_type: &'a str,
    /// `(member, axis value)` for every member of the field's enum.
    members: Vec<(&'a str, &'a Box<str>)>,
}

fn selections<'a>(
    group: &'a GpuTypeVariantGroup,
    axes: &'a [AxisSpec],
    enum_paths: &'a EnumPaths,
) -> anyhow::Result<GroupSelections<'a>> {
    let declared = group
        .axes
        .iter()
        .map(|axis| {
            let spec = axes.iter().find(|a| &a.name == axis).context("axis is not declared")?;
            Ok(&*spec.values)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let arms = group
        .arms
        .iter()
        .filter_map(|arm| match arm {
            VariantGroupArm::Product {
                name,
                fields,
            } => Some((name, fields)),
            VariantGroupArm::Unit {
                ..
            } => None,
        })
        .map(|(name, fields)| {
            let fields = fields
                .iter()
                .zip(group.axes.iter())
                .zip(declared.iter())
                .map(|(((field, field_type), axis), axis_values)| {
                    let members = enum_paths
                        .variants_for(field_type)
                        .with_context(|| format!("`{field_type}` is not an enum gpu type"))?
                        .iter()
                        .map(|(member, discriminant)| {
                            let value = axis_value_for(axis_values, member, *discriminant).with_context(|| {
                                format!("`{field_type}::{member}` does not match any declared value of axis `{axis}`")
                            })?;
                            Ok((member.as_ref(), value))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    Ok(FieldSelection {
                        field,
                        field_type,
                        members,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(ArmSelection {
                name,
                fields,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // A unit arm means "whatever the struct arms leave over", which is what makes
    // `FullPrecision` mean "zero bits, zero group size" without anyone writing that down.
    let unit_name = group.arms.iter().find_map(|arm| match arm {
        VariantGroupArm::Unit {
            name,
        } => Some(name.as_ref()),
        VariantGroupArm::Product {
            ..
        } => None,
    });

    let unit = unit_name
        .map(|name| -> anyhow::Result<_> {
            let values = group
                .axes
                .iter()
                .enumerate()
                .zip(declared.iter())
                .map(|((axis_index, axis), axis_values)| {
                    let covered = arms
                        .iter()
                        .flat_map(|arm| arm.fields[axis_index].members.iter().map(|(_, value)| *value))
                        .collect::<Vec<_>>();
                    let mut remaining = axis_values.iter().filter(|value| !covered.contains(value));
                    let (Some(value), None) = (remaining.next(), remaining.next()) else {
                        bail!(
                            "axis `{axis}` must have exactly one value left over for the unit variant, but its \
                             declared values are {axis_values:?} and the struct variants cover {covered:?}",
                        );
                    };
                    Ok(value)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((name, values))
        })
        .transpose()?;

    Ok(GroupSelections {
        arms,
        unit,
    })
}

/// The axis-value tuples a variant group's Rust definition admits: the product of each
/// struct arm's fields, plus the unit arm's leftovers.
fn group_tuples(
    group: &GpuTypeVariantGroup,
    axes: &[AxisSpec],
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<Vec<Box<str>>>> {
    let selections = selections(group, axes, enum_paths)?;

    let mut tuples: Vec<Vec<Box<str>>> = selections
        .arms
        .iter()
        .flat_map(|arm| {
            arm.fields
                .iter()
                .map(|field| field.members.iter().map(|(_, value)| (*value).clone()))
                .multi_cartesian_product()
        })
        .collect();

    if let Some((_, values)) = &selections.unit {
        tuples.push(values.iter().map(|value| (*value).clone()).collect());
    }

    Ok(tuples)
}

/// The `to_template_args()` that flattens a variant group back into the axes it stands
/// for -- the inverse of the enumeration above, emitted from the same resolved arms so
/// the two cannot disagree about what an arm means.
pub fn flattening_impl(
    group: &GpuTypeVariantGroup,
    axes: &[AxisSpec],
    enum_paths: &EnumPaths,
) -> anyhow::Result<TokenStream> {
    let selections = selections(group, axes, enum_paths)?;
    let grouped_axes = group
        .axes
        .iter()
        .map(|axis| axes.iter().find(|a| &a.name == axis).context("axis is not declared"))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let group_path: syn::Path = syn::parse_str(
        enum_paths.variant_group_path(&group.name).with_context(|| format!("no Rust path for `{}`", group.name))?,
    )?;

    let types = grouped_axes.iter().map(|axis| axis_rust_type(axis, enum_paths)).collect::<anyhow::Result<Vec<_>>>()?;

    let mut arms = Vec::new();
    for arm in selections.arms.iter() {
        let arm_name = format_ident!("{}", arm.name);
        let bindings = arm.fields.iter().map(|field| format_ident!("{}", field.field));
        let values = arm
            .fields
            .iter()
            .zip(grouped_axes.iter())
            .map(|(field, axis)| {
                let binding = format_ident!("{}", field.field);
                let field_path: syn::Path = syn::parse_str(
                    enum_paths
                        .full_path_for(field.field_type)
                        .with_context(|| format!("no Rust path for enum `{}`", field.field_type))?,
                )?;
                let members = field
                    .members
                    .iter()
                    .map(|(member, value)| {
                        let member = format_ident!("{member}");
                        let value = axis_literal(axis, value, enum_paths)?;
                        Ok(quote! { #field_path::#member => #value })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                Ok(quote! { match #binding { #(#members,)* } })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        arms.push(quote! { Self::#arm_name { #(#bindings),* } => (#(#values),*) });
    }

    if let Some((name, values)) = &selections.unit {
        let name = format_ident!("{name}");
        let values = values
            .iter()
            .zip(grouped_axes.iter())
            .map(|(value, axis)| axis_literal(axis, value, enum_paths))
            .collect::<anyhow::Result<Vec<_>>>()?;
        arms.push(quote! { Self::#name => (#(#values),*) });
    }

    let documentation = format!(
        " Flattens back to the shader's {} arguments.",
        group.axes.iter().map(|axis| format!("`{axis}`")).join(", ")
    );

    Ok(quote! {
        #[allow(clippy::style, clippy::complexity, dead_code)]
        impl #group_path {
            #[doc = #documentation]
            pub fn to_template_args(self) -> (#(#types),*) {
                match self {
                    #(#arms,)*
                }
            }
        }
    })
}

/// The Rust type holding one axis's value.
pub fn axis_rust_type(
    axis: &AxisSpec,
    enum_paths: &EnumPaths,
) -> anyhow::Result<syn::Type> {
    Ok(match &axis.ty {
        constraint_expr::Type::DType => syn::parse_str("crate::data_type::DataType")?,
        constraint_expr::Type::Bool => syn::parse_str("bool")?,
        constraint_expr::Type::Int => syn::parse_str("u32")?,
        constraint_expr::Type::Enum(name) => {
            syn::parse_str(enum_paths.full_path_for(name).with_context(|| format!("no Rust path for enum `{name}`"))?)?
        },
    })
}

/// One of an axis's declared values, as a Rust literal.
pub fn axis_literal(
    axis: &AxisSpec,
    value: &str,
    enum_paths: &EnumPaths,
) -> anyhow::Result<TokenStream> {
    Ok(match &axis.ty {
        constraint_expr::Type::DType => {
            let variant = format_ident!("{}", data_type_variant(value)?);
            quote! { crate::data_type::DataType::#variant }
        },
        constraint_expr::Type::Bool => {
            let value: bool = value.parse()?;
            quote! { #value }
        },
        constraint_expr::Type::Int => {
            let value: u32 = value.parse()?;
            quote! { #value }
        },
        constraint_expr::Type::Enum(_) => {
            let (enum_name, variant) =
                value.rsplit_once("::").with_context(|| format!("`{value}` is not an enum variant"))?;
            let path: syn::Path = syn::parse_str(
                enum_paths.full_path_for(enum_name).with_context(|| format!("no Rust path for enum `{enum_name}`"))?,
            )?;
            let variant = format_ident!("{variant}");
            quote! { #path::#variant }
        },
    })
}

fn data_type_variant(metal_type: &str) -> anyhow::Result<&'static str> {
    Ok(match metal_type {
        "float" => "F32",
        "half" => "F16",
        "bfloat" => "BF16",
        other => bail!("no DataType corresponds to the Metal type `{other}`"),
    })
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
