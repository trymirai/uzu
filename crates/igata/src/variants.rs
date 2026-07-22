//! Enumeration of a kernel's variant space.
//!
//! A templated kernel declares axes; a `#[variant_group]` binds some of them into one
//! sum type; `CONSTRAINT`s prune the rest. What survives is the exact set of
//! template-argument tuples the build instantiates, so this module is the single source
//! of truth for which kernels ship.

use std::collections::HashMap;

use anyhow::{Context, bail};
use itertools::Itertools;

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

        ConstraintSet::compile(self.name, axes, enum_paths.constraint_helpers(), self.constraints.iter())
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

/// The axis-value tuples a variant group's Rust definition admits.
///
/// Each field of a struct arm names an enum whose members select values of the
/// corresponding axis — by variant name for an enum axis, by discriminant for a numeric
/// one. A unit arm takes each axis's remaining value, which is what makes
/// `FullPrecision` mean "zero bits, zero group size" without anyone writing that down.
fn group_tuples(
    group: &GpuTypeVariantGroup,
    axes: &[AxisSpec],
    enum_paths: &EnumPaths,
) -> anyhow::Result<Vec<Vec<Box<str>>>> {
    let declared = group
        .axes
        .iter()
        .map(|axis| {
            let spec = axes.iter().find(|a| &a.name == axis).context("axis is not declared")?;
            Ok(&*spec.values)
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
