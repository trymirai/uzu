use anyhow::{Context, bail, ensure};
use itertools::Itertools;
use syn::{Attribute, Fields, ItemEnum, Meta, Path, punctuated::Punctuated, token::Comma};

use crate::mangling::field_name;

/// One arm of a `#[variant_group]` enum.
///
/// A unit variant takes each axis's leftover value; a struct variant contributes the
/// product of its field enums, one field per axis, matched to the axis by name.
#[derive(Debug, Clone)]
pub enum VariantGroupArm {
    Unit {
        name: Box<str>,
    },
    Product {
        name: Box<str>,
        /// Field names and enum type names, one per axis, in axis order.
        fields: Box<[(Box<str>, Box<str>)]>,
    },
}

/// An enum marked `#[variant_group(AXIS, ...)]`: the Rust sum type whose shape defines
/// which combinations of a group of shader template axes legally exist.
#[derive(Debug, Clone)]
pub struct GpuTypeVariantGroup {
    pub name: Box<str>,
    pub axes: Box<[Box<str>]>,
    pub arms: Box<[VariantGroupArm]>,
}

impl GpuTypeVariantGroup {
    /// The axis names in `#[variant_group(...)]`, or `None` if the attribute is absent.
    pub fn axes_of(attrs: &[Attribute]) -> anyhow::Result<Option<Box<[Box<str>]>>> {
        let Some(attribute) =
            attrs.iter().find(|attr| attr.path().segments.last().is_some_and(|s| s.ident == "variant_group"))
        else {
            return Ok(None);
        };

        let Meta::List(list) = &attribute.meta else {
            bail!("variant_group must list the axes it groups, e.g. #[variant_group(A, B)]");
        };

        let axes = list
            .parse_args_with(Punctuated::<Path, Comma>::parse_terminated)
            .context("cannot parse variant_group axes")?
            .into_iter()
            .map(|path| {
                Ok(path.get_ident().context("variant_group axes must be plain names")?.to_string().into_boxed_str())
            })
            .collect::<anyhow::Result<Box<[Box<str>]>>>()?;

        ensure!(!axes.is_empty(), "variant_group must name at least one axis");

        Ok(Some(axes))
    }

    pub fn parse(
        item: ItemEnum,
        axes: Box<[Box<str>]>,
    ) -> anyhow::Result<Self> {
        let name: Box<str> = item.ident.to_string().into();

        let arms = item
            .variants
            .into_iter()
            .map(|variant| {
                let variant_name = variant.ident.to_string();
                match variant.fields {
                    Fields::Unit => Ok(VariantGroupArm::Unit {
                        name: variant_name.into(),
                    }),
                    Fields::Named(named) => {
                        let mut declared = named
                            .named
                            .into_iter()
                            .map(|field| {
                                let syn::Type::Path(path) = &field.ty else {
                                    bail!("field `{:?}` of `{variant_name}` must be an enum type", field.ident);
                                };
                                let field_name = field.ident.context("variant group fields must be named")?.to_string();
                                let field_type =
                                    path.path.segments.last().context("empty field type path")?.ident.to_string();
                                Ok((field_name.into_boxed_str(), field_type.into_boxed_str()))
                            })
                            .collect::<anyhow::Result<Vec<(Box<str>, Box<str>)>>>()?;

                        // Fields are matched to axes by name, not by position, so
                        // reordering the struct cannot silently rebind the axes.
                        let all_fields = declared.iter().map(|(field, _)| field.as_ref()).join(", ");
                        let fields = axes
                            .iter()
                            .map(|axis| {
                                let expected = field_name(axis);
                                let index =
                                    declared.iter().position(|(field, _)| **field == *expected).with_context(|| {
                                        format!(
                                            "variant `{variant_name}` of `{name}` has no field `{expected}` for axis \
                                             `{axis}` (its fields are: {all_fields})"
                                        )
                                    })?;
                                Ok(declared.remove(index))
                            })
                            .collect::<anyhow::Result<Box<[(Box<str>, Box<str>)]>>>()?;

                        ensure!(
                            declared.is_empty(),
                            "variant `{variant_name}` of `{name}` has field(s) {} naming no axis (it groups: {})",
                            declared.iter().map(|(field, _)| format!("`{field}`")).join(", "),
                            axes.iter().map(|axis| axis.as_ref()).join(", "),
                        );

                        Ok(VariantGroupArm::Product {
                            name: variant_name.into(),
                            fields,
                        })
                    },
                    Fields::Unnamed(_) => bail!("variant `{variant_name}` must use named fields"),
                }
            })
            .collect::<anyhow::Result<Box<[VariantGroupArm]>>>()?;

        // A unit arm means "whatever the struct arms leave over", which is only
        // well defined for one of them.
        let mut units = arms.iter().filter_map(|arm| match arm {
            VariantGroupArm::Unit {
                name,
            } => Some(name),
            VariantGroupArm::Product {
                ..
            } => None,
        });
        if let (Some(first), Some(second)) = (units.next(), units.next()) {
            bail!("`{name}` has more than one unit variant (`{first}` and `{second}`)");
        }

        Ok(Self {
            name,
            axes,
            arms,
        })
    }
}
