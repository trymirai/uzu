use anyhow::{Context, bail, ensure};
use syn::{Attribute, Fields, ItemEnum, Meta, Path, punctuated::Punctuated, token::Comma};

/// One arm of a `#[variant_group]` enum.
///
/// A unit variant pins the first axis to the value its own name matches and leaves the
/// rest at their absent value; a struct variant contributes the product of its field
/// enums, one field per axis.
#[derive(Debug, Clone)]
pub enum VariantGroupArm {
    Unit {
        name: Box<str>,
    },
    Product {
        /// Field enum type names, one per axis, in axis order.
        field_types: Box<[Box<str>]>,
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
                        ensure!(
                            named.named.len() == axes.len(),
                            "variant `{variant_name}` has {} fields but {name} groups {} axes",
                            named.named.len(),
                            axes.len(),
                        );

                        let field_types = named
                            .named
                            .into_iter()
                            .map(|field| {
                                let syn::Type::Path(path) = &field.ty else {
                                    bail!("field `{:?}` of `{variant_name}` must be an enum type", field.ident);
                                };
                                Ok(path
                                    .path
                                    .segments
                                    .last()
                                    .context("empty field type path")?
                                    .ident
                                    .to_string()
                                    .into_boxed_str())
                            })
                            .collect::<anyhow::Result<Box<[Box<str>]>>>()?;

                        Ok(VariantGroupArm::Product {
                            field_types,
                        })
                    },
                    Fields::Unnamed(_) => bail!("variant `{variant_name}` must use named fields"),
                }
            })
            .collect::<anyhow::Result<Box<[VariantGroupArm]>>>()?;

        Ok(Self {
            name,
            axes,
            arms,
        })
    }
}
