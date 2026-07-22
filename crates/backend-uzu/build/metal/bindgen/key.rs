//! Per-kernel key types and their generated `validate()`.
//!
//! A key names one instantiated variant of a templated kernel. Its `validate()` is
//! emitted from the very expressions the build script evaluated to decide which variants
//! to compile, so "the selector picked a kernel that was never built" is reported against
//! the shader rule that excluded it rather than as a missing-pipeline failure later on.

use anyhow::{Context, Result};
use igata::{
    constraint_expr::{RustBindings, Type as AxisType},
    enum_paths::EnumPaths,
    gpu_types::VariantGroupArm,
    mangling::{dynamic_mangle, field_name, snake_case, static_mangle},
    variants::{AxisSpec, KernelSpace, KeyField},
};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Type;

use super::super::{ast::MetalKernelInfo, wrapper::KernelAxes};

pub struct KeyEmission {
    pub tokens: TokenStream,
}

pub fn build(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> Result<Option<KeyEmission>> {
    let kernel_axes = KernelAxes::of(kernel)?;
    let space = kernel_axes.space();
    let fields = space.key_layout(enum_paths);
    if fields.is_empty() {
        return Ok(None);
    }

    let kernel_name = kernel.name.as_ref();
    let key_name = format_ident!("{kernel_name}Key");

    let struct_fields = fields
        .iter()
        .map(|field| match field {
            KeyField::Axis(axis) => {
                let name = format_ident!("{}", field_name(&axis.name));
                let ty = axis_rust_type(axis, enum_paths)?;
                Ok(quote! { pub #name: #ty })
            },
            KeyField::Group {
                type_name,
                ..
            } => {
                let name = format_ident!("{}", snake_case(type_name));
                let path: Type = syn::parse_str(
                    enum_paths
                        .variant_group_path(type_name)
                        .with_context(|| format!("no Rust path for variant group `{type_name}`"))?,
                )?;
                Ok(quote! { pub #name: #path })
            },
        })
        .collect::<Result<Vec<_>>>()?;

    // Grouped axes are flattened once at the top of every method that needs them, so a
    // constraint can go on referring to the shader's axis names.
    let group_prelude = fields
        .iter()
        .filter_map(|field| match field {
            KeyField::Group {
                type_name,
                axes,
            } => {
                let field = format_ident!("{}", snake_case(type_name));
                let names = axes.iter().map(|axis| format_ident!("{}", field_name(axis)));
                Some(quote! { let (#(#names),*) = self.#field.to_template_args(); })
            },
            _ => None,
        })
        .collect::<Vec<_>>();

    let entry_name = entry_name(kernel_name, &space, &fields);
    let validate = validate(&space, enum_paths, &fields, &group_prelude)?;
    let agreement_test = agreement_test(&space, enum_paths, &fields, &quote! { #key_name })?;

    Ok(Some(KeyEmission {
        tokens: quote! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            #[allow(dead_code)]
            pub struct #key_name {
                #(#struct_fields,)*
            }

            #[allow(clippy::style, clippy::complexity, clippy::perf, unused_variables, dead_code)]
            impl #key_name {
                /// The mangled entry point name of the instantiated kernel.
                pub fn entry_name(&self) -> String {
                    #(#group_prelude)*
                    #entry_name
                }

                #validate
            }

            #agreement_test
        },
    }))
}

/// The Rust type of a key field holding this axis's value.
fn axis_rust_type(
    axis: &AxisSpec,
    enum_paths: &EnumPaths,
) -> Result<Type> {
    Ok(match &axis.ty {
        AxisType::DType => syn::parse_str("crate::data_type::DataType")?,
        AxisType::Bool => syn::parse_str("bool")?,
        AxisType::Int => syn::parse_str("u32")?,
        AxisType::Enum(name) => {
            syn::parse_str(enum_paths.full_path_for(name).with_context(|| format!("no Rust path for enum `{name}`"))?)?
        },
    })
}

/// Whether this axis is one the key holds flattened out of a variant group.
fn is_grouped(
    fields: &[KeyField<'_>],
    axis: &str,
) -> bool {
    fields.iter().any(|field| matches!(field, KeyField::Group { axes, .. } if axes.iter().any(|a| a.as_ref() == axis)))
}

/// The expression holding an axis's value: a local from the group prelude if it was
/// flattened out of a variant group, a field of the key otherwise.
fn axis_access(
    fields: &[KeyField<'_>],
    axis: &str,
) -> TokenStream {
    let name = format_ident!("{}", field_name(axis));
    if is_grouped(fields, axis) {
        quote! { #name }
    } else {
        quote! { self.#name }
    }
}

/// Rebuilds the mangled name from the key's fields, in template parameter order.
fn entry_name(
    kernel_name: &str,
    space: &KernelSpace<'_>,
    fields: &[KeyField<'_>],
) -> TokenStream {
    let values = space.axes.unwrap_or_default().iter().map(|axis| {
        let access = axis_access(fields, &axis.name);
        match axis.ty {
            AxisType::DType => quote! { #access.metal_type() },
            _ => quote! { #access.to_string() },
        }
    });

    dynamic_mangle(kernel_name, values)
}

fn validate(
    space: &KernelSpace<'_>,
    enum_paths: &EnumPaths,
    fields: &[KeyField<'_>],
    group_prelude: &[TokenStream],
) -> Result<TokenStream> {
    let constraints = space.constraint_set(enum_paths)?;

    let axes = space
        .axes
        .unwrap_or_default()
        .iter()
        .map(|axis| (axis.name.clone(), axis_access(fields, &axis.name)))
        .collect();

    let bindings = RustBindings {
        axes,
        helpers: enum_paths.helpers().iter().map(|(name, helper)| (name.clone(), helper.rust_method.clone())).collect(),
        enum_paths: enum_paths.rust_paths(),
    };

    let kernel_name = space.name;

    // A constraint only says which combinations are legal; it cannot catch a value that
    // was never declared at all, such as a head dimension no kernel was built for.
    // Grouped axes need no check -- their sum type has no illegal value to hold.
    let declared = fields
        .iter()
        .filter_map(|field| match field {
            KeyField::Axis(axis) => Some(axis),
            KeyField::Group {
                ..
            } => None,
        })
        .map(|axis| {
            let name = format_ident!("{}", field_name(&axis.name));
            let literals = axis_literals(axis, enum_paths)?;
            let rule = format!("{} is one of {}", axis.name, axis.values.join(", "));
            Ok(quote! {
                if ![#(#literals),*].contains(&self.#name) {
                    return Err(crate::backends::metal::kernel::InvalidKernelKey {
                        kernel: #kernel_name,
                        rule: #rule,
                    });
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let checks = constraints.to_rust(&bindings)?.into_iter().map(|(rule, predicate)| {
        let rule = rule.as_ref();
        quote! {
            if !(#predicate) {
                return Err(crate::backends::metal::kernel::InvalidKernelKey {
                    kernel: #kernel_name,
                    rule: #rule,
                });
            }
        }
    });

    Ok(quote! {
        /// Whether this variant was actually instantiated. Generated from the shader's
        /// own CONSTRAINT expressions, so it cannot drift from what the build compiled.
        pub fn validate(&self) -> Result<(), crate::backends::metal::kernel::InvalidKernelKey> {
            #(#group_prelude)*
            #(#declared)*
            #(#checks)*
            Ok(())
        }
    })
}

/// Rust literals for every value an axis can take, in declared order.
fn axis_literals(
    axis: &AxisSpec,
    enum_paths: &EnumPaths,
) -> Result<Vec<TokenStream>> {
    axis.values
        .iter()
        .map(|value| match &axis.ty {
            AxisType::DType => {
                let variant = format_ident!("{}", data_type_variant(value)?);
                Ok(quote! { crate::data_type::DataType::#variant })
            },
            AxisType::Bool => {
                let value: bool = value.parse()?;
                Ok(quote! { #value })
            },
            AxisType::Int => {
                let value: u32 = value.parse()?;
                Ok(quote! { #value })
            },
            AxisType::Enum(_) => {
                let (enum_name, variant) =
                    value.rsplit_once("::").with_context(|| format!("`{value}` is not an enum variant"))?;
                let path: Type = syn::parse_str(
                    enum_paths
                        .full_path_for(enum_name)
                        .with_context(|| format!("no Rust path for enum `{enum_name}`"))?,
                )?;
                let variant = format_ident!("{variant}");
                Ok(quote! { #path::#variant })
            },
        })
        .collect()
}

/// Rust literals for every value a variant group admits, reconstructed from its arms.
fn group_literals(
    type_name: &str,
    enum_paths: &EnumPaths,
) -> Result<Vec<TokenStream>> {
    let group = enum_paths
        .variant_groups()
        .iter()
        .find(|group| group.name.as_ref() == type_name)
        .with_context(|| format!("no variant group named `{type_name}`"))?;

    let path: Type = syn::parse_str(
        enum_paths.variant_group_path(type_name).with_context(|| format!("no Rust path for `{type_name}`"))?,
    )?;

    let mut literals = Vec::new();
    for arm in group.arms.iter() {
        match arm {
            VariantGroupArm::Unit {
                name,
            } => {
                let name = format_ident!("{name}");
                literals.push(quote! { #path::#name });
            },
            VariantGroupArm::Product {
                name,
                fields,
            } => {
                let arm_name = format_ident!("{name}");
                let per_field = fields
                    .iter()
                    .map(|(field, field_type)| {
                        let field_path: Type = syn::parse_str(
                            enum_paths
                                .full_path_for(field_type)
                                .with_context(|| format!("no Rust path for enum `{field_type}`"))?,
                        )?;
                        let field = format_ident!("{field}");
                        Ok(enum_paths
                            .variants_for(field_type)
                            .with_context(|| format!("`{field_type}` is not an enum gpu type"))?
                            .iter()
                            .map(|(member, _)| {
                                let member = format_ident!("{member}");
                                quote! { #field: #field_path::#member }
                            })
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>>>()?;

                for combination in per_field.into_iter().multi_cartesian_product() {
                    literals.push(quote! { #path::#arm_name { #(#combination),* } });
                }
            },
        }
    }

    Ok(literals)
}

fn data_type_variant(metal_type: &str) -> Result<&'static str> {
    Ok(match metal_type {
        "float" => "F32",
        "half" => "F16",
        "bfloat" => "BF16",
        other => anyhow::bail!("no DataType corresponds to the Metal type `{other}`"),
    })
}

/// A test asserting the generated `validate()` accepts exactly the variants the build
/// compiled. The interpreter and the emitter walk the same tree, and this pins that
/// down end to end -- including that `entry_name()` reproduces the mangling the wrapper
/// emitted, which is easy to get wrong when a variant group reorders axes.
fn agreement_test(
    space: &KernelSpace<'_>,
    enum_paths: &EnumPaths,
    fields: &[KeyField<'_>],
    key_name: &TokenStream,
) -> Result<TokenStream> {
    let mut loops = Vec::new();
    let mut initializers = Vec::new();

    for field in fields {
        let (binding, literals) = match field {
            KeyField::Axis(axis) => (format_ident!("{}", field_name(&axis.name)), axis_literals(axis, enum_paths)?),
            KeyField::Group {
                type_name,
                ..
            } => (format_ident!("{}", snake_case(type_name)), group_literals(type_name, enum_paths)?),
        };
        initializers.push(binding.clone());
        loops.push((binding, literals));
    }

    let mut body = quote! { declared.push(#key_name { #(#initializers),* }); };
    for (binding, literals) in loops.into_iter().rev() {
        body = quote! {
            for #binding in [#(#literals),*] {
                #body
            }
        };
    }

    let accepted = space
        .accepted_variants(enum_paths)?
        .into_iter()
        .flatten()
        .map(|variant| static_mangle(space.name, variant.iter().map(|(_, value)| value.as_str())))
        .collect::<Vec<_>>();

    let test_name = format_ident!("{}_key_validate_matches_build", snake_case(space.name));
    let kernel_name = space.name;

    Ok(quote! {
        #[cfg(test)]
        #[proc_macros::uzu_test]
        fn #test_name() {
            let mut declared = Vec::new();
            #body

            let mut accepted: Vec<String> =
                declared.iter().filter(|key| key.validate().is_ok()).map(|key| key.entry_name()).collect();
            accepted.sort();

            let mut built: Vec<String> = [#(#accepted),*].iter().map(|name| name.to_string()).collect();
            built.sort();

            assert_eq!(
                accepted, built,
                concat!(
                    "the generated validate() for ", #kernel_name,
                    " does not accept exactly the variants the build compiled",
                ),
            );
        }
    })
}
