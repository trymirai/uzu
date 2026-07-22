//! Per-kernel key types and their generated `validate()`.
//!
//! A key names one instantiated variant of a templated kernel. Its `validate()` is
//! emitted from the very expressions the build script evaluated to decide which variants
//! to compile, so "the selector picked a kernel that was never built" is reported against
//! the shader rule that excluded it rather than as a missing-pipeline failure later on.

use anyhow::{Context, Result};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Type;

use super::super::{
    ast::{MetalKernelInfo, MetalTemplateParameter, MetalTemplateParameterType},
    wrapper::{KeyField, constraint_set, key_layout},
};
use crate::common::{
    constraint_expr::RustBindings,
    enum_paths::EnumPaths,
    gpu_types::VariantGroupArm,
    mangling::{dynamic_mangle, field_name, snake_case, static_mangle},
};

pub struct KeyEmission {
    pub tokens: TokenStream,
}

pub fn build(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> Result<Option<KeyEmission>> {
    let fields = key_layout(kernel, enum_paths);
    if fields.is_empty() {
        return Ok(None);
    }

    let kernel_name = kernel.name.as_ref();
    let key_name = format_ident!("{kernel_name}Key");

    let struct_fields = fields
        .iter()
        .map(|field| match field {
            KeyField::Axis(parameter) => {
                let name = format_ident!("{}", field_name(&parameter.name));
                let ty = axis_rust_type(parameter)?;
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

    let entry_name = entry_name(kernel, &fields);
    let validate = validate(kernel, enum_paths, &fields, &group_prelude)?;
    let agreement_test = agreement_test(kernel, enum_paths, &fields, &quote! { #key_name })?;

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

fn axis_rust_type(parameter: &MetalTemplateParameter) -> Result<Type> {
    Ok(match &parameter.ty {
        MetalTemplateParameterType::Type => syn::parse_str("crate::data_type::DataType")?,
        MetalTemplateParameterType::Value(text) => syn::parse_str(text.as_ref())?,
    })
}

/// Rebuilds the mangled name from the key's fields, in template parameter order.
fn entry_name(
    kernel: &MetalKernelInfo,
    fields: &[KeyField<'_>],
) -> TokenStream {
    let parameters = kernel.variants.as_deref().unwrap_or_default();

    let values = parameters.iter().map(|parameter| {
        let name = format_ident!("{}", field_name(&parameter.name));
        // Grouped axes are locals from the prelude; everything else is a field.
        let grouped = fields.iter().any(
            |field| matches!(field, KeyField::Group { axes, .. } if axes.iter().any(|axis| axis == &parameter.name)),
        );
        let access = if grouped {
            quote! { #name }
        } else {
            quote! { self.#name }
        };

        match &parameter.ty {
            MetalTemplateParameterType::Type => quote! { #access.metal_type() },
            MetalTemplateParameterType::Value(_) => quote! { #access.to_string() },
        }
    });

    dynamic_mangle(kernel.name.as_ref(), values)
}

fn validate(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
    fields: &[KeyField<'_>],
    group_prelude: &[TokenStream],
) -> Result<TokenStream> {
    let constraints = constraint_set(kernel, enum_paths)?;

    let axes = kernel
        .variants
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|parameter| {
            let name = format_ident!("{}", field_name(&parameter.name));
            let grouped = fields.iter().any(|field| {
                matches!(field, KeyField::Group { axes, .. } if axes.iter().any(|axis| axis == &parameter.name))
            });
            let access = if grouped {
                quote! { #name }
            } else {
                quote! { self.#name }
            };
            (parameter.name.clone(), access)
        })
        .collect();

    let bindings = RustBindings {
        axes,
        helpers: enum_paths.helpers().iter().map(|(name, helper)| (name.clone(), helper.rust_method.clone())).collect(),
        enum_paths: enum_paths.rust_paths(),
    };

    let kernel_name = kernel.name.as_ref();

    // A constraint only says which combinations are legal; it cannot catch a value that
    // was never declared at all, such as a head dimension no kernel was built for.
    // Grouped axes need no check -- their sum type has no illegal value to hold.
    let declared = fields
        .iter()
        .filter_map(|field| match field {
            KeyField::Axis(parameter) => Some(parameter),
            KeyField::Group {
                ..
            } => None,
        })
        .map(|parameter| {
            let name = format_ident!("{}", field_name(&parameter.name));
            let literals = axis_literals(parameter, enum_paths)?;
            let rule = format!("{} is one of {}", parameter.name, parameter.variants.join(", "));
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
    parameter: &MetalTemplateParameter,
    enum_paths: &EnumPaths,
) -> Result<Vec<TokenStream>> {
    parameter
        .variants
        .iter()
        .map(|value| match &parameter.ty {
            MetalTemplateParameterType::Type => {
                let variant = format_ident!("{}", data_type_variant(value)?);
                Ok(quote! { crate::data_type::DataType::#variant })
            },
            MetalTemplateParameterType::Value(rust_type) => match rust_type.as_ref() {
                "bool" => {
                    let value: bool = value.parse()?;
                    Ok(quote! { #value })
                },
                "u32" => {
                    let value: u32 = value.parse()?;
                    Ok(quote! { #value })
                },
                _ => {
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
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
    fields: &[KeyField<'_>],
    key_name: &TokenStream,
) -> Result<TokenStream> {
    let mut loops = Vec::new();
    let mut initializers = Vec::new();

    for field in fields {
        let (binding, literals) = match field {
            KeyField::Axis(parameter) => {
                (format_ident!("{}", field_name(&parameter.name)), axis_literals(parameter, enum_paths)?)
            },
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

    let accepted = super::super::wrapper::accepted_variants(kernel, enum_paths)?
        .into_iter()
        .flatten()
        .map(|variant| static_mangle(kernel.name.as_ref(), variant.iter().map(|(_, value)| value.as_str())))
        .collect::<Vec<_>>();

    let test_name = format_ident!("{}_key_validate_matches_build", snake_case(kernel.name.as_ref()));
    let kernel_name = kernel.name.as_ref();

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
