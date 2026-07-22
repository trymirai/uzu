//! Per-kernel key types and their generated `validate()`.
//!
//! A key names one instantiated variant of a templated kernel, and `validate()` asks the
//! only question that matters -- was this one built? -- of the list of names the build
//! actually compiled, so "the selector picked a kernel that was never built" is caught
//! where the key is formed instead of as a missing pipeline later on.

use anyhow::{Context, Result};
use igata::{
    constraint_expr::Type as AxisType,
    enum_paths::EnumPaths,
    mangling::{dynamic_mangle, field_name, snake_case, static_mangle},
    variants::{KernelSpace, KeyField, axis_rust_type},
};
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
    let built_name = format_ident!("{}_BUILT", snake_case(kernel_name).to_uppercase());

    // Sorted, so the membership test is a binary search over `&'static str`s the linker
    // deduplicates against the mangled names the wrappers were emitted under.
    let mut built = space
        .accepted_variants(enum_paths)?
        .into_iter()
        .flatten()
        .map(|variant| static_mangle(kernel_name, variant.iter().map(|(_, value)| value.as_str())))
        .collect::<Vec<_>>();
    built.sort();

    Ok(Some(KeyEmission {
        tokens: quote! {
            /// Every variant of this kernel the build compiled, sorted.
            static #built_name: &[&str] = &[#(#built),*];

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

                /// Whether this variant was one of the instantiated ones. Membership in
                /// the built set is the whole question, so no rule can be restated here
                /// and get it wrong.
                pub fn validate(&self) -> Result<(), crate::backends::metal::kernel::InvalidKernelKey> {
                    match #built_name.binary_search(&self.entry_name().as_str()) {
                        Ok(_) => Ok(()),
                        Err(_) => Err(crate::backends::metal::kernel::InvalidKernelKey {
                            kernel: #kernel_name,
                        }),
                    }
                }
            }
        },
    }))
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
