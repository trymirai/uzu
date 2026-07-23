use std::collections::BTreeSet;

use anyhow::Result;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Type};

use super::super::{
    ast::{MetalKernelInfo, MetalTemplateParameterType},
    enum_path_rewrite::rewrite_for_rust,
    wrapper::KernelVariant,
};
use crate::common::enum_paths::EnumPaths;

pub struct VariantBind {
    pub parameter_name: Ident,
    pub field_name: Ident,
    parsed_type: Option<Type>,
}

pub struct RequestEmission {
    pub tokens: TokenStream,
    pub initializer: TokenStream,
}

pub fn parse(kernel: &MetalKernelInfo) -> Result<Vec<VariantBind>> {
    kernel
        .variants
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|variant| {
            let parameter_name = Ident::new(variant.name.as_ref(), Span::call_site());
            let field_name = format_ident!("{}", variant.name.as_ref().to_ascii_lowercase());
            let parsed_type = match &variant.ty {
                MetalTemplateParameterType::Type => None,
                MetalTemplateParameterType::Value(type_text) => Some(syn::parse_str::<Type>(type_text.as_ref())?),
            };
            Ok(VariantBind {
                parameter_name,
                field_name,
                parsed_type,
            })
        })
        .collect()
}

impl VariantBind {
    pub fn constructor_argument(&self) -> TokenStream {
        let parameter_name = &self.parameter_name;
        match &self.parsed_type {
            None => quote! { #[allow(non_snake_case)] #parameter_name: crate::data_type::DataType },
            Some(parsed_type) => {
                quote! { #[allow(non_snake_case)] #parameter_name: #parsed_type }
            },
        }
    }

    pub fn struct_field(
        &self,
        referenced_parameter_names: &BTreeSet<String>,
    ) -> Option<TokenStream> {
        let parsed_type = self.parsed_type.as_ref()?;
        if !referenced_parameter_names.contains(&self.parameter_name.to_string()) {
            return None;
        }
        let field_name = &self.field_name;
        Some(quote! { #field_name: #parsed_type })
    }

    pub fn struct_initializer(
        &self,
        referenced_parameter_names: &BTreeSet<String>,
    ) -> Option<TokenStream> {
        self.parsed_type.as_ref()?;
        if !referenced_parameter_names.contains(&self.parameter_name.to_string()) {
            return None;
        }
        let field_name = &self.field_name;
        let parameter_name = &self.parameter_name;
        Some(quote! { #field_name: #parameter_name })
    }
}

pub fn request(
    kernel: &MetalKernelInfo,
    binds: &[VariantBind],
    variants: &[KernelVariant],
    enum_paths: &EnumPaths,
) -> Result<Option<RequestEmission>> {
    if binds.is_empty() {
        return Ok(None);
    }

    let kernel_name = kernel.name.as_ref();
    let request_name = format_ident!("{kernel_name}Request");
    let fields = binds.iter().map(|bind| {
        let name = &bind.field_name;
        let ty = bind.parsed_type.as_ref().map_or_else(|| quote! { crate::data_type::DataType }, |ty| quote! { #ty });
        quote! { pub(crate) #name: #ty }
    });
    let field_names = binds.iter().map(|bind| &bind.field_name).collect::<Vec<_>>();
    let parameter_names = binds.iter().map(|bind| &bind.parameter_name).collect::<Vec<_>>();

    let arms = variants
        .iter()
        .map(|variant| {
            let patterns = binds
                .iter()
                .zip(&variant.bindings)
                .map(|(bind, (name, value))| {
                    if bind.parameter_name != name {
                        anyhow::bail!(
                            "kernel `{}` variant parameter mismatch: expected `{}`, got `{name}`",
                            kernel.name,
                            bind.parameter_name,
                        );
                    }
                    bind.rust_value(value, enum_paths)
                })
                .collect::<Result<Vec<_>>>()?;
            let entry_name = &variant.entry_name;
            Ok(quote! { (#(#patterns,)*) => Ok(#entry_name) })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(RequestEmission {
        tokens: quote! {
            #[derive(Debug)]
            pub(crate) struct #request_name {
                #(#fields,)*
            }

            impl #request_name {
                pub(crate) fn resolve(&self) -> Result<&'static str, MetalError> {
                    match (#(self.#field_names,)*) {
                        #(#arms,)*
                        _ => Err(MetalError::UnsupportedKernelVariant {
                            kernel: #kernel_name,
                            request: format!("{self:?}"),
                        }),
                    }
                }
            }
        },
        initializer: quote! { #request_name { #(#field_names: #parameter_names,)* } },
    }))
}

impl VariantBind {
    fn rust_value(
        &self,
        value: &str,
        enum_paths: &EnumPaths,
    ) -> Result<TokenStream> {
        if self.parsed_type.is_some() {
            rewrite_for_rust(enum_paths, value)
        } else {
            let variant = match value {
                "bfloat" => quote! { crate::data_type::DataType::BF16 },
                "half" => quote! { crate::data_type::DataType::F16 },
                "float" => quote! { crate::data_type::DataType::F32 },
                _ => anyhow::bail!("unsupported Metal data type variant `{value}`"),
            };
            Ok(variant)
        }
    }
}
