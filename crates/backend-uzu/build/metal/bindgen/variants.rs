use std::collections::BTreeSet;

use anyhow::Result;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Type};

use super::super::ast::{MetalKernelInfo, MetalTemplateParameterType};

pub struct VariantBind {
    pub parameter_name: Ident,
    pub field_name: Ident,
    parsed_type: Option<Type>,
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
            None => quote! { #[allow(non_snake_case)] #parameter_name: crate::DataType },
            Some(parsed_type) => {
                quote! { #[allow(non_snake_case)] #parameter_name: #parsed_type }
            },
        }
    }

    pub fn kernel_format(&self) -> TokenStream {
        let parameter_name = &self.parameter_name;
        match &self.parsed_type {
            None => quote! { #parameter_name.metal_type() },
            Some(_) => quote! { #parameter_name.to_string() },
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
