use std::collections::BTreeSet;

use anyhow::{Context, Result};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Expr;

use super::variants::VariantBind;
use crate::common::expr_rewrite::rewrite_paths_with;

pub struct HostExpressionRewriter<'context> {
    variants: &'context [VariantBind],
    specialization_names: Vec<String>,
    referenced_parameter_names: BTreeSet<String>,
    kernel_name: &'context str,
}

impl<'context> HostExpressionRewriter<'context> {
    pub fn new(
        variants: &'context [VariantBind],
        specialization_names: Vec<String>,
        kernel_name: &'context str,
    ) -> Self {
        Self {
            variants,
            specialization_names,
            referenced_parameter_names: BTreeSet::new(),
            kernel_name,
        }
    }

    pub fn rewrite(
        &mut self,
        expression_text: &str,
    ) -> Result<TokenStream> {
        let mut expression: Expr = syn::parse_str(expression_text).with_context(|| {
            format!("rust expression `{}` in kernel `{}` cannot be parsed", expression_text, self.kernel_name)
        })?;

        let variants = self.variants;
        let specialization_names = &self.specialization_names;
        let referenced_parameter_names = &mut self.referenced_parameter_names;
        rewrite_paths_with(&mut expression, |path| {
            if path.leading_colon.is_some()
                || path.segments.len() != 1
                || !matches!(path.segments[0].arguments, syn::PathArguments::None)
            {
                return None;
            }

            let name = path.segments[0].ident.to_string();
            let field_name = if let Some(variant) = variants.iter().find(|variant| variant.parameter_name == name) {
                variant.field_name.clone()
            } else if specialization_names.contains(&name) {
                format_ident!("specialize_{name}")
            } else {
                return None;
            };
            referenced_parameter_names.insert(name);
            Some(syn::parse_quote! { self.#field_name })
        });

        Ok(quote! { #expression })
    }

    pub fn finish(self) -> BTreeSet<String> {
        self.referenced_parameter_names
    }
}
