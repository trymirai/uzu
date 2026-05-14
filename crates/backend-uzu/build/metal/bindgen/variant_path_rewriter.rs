use std::collections::BTreeSet;

use anyhow::{Context, Result};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Expr;

use super::variants::VariantBind;
use crate::common::expr_rewrite::rewrite_paths_with;

pub struct VariantPathRewriter<'context> {
    variants: &'context [VariantBind],
    referenced_parameter_names: BTreeSet<String>,
    kernel_name: &'context str,
}

impl<'context> VariantPathRewriter<'context> {
    pub fn new(
        variants: &'context [VariantBind],
        kernel_name: &'context str,
    ) -> Self {
        Self {
            variants,
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
        let referenced_parameter_names = &mut self.referenced_parameter_names;
        rewrite_paths_with(&mut expression, |path| {
            if path.leading_colon.is_some()
                || path.segments.len() != 1
                || !matches!(path.segments[0].arguments, syn::PathArguments::None)
            {
                return None;
            }
            let segment_name = path.segments[0].ident.to_string();
            let bind = variants.iter().find(|variant| variant.parameter_name == segment_name)?;
            referenced_parameter_names.insert(segment_name);
            let field_name = &bind.field_name;
            Some(syn::parse_quote! { self.#field_name })
        });

        Ok(quote! { #expression })
    }

    pub fn finish(self) -> BTreeSet<String> {
        self.referenced_parameter_names
    }
}
