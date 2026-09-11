use anyhow::Context;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{
    Attribute, Expr, Ident, Token, Visibility, braced,
    parse::{Parse, ParseStream},
};

#[derive(Debug)]
pub struct GpuTypeOptionSetVariant {
    pub name: String,
    /// Raw expression (e.g. `1 << 0`) emitted verbatim into C++.
    pub value_expression: String,
}

#[derive(Debug)]
pub struct GpuTypeOptionSet {
    pub name: String,
    pub underlying_type: String,
    pub variants: Vec<GpuTypeOptionSetVariant>,
}

impl GpuTypeOptionSet {
    /// Parse the token-stream inside a `bitflags! { … }` macro invocation.
    pub fn parse(tokens: TokenStream) -> anyhow::Result<Self> {
        let block: BitflagsBlock = syn::parse2(tokens).context("Cannot parse bitflags! contents")?;
        Ok(Self {
            name: block.name.to_string(),
            underlying_type: block.underlying.to_string(),
            variants: block
                .variants
                .into_iter()
                .map(|(name, expression)| GpuTypeOptionSetVariant {
                    name: name.to_string(),
                    value_expression: expression.into_token_stream().to_string(),
                })
                .collect(),
        })
    }
}

struct BitflagsBlock {
    name: Ident,
    underlying: Ident,
    variants: Vec<(Ident, Expr)>,
}

impl Parse for BitflagsBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _attrs = input.call(Attribute::parse_outer)?;
        let _vis: Visibility = input.parse()?;
        let _struct_kw: Token![struct] = input.parse()?;
        let name: Ident = input.parse()?;
        let _colon: Token![:] = input.parse()?;
        let underlying: Ident = input.parse()?;

        let body;
        braced!(body in input);

        let mut variants = Vec::new();
        while !body.is_empty() {
            let _const_kw: Token![const] = body.parse()?;
            let variant_name: Ident = body.parse()?;
            let _equals: Token![=] = body.parse()?;
            let expression: Expr = body.parse()?;
            let _semi: Token![;] = body.parse()?;
            variants.push((variant_name, expression));
        }

        Ok(Self {
            name,
            underlying,
            variants,
        })
    }
}
