#![cfg(all(feature = "metal", target_os = "macos"))]

use std::iter::repeat_n;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

/// `GEMM_TILING` -> `gemm_tiling`, `AT` -> `at`: the field holding an axis's value.
pub fn field_name(axis: &str) -> String {
    axis.to_ascii_lowercase()
}

/// `WeightsKey` -> `weights_key`.
pub fn snake_case(type_name: &str) -> String {
    let mut out = String::new();
    for (index, character) in type_name.char_indices() {
        if character.is_uppercase() && index != 0 {
            out.push('_');
        }
        out.extend(character.to_lowercase());
    }
    out
}

pub fn unqualify_variant(value: &str) -> &str {
    value.rsplit("::").next().unwrap_or(value)
}

pub fn static_mangle(
    function_name: impl AsRef<str>,
    variant: impl IntoIterator<Item = impl AsRef<str>>,
) -> String {
    format!(
        "_D{}{}{}",
        function_name.as_ref().len(),
        function_name.as_ref(),
        variant
            .into_iter()
            .map(|v| {
                let v = unqualify_variant(v.as_ref()).replace('-', "n");
                format!("S{}V{}", v.len(), v)
            })
            .join("")
    )
}

pub fn dynamic_mangle(
    function_name: impl AsRef<str>,
    variant: impl IntoIterator<Item = TokenStream>,
) -> TokenStream {
    let variant = variant.into_iter().collect::<Vec<TokenStream>>();

    let format_string = format!(
        "_D{}{}{}",
        function_name.as_ref().len(),
        function_name.as_ref(),
        repeat_n("S{}V{}", variant.len()).join("")
    );

    quote! { format!(#format_string #(, #variant.len(), #variant.replace('-', "n"))*) }
}
