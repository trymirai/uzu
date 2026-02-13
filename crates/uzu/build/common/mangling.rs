use std::iter::repeat_n;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

pub fn static_mangle(
    function_name: impl AsRef<str>,
    variant: impl IntoIterator<Item = impl AsRef<str>>,
) -> String {
    format!(
        "_D{}{}{}",
        &function_name.as_ref().len().to_string(),
        function_name.as_ref(),
        variant.into_iter().map(|v| format!("S{}V{}", v.as_ref().len(), v.as_ref().replace('-', "n"))).join("")
    )
}

pub fn dynamic_mangle(
    function_name: impl AsRef<str>,
    variant: impl IntoIterator<Item = TokenStream>,
) -> TokenStream {
    let variant = variant.into_iter().collect::<Vec<TokenStream>>();

    let format_string = format!(
        "_D{}{}{}",
        &function_name.as_ref().len().to_string(),
        function_name.as_ref(),
        repeat_n("S{}V{}", variant.len()).join("")
    );

    quote! { format!(#format_string #(, #variant.len(), #variant.replace('-', "n"))*) }
}
