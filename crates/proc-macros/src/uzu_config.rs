use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

pub fn uzu_config(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    quote! {
        #[derive(Debug, Clone, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        #input
    }
    .into()
}
