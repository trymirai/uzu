use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

pub fn uzu_test(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    quote! {
        #[test]
        #func
    }
    .into()
}

pub fn uzu_bench(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    quote! {
        #[::criterion_macro::criterion]
        #func
    }
    .into()
}
