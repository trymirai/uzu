use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input};

#[proc_macro_attribute]
pub fn kernel(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let file = format!("/cpu/{}.rs", args.to_string());

    let mut func = parse_macro_input!(input as ItemFn);

    func.attrs.retain(|attr| !attr.path().is_ident("variants") && !attr.path().is_ident("constraint"));

    for arg in &mut func.sig.inputs {
        if let FnArg::Typed(pat) = arg {
            pat.attrs.retain(|attr| {
                let attr_path = attr.path();

                !(attr_path.is_ident("specialize") || attr_path.is_ident("optional"))
            });
        }
    }
    quote! {
        #func
        include!(concat!(env!("OUT_DIR"), #file));
    }
    .into()
}

#[proc_macro_attribute]
pub fn __internal_uzu_test(
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

#[proc_macro_attribute]
pub fn __internal_uzu_bench(
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

#[proc_macro_attribute]
pub fn __internal_uzu_ignored(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    quote! {
        #[allow(unused)]
        #func
    }
    .into()
}
