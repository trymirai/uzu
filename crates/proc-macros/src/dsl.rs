use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input};

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
