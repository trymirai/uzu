use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(Display)]
pub fn display_derive(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        ..
    } = parse_macro_input!(input as DeriveInput);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics ::core::fmt::Display for #ident #ty_generics #where_clause {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(self, f)
            }
        }
    }
    .into()
}
