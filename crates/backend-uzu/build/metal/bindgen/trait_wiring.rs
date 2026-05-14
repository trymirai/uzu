use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::super::ast::MetalKernelInfo;

pub struct TraitWiring {
    pub trait_implementation_for: TokenStream,
    pub associate_backend: TokenStream,
    pub associated_type: Option<TokenStream>,
    pub method_visibility: TokenStream,
}

pub fn build(
    kernel: &MetalKernelInfo,
    trait_name: &Ident,
    struct_name: &Ident,
) -> TraitWiring {
    if kernel.public {
        TraitWiring {
            trait_implementation_for: quote! { crate::backends::common::kernel::#trait_name for },
            associate_backend: quote! { type Backend = crate::backends::metal::Metal; },
            associated_type: Some(quote! { type #trait_name = #struct_name; }),
            method_visibility: quote! {},
        }
    } else {
        TraitWiring {
            trait_implementation_for: quote! {},
            associate_backend: quote! {},
            associated_type: None,
            method_visibility: quote! { pub(crate) },
        }
    }
}
