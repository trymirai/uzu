use quote::quote;
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum | BindingKind::Class => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(object))]
        },
        BindingKind::Implementation => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        BindingKind::Alias => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        _ => quote! {},
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Attribute {
    match flavor {
        MethodFlavor::Plain => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi)]
        },
        MethodFlavor::Constructor => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(constructor))]
        },
        MethodFlavor::Factory => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(factory))]
        },
    }
}

pub fn error_implementation(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        impl From<#type_name> for napi::Error {
            fn from(error: #type_name) -> Self {
                napi::Error::from_reason(error.to_string())
            }
        }
    }
}
