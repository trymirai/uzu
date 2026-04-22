use quote::quote;
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Enum))]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        },
        BindingKind::Class => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Object))]
        },
        BindingKind::ClassCloneable => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        },
        BindingKind::Implementation => quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
        },
        _ => quote! {},
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Option<Attribute> {
    match flavor {
        MethodFlavor::Plain | MethodFlavor::Getter | MethodFlavor::Setter => None,
        MethodFlavor::Constructor | MethodFlavor::Factory => Some(parse_quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::constructor)]
        }),
    }
}

pub fn error_attribute() -> proc_macro2::TokenStream {
    quote! {
        #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Error))]
    }
}

#[allow(dead_code)]
pub fn error_implementations(_type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {}
}
