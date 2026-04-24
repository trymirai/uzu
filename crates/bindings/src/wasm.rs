use quote::quote;
use syn::Ident;

use crate::BindingKind;

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum | BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
            #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
        },
        BindingKind::Class | BindingKind::ClassCloneable | BindingKind::Stream | BindingKind::Implementation => {
            quote! {
                #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
            }
        },
        _ => quote! {},
    }
}

pub fn error_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-wasm")]
        impl From<#type_name> for wasm_bindgen::JsValue {
            fn from(error: #type_name) -> Self {
                wasm_bindgen::JsValue::from_str(&error.to_string())
            }
        }
    }
}
