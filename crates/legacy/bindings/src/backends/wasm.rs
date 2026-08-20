use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    backends::Backend,
    contexts::{ClassContext, EnumerationContext, ErrorContext, ImplementationContext, StructureContext},
    types::StructureFlavor,
};

pub struct Wasm;

impl Backend for Wasm {
    fn structure_attributes(context: &StructureContext) -> TokenStream {
        match context.flavor {
            StructureFlavor::Plain => quote! {
                #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
                #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
            },
            StructureFlavor::Class => quote! {
                #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
            },
        }
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        }
    }

    fn enumeration_attributes(_context: &EnumerationContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
            #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
        }
    }

    fn implementation_attributes(_context: &ImplementationContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        }
    }

    fn error_companions(context: &ErrorContext) -> TokenStream {
        let type_name = &context.item.ident;
        quote! {
            #[cfg(feature = "bindings-wasm")]
            impl From<#type_name> for wasm_bindgen::JsValue {
                fn from(error: #type_name) -> Self {
                    wasm_bindgen::JsValue::from_str(&error.to_string())
                }
            }
        }
    }
}
