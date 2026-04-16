use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

enum BindingKind {
    Enum,
    Struct,
    Class,
    Impl,
    Error,
}

struct ExportArguments {
    kind: BindingKind,
    name: Option<LitStr>,
}

impl Parse for ExportArguments {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let identifier: Ident = input.parse()?;
        let kind = match identifier.to_string().as_str() {
            "Enum" => BindingKind::Enum,
            "Struct" => BindingKind::Struct,
            "Class" => BindingKind::Class,
            "Impl" => BindingKind::Impl,
            "Error" => BindingKind::Error,
            other => {
                return Err(syn::Error::new(
                    identifier.span(),
                    format!("Unknown binding kind: {other}"),
                ));
            },
        };

        let name = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: Ident = input.parse()?;
            if key != "name" {
                return Err(syn::Error::new(
                    key.span(),
                    format!("Unknown parameter: {key}, expected 'name'"),
                ));
            }
            if matches!(kind, BindingKind::Impl) {
                return Err(syn::Error::new(
                    key.span(),
                    "'name' parameter is not supported for Impl",
                ));
            }
            input.parse::<Token![=]>()?;
            Some(input.parse::<LitStr>()?)
        } else {
            None
        };

        Ok(ExportArguments {
            kind,
            name,
        })
    }
}

fn uniffi_name_attribute(name: &Option<LitStr>) -> proc_macro2::TokenStream {
    match name {
        Some(name) => quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi(name = #name))]
        },
        None => quote! {},
    }
}

#[proc_macro_attribute]
pub fn export(
    arguments: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let ExportArguments {
        kind,
        name,
    } = parse_macro_input!(arguments as ExportArguments);
    let item = parse_macro_input!(item as syn::Item);
    let uniffi_name = uniffi_name_attribute(&name);

    let attributes = match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Enum))]
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(eq, from_py_object))]
            #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
            #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
            #uniffi_name
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(object))]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(get_all, from_py_object))]
            #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
            #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
            #uniffi_name
        },
        BindingKind::Class => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Object))]
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass)]
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
            #uniffi_name
        },
        BindingKind::Impl => quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pymethods)]
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        },
        BindingKind::Error => {
            let type_name = match &item {
                syn::Item::Enum(item_enum) => &item_enum.ident,
                _ => {
                    return syn::Error::new_spanned(
                        &item,
                        "Error can only be applied to enums",
                    )
                    .to_compile_error()
                    .into();
                },
            };

            return quote! {
                #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Error))]
                #uniffi_name
                #item

                #[cfg(feature = "bindings-napi")]
                impl From<#type_name> for napi::Error {
                    fn from(error: #type_name) -> Self {
                        napi::Error::from_reason(error.to_string())
                    }
                }

                #[cfg(feature = "bindings-pyo3")]
                impl From<#type_name> for pyo3::PyErr {
                    fn from(error: #type_name) -> Self {
                        pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
                    }
                }

                #[cfg(feature = "bindings-wasm")]
                impl From<#type_name> for wasm_bindgen::JsValue {
                    fn from(error: #type_name) -> Self {
                        wasm_bindgen::JsValue::from_str(&error.to_string())
                    }
                }
            }
            .into();
        },
    };

    quote! {
        #attributes
        #item
    }
    .into()
}
