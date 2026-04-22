use quote::quote;
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(string_enum))]
        },
        BindingKind::Class | BindingKind::ClassCloneable => quote! {
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
        MethodFlavor::Getter => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(getter))]
        },
        MethodFlavor::Setter => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(setter))]
        },
    }
}

pub fn struct_value_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        const _: () = {
            use napi::bindgen_prelude::ToNapiValue;
            use napi::sys::{napi_env, napi_value};

            impl ToNapiValue for &#type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl ToNapiValue for &mut #type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }
        };
    }
}

pub fn class_value_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        const _: () = {
            use napi::bindgen_prelude::{FromNapiValue, ToNapiValue};
            use napi::sys::{napi_env, napi_value};

            impl ToNapiValue for &#type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl ToNapiValue for &mut #type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl FromNapiValue for #type_name {
                unsafe fn from_napi_value(env: napi_env, value: napi_value) -> napi::Result<Self> {
                    let reference = <&#type_name as FromNapiValue>::from_napi_value(env, value)?;
                    Ok(::core::clone::Clone::clone(reference))
                }
            }
        };
    }
}

pub fn error_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        impl From<#type_name> for napi::Error {
            fn from(error: #type_name) -> Self {
                napi::Error::from_reason(error.to_string())
            }
        }

        #[cfg(feature = "bindings-napi")]
        impl From<#type_name> for napi::JsError {
            fn from(error: #type_name) -> Self {
                let napi_error: napi::Error = error.into();
                napi::JsError::from(napi_error)
            }
        }
    }
}
