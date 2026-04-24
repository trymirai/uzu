use quote::quote;
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(eq, from_py_object))]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(get_all, from_py_object))]
        },
        BindingKind::Class | BindingKind::ClassCloneable | BindingKind::Stream => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass)]
        },
        BindingKind::Implementation => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pymethods)]
        },
        _ => quote! {},
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Option<Attribute> {
    match flavor {
        MethodFlavor::Plain | MethodFlavor::Getter | MethodFlavor::Setter => None,
        MethodFlavor::Constructor => Some(parse_quote! {
            #[cfg_attr(feature = "bindings-pyo3", new)]
        }),
        MethodFlavor::Factory => Some(parse_quote! {
            #[cfg_attr(feature = "bindings-pyo3", staticmethod)]
        }),
    }
}

pub fn error_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-pyo3")]
        impl From<#type_name> for pyo3::PyErr {
            fn from(error: #type_name) -> Self {
                pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
            }
        }
    }
}
