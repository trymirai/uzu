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

pub fn factory(
    self_type: &syn::Type,
    method: &syn::ImplItemFn,
) -> proc_macro2::TokenStream {
    let method_ident = &method.sig.ident;
    let method_name = method_ident.to_string();
    let wrapper_ident = quote::format_ident!("{}_bindings_pyo3", method_ident);
    let inputs = &method.sig.inputs;
    let output = &method.sig.output;
    let asyncness = &method.sig.asyncness;
    let arg_idents = factory_arg_idents(inputs);
    let forward_await = asyncness.as_ref().map(|_| quote! { .await });

    quote! {
        #[cfg(feature = "bindings-pyo3")]
        #[pyo3::pymethods]
        impl #self_type {
            #[staticmethod]
            #[pyo3(name = #method_name)]
            pub #asyncness fn #wrapper_ident( #inputs ) #output {
                <#self_type>::#method_ident( #( #arg_idents ),* )#forward_await
            }
        }
    }
}

fn factory_arg_idents(inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>) -> Vec<syn::Ident> {
    inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                _ => None,
            },
            syn::FnArg::Receiver(_) => None,
        })
        .collect()
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
