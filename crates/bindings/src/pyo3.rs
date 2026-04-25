use quote::quote;
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(eq, from_py_object))]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(get_all, from_py_object))]
        },
        BindingKind::Class | BindingKind::Stream => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(from_py_object))]
        },
        BindingKind::ClassCloneable => quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(get_all, from_py_object))]
        },
        BindingKind::Implementation => quote! {},
        _ => quote! {},
    }
}

pub fn enum_stub_gen_attribute(is_data_variant: bool) -> proc_macro2::TokenStream {
    if is_data_variant {
        quote! { #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass_complex_enum)] }
    } else {
        quote! { #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass_enum)] }
    }
}

pub fn implementation_expansion(
    self_type: &syn::Type,
    methods: &[(syn::ImplItemFn, MethodFlavor, bool)],
) -> proc_macro2::TokenStream {
    let mut wrappers: Vec<proc_macro2::TokenStream> = methods
        .iter()
        .filter(|(_, flavor, _)| !matches!(flavor, MethodFlavor::Factory))
        .map(|(method, flavor, _)| build_method_wrapper(method, flavor))
        .collect();

    // Stream protocol: __aiter__, __anext__, iterator() — emitted when any method is StreamNext.
    if let Some((method, _, _)) = methods.iter().find(|(_, _, is_stream_next)| *is_stream_next) {
        wrappers.push(build_stream_protocol(method));
    }

    if wrappers.is_empty() {
        return quote! {};
    }

    quote! {
        #[cfg(feature = "bindings-pyo3")]
        const _: () = {
            #[allow(unused_imports)]
            use ::pyo3::prelude::*;
            #[pyo3_stub_gen::derive::gen_stub_pymethods]
            #[pyo3::pymethods]
            impl #self_type {
                #( #wrappers )*
            }
        };
    }
}

fn build_stream_protocol(next_method: &syn::ImplItemFn) -> proc_macro2::TokenStream {
    let next_ident = &next_method.sig.ident;
    let item_repr = next_item_type_repr(&next_method.sig.output).unwrap_or_else(|| "typing.Any".to_string());
    let anext_repr = format!("collections.abc.Awaitable[{item_repr}]");

    quote! {
        pub fn __aiter__(slf: ::pyo3::PyRef<'_, Self>) -> ::pyo3::PyRef<'_, Self> {
            slf
        }

        #[gen_stub(override_return_type(type_repr = #anext_repr, imports = ("collections.abc")))]
        pub fn __anext__<'py>(
            &self,
            py: ::pyo3::Python<'py>,
        ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
            let __this = ::std::clone::Clone::clone(self);
            ::pyo3_async_runtimes::tokio::future_into_py(py, async move {
                match __this.#next_ident().await {
                    ::std::option::Option::Some(value) => ::std::result::Result::Ok(value),
                    ::std::option::Option::None => ::std::result::Result::Err(
                        ::pyo3::exceptions::PyStopAsyncIteration::new_err("end of stream"),
                    ),
                }
            })
        }

        pub fn iterator(slf: ::pyo3::PyRef<'_, Self>) -> ::pyo3::PyRef<'_, Self> {
            slf
        }
    }
}

fn next_item_type_repr(output: &syn::ReturnType) -> Option<String> {
    let ty = match output {
        syn::ReturnType::Default => return None,
        syn::ReturnType::Type(_, ty) => ty.as_ref(),
    };
    let path = match ty {
        syn::Type::Path(type_path) => &type_path.path,
        _ => return None,
    };
    let last = path.segments.last()?;
    if last.ident != "Option" {
        return None;
    }
    let args = match &last.arguments {
        syn::PathArguments::AngleBracketed(a) => &a.args,
        _ => return None,
    };
    let inner = args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })?;
    let inner_path = match inner {
        syn::Type::Path(type_path) => &type_path.path,
        _ => return None,
    };
    Some(inner_path.segments.last()?.ident.to_string())
}

fn return_is_result(output: &syn::ReturnType) -> bool {
    let ty = match output {
        syn::ReturnType::Default => return false,
        syn::ReturnType::Type(_, ty) => ty.as_ref(),
    };
    let path = match ty {
        syn::Type::Path(type_path) => &type_path.path,
        _ => return false,
    };
    path.segments.last().map(|seg| seg.ident == "Result").unwrap_or(false)
}

fn build_method_wrapper(
    method: &syn::ImplItemFn,
    flavor: &MethodFlavor,
) -> proc_macro2::TokenStream {
    let original_ident = &method.sig.ident;
    let original_name = original_ident.to_string();
    let wrapper_ident = quote::format_ident!("__pyo3_{}", original_ident);
    let asyncness = method.sig.asyncness.is_some();

    let pyo3_attr = if asyncness {
        // Python properties can't be async — emit as plain methods.
        match flavor {
            MethodFlavor::Constructor => quote! { #[new] },
            MethodFlavor::Factory => quote! { #[staticmethod] },
            _ => quote! {},
        }
    } else {
        match flavor {
            MethodFlavor::Plain => quote! {},
            MethodFlavor::Getter => quote! { #[getter] },
            MethodFlavor::Setter => quote! { #[setter] },
            MethodFlavor::Constructor => quote! { #[new] },
            MethodFlavor::Factory => quote! { #[staticmethod] },
        }
    };

    // Determine receiver kind and arg list (excluding receiver).
    let mut receiver: Option<&syn::Receiver> = None;
    let mut typed_args: Vec<&syn::PatType> = Vec::new();
    for input in &method.sig.inputs {
        match input {
            syn::FnArg::Receiver(r) => receiver = Some(r),
            syn::FnArg::Typed(pt) => typed_args.push(pt),
        }
    }
    let arg_idents: Vec<syn::Ident> = typed_args
        .iter()
        .filter_map(|pt| match pt.pat.as_ref() {
            syn::Pat::Ident(pi) => Some(pi.ident.clone()),
            _ => None,
        })
        .collect();
    // For each arg: if its type is `&T`, the wrapper takes `T` and the call site adds `&`.
    let typed_arg_tokens: Vec<proc_macro2::TokenStream> = typed_args
        .iter()
        .map(|pt| {
            let pat = &pt.pat;
            let ty = strip_outer_ref(&pt.ty);
            quote! { #pat: #ty }
        })
        .collect();
    let arg_call_tokens: Vec<proc_macro2::TokenStream> = typed_args
        .iter()
        .zip(arg_idents.iter())
        .map(|(pt, ident)| {
            if matches!(pt.ty.as_ref(), syn::Type::Reference(_)) {
                quote! { &#ident }
            } else {
                quote! { #ident }
            }
        })
        .collect();

    let receiver_tokens = match receiver {
        Some(r) if r.mutability.is_some() => quote! { &mut self },
        Some(_) => quote! { &self },
        None => quote! {},
    };
    let receiver_separator = if receiver.is_some() && !typed_arg_tokens.is_empty() {
        quote! { , }
    } else {
        quote! {}
    };

    let pyo3_name_attr = match flavor {
        MethodFlavor::Constructor => quote! {},
        _ => quote! { #[pyo3(name = #original_name)] },
    };

    let call_target = if receiver.is_some() {
        quote! { self.#original_ident }
    } else {
        quote! { Self::#original_ident }
    };

    if asyncness {
        let inner_repr = rust_return_to_python_repr(&method.sig.output);
        let awaitable_repr = format!("collections.abc.Awaitable[{inner_repr}]");
        let stub_attr = quote! {
            #[gen_stub(override_return_type(type_repr = #awaitable_repr, imports = ("collections.abc")))]
        };
        // Async wrapper: clone receiver into the future, schedule on tokio.
        let py_arg = quote! { py: ::pyo3::Python<'py> };
        let inputs_with_py = if receiver.is_some() {
            if typed_arg_tokens.is_empty() {
                quote! { #receiver_tokens, #py_arg }
            } else {
                quote! { #receiver_tokens, #py_arg, #( #typed_arg_tokens ),* }
            }
        } else if typed_arg_tokens.is_empty() {
            quote! { #py_arg }
        } else {
            quote! { #py_arg, #( #typed_arg_tokens ),* }
        };
        let capture = if receiver.is_some() {
            quote! { let __this = ::std::clone::Clone::clone(self); }
        } else {
            quote! {}
        };
        let invoke = if receiver.is_some() {
            quote! { __this.#original_ident( #( #arg_call_tokens ),* ).await }
        } else {
            quote! { Self::#original_ident( #( #arg_call_tokens ),* ).await }
        };
        let body = if return_is_result(&method.sig.output) {
            quote! {
                let __result = #invoke;
                __result.map_err(::std::convert::Into::into)
            }
        } else {
            quote! {
                let __value = #invoke;
                ::std::result::Result::<_, ::pyo3::PyErr>::Ok(__value)
            }
        };
        quote! {
            #pyo3_attr
            #pyo3_name_attr
            #stub_attr
            pub fn #wrapper_ident<'py>(
                #inputs_with_py
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                #capture
                ::pyo3_async_runtimes::tokio::future_into_py(py, async move {
                    #body
                })
            }
        }
    } else {
        // Sync wrapper: forward.
        let output = &method.sig.output;
        let inputs = if receiver.is_some() {
            quote! { #receiver_tokens #receiver_separator #( #typed_arg_tokens ),* }
        } else {
            quote! { #( #typed_arg_tokens ),* }
        };
        quote! {
            #pyo3_attr
            #pyo3_name_attr
            pub fn #wrapper_ident( #inputs ) #output {
                #call_target( #( #arg_call_tokens ),* )
            }
        }
    }
}

fn strip_outer_ref(ty: &syn::Type) -> proc_macro2::TokenStream {
    if let syn::Type::Reference(reference) = ty {
        let inner = &reference.elem;
        quote! { #inner }
    } else {
        quote! { #ty }
    }
}

fn rust_return_to_python_repr(output: &syn::ReturnType) -> String {
    let ty = match output {
        syn::ReturnType::Default => return "None".to_string(),
        syn::ReturnType::Type(_, ty) => ty.as_ref(),
    };
    rust_type_to_python_repr(unwrap_result(ty))
}

fn rust_return_to_python_repr_for_self(
    output: &syn::ReturnType,
    self_type: &syn::Type,
) -> String {
    let repr = rust_return_to_python_repr(output);
    let self_name =
        type_path_last_ident(self_type).map(|ident| ident.to_string()).unwrap_or_else(|| "typing.Any".to_string());
    repr.replace("Self", &self_name)
}

fn type_path_last_ident(ty: &syn::Type) -> Option<&Ident> {
    if let syn::Type::Path(type_path) = ty {
        return type_path.path.segments.last().map(|seg| &seg.ident);
    }
    None
}

fn unwrap_result(ty: &syn::Type) -> &syn::Type {
    if let syn::Type::Path(type_path) = ty {
        if let Some(last) = type_path.path.segments.last() {
            if last.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) =
                        args.args.iter().find(|arg| matches!(arg, syn::GenericArgument::Type(_)))
                    {
                        return inner;
                    }
                }
            }
        }
    }
    ty
}

fn rust_type_to_python_repr(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => {
            let Some(last) = type_path.path.segments.last() else {
                return "typing.Any".to_string();
            };
            let name = last.ident.to_string();
            match name.as_str() {
                "bool" => "builtins.bool".to_string(),
                "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "isize" | "usize" => {
                    "builtins.int".to_string()
                },
                "f32" | "f64" => "builtins.float".to_string(),
                "String" | "str" => "builtins.str".to_string(),
                "Option" => match generic_args(&last.arguments).as_slice() {
                    [t] => format!("{} | None", rust_type_to_python_repr(t)),
                    _ => "typing.Any".to_string(),
                },
                "Vec" => match generic_args(&last.arguments).as_slice() {
                    [t] => format!("builtins.list[{}]", rust_type_to_python_repr(t)),
                    _ => "typing.Any".to_string(),
                },
                "HashMap" | "BTreeMap" | "IndexMap" => match generic_args(&last.arguments).as_slice() {
                    [k, v] => {
                        format!("builtins.dict[{}, {}]", rust_type_to_python_repr(k), rust_type_to_python_repr(v))
                    },
                    _ => "typing.Any".to_string(),
                },
                _ => name,
            }
        },
        syn::Type::Tuple(tuple) if tuple.elems.is_empty() => "None".to_string(),
        syn::Type::Reference(reference) => rust_type_to_python_repr(reference.elem.as_ref()),
        _ => "typing.Any".to_string(),
    }
}

fn generic_args(arguments: &syn::PathArguments) -> Vec<&syn::Type> {
    let syn::PathArguments::AngleBracketed(angle) = arguments else {
        return Vec::new();
    };
    angle
        .args
        .iter()
        .filter_map(|arg| match arg {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
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
    let asyncness = method.sig.asyncness.is_some();
    let arg_idents = factory_arg_idents(inputs);

    if asyncness {
        let body = if return_is_result(output) {
            quote! {
                let __result = <#self_type>::#method_ident( #( #arg_idents ),* ).await;
                __result.map_err(::std::convert::Into::into)
            }
        } else {
            quote! {
                let __value = <#self_type>::#method_ident( #( #arg_idents ),* ).await;
                ::std::result::Result::<_, ::pyo3::PyErr>::Ok(__value)
            }
        };
        let inner_repr = rust_return_to_python_repr_for_self(output, self_type);
        let awaitable_repr = format!("collections.abc.Awaitable[{inner_repr}]");
        quote! {
            #[cfg(feature = "bindings-pyo3")]
            const _: () = {
                #[allow(unused_imports)]
                use ::pyo3::prelude::*;
                #[pyo3_stub_gen::derive::gen_stub_pymethods]
                #[pyo3::pymethods]
                impl #self_type {
                    #[staticmethod]
                    #[pyo3(name = #method_name)]
                    #[gen_stub(override_return_type(type_repr = #awaitable_repr, imports = ("collections.abc")))]
                    pub fn #wrapper_ident<'py>(
                        py: ::pyo3::Python<'py>,
                        #inputs
                    ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                        ::pyo3_async_runtimes::tokio::future_into_py(py, async move {
                            #body
                        })
                    }
                }
            };
        }
    } else {
        quote! {
            #[cfg(feature = "bindings-pyo3")]
            const _: () = {
                #[allow(unused_imports)]
                use ::pyo3::prelude::*;
                #[pyo3_stub_gen::derive::gen_stub_pymethods]
                #[pyo3::pymethods]
                impl #self_type {
                    #[staticmethod]
                    #[pyo3(name = #method_name)]
                    pub fn #wrapper_ident( #inputs ) #output {
                        <#self_type>::#method_ident( #( #arg_idents ),* )
                    }
                }
            };
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

pub fn struct_constructor(
    item_struct: &syn::ItemStruct,
    kind: &BindingKind,
) -> proc_macro2::TokenStream {
    if !matches!(kind, BindingKind::Struct | BindingKind::ClassCloneable) {
        return quote! {};
    }
    let type_name = &item_struct.ident;
    let fields = match &item_struct.fields {
        syn::Fields::Named(named) => &named.named,
        _ => return quote! {},
    };
    let pub_fields: Vec<&syn::Field> =
        fields.iter().filter(|field| matches!(field.vis, syn::Visibility::Public(_))).collect();
    if pub_fields.is_empty() {
        return quote! {};
    }
    let params: Vec<proc_macro2::TokenStream> = pub_fields
        .iter()
        .map(|field| {
            let ident = field.ident.as_ref().unwrap();
            let ty = &field.ty;
            quote! { #ident: #ty }
        })
        .collect();
    let assigns: Vec<proc_macro2::TokenStream> = pub_fields
        .iter()
        .map(|field| {
            let ident = field.ident.as_ref().unwrap();
            quote! { #ident }
        })
        .collect();
    quote! {
        #[cfg(feature = "bindings-pyo3")]
        const _: () = {
            #[allow(unused_imports)]
            use ::pyo3::prelude::*;
            #[pyo3_stub_gen::derive::gen_stub_pymethods]
            #[pyo3::pymethods]
            impl #type_name {
                #[new]
                fn __pyo3_new(#(#params),*) -> Self {
                    Self { #(#assigns),* }
                }
            }
        };
    }
}

pub fn registration(type_name: &Ident) -> proc_macro2::TokenStream {
    let type_name_str = type_name.to_string();
    quote! {
        #[cfg(feature = "bindings-pyo3")]
        ::inventory::submit! {
            ::bindings_types::PyClassRegistration {
                register: |module| {
                    use ::pyo3::types::{PyAnyMethods, PyModuleMethods};
                    module.add_class::<#type_name>()?;
                    let cls = module.as_any().getattr(#type_name_str)?;
                    let module_name: ::std::string::String =
                        module.as_any().getattr("__name__")?.extract()?;
                    let public_module = module_name
                        .rsplit_once('.')
                        .map(|(parent, _)| parent.to_owned())
                        .unwrap_or(module_name);
                    cls.setattr("__module__", public_module)?;
                    ::std::result::Result::Ok(())
                },
            }
        }
    }
}

pub fn error_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    let from_py_message = format!("{} cannot be received from Python", type_name);
    quote! {
        #[cfg(feature = "bindings-pyo3")]
        impl From<#type_name> for ::pyo3::PyErr {
            fn from(error: #type_name) -> Self {
                ::pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
            }
        }

        #[cfg(feature = "bindings-pyo3")]
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #type_name {
            type Error = ::pyo3::PyErr;
            fn extract(_obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::std::result::Result<Self, Self::Error> {
                ::std::result::Result::Err(
                    ::pyo3::exceptions::PyTypeError::new_err(#from_py_message),
                )
            }
        }

        #[cfg(feature = "bindings-pyo3")]
        impl<'py> ::pyo3::IntoPyObject<'py> for #type_name {
            type Target = ::pyo3::types::PyString;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
            type Error = ::std::convert::Infallible;

            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<Self::Output, Self::Error> {
                ::std::result::Result::Ok(::pyo3::types::PyString::new(py, &self.to_string()))
            }
        }

        #[cfg(feature = "bindings-pyo3")]
        impl ::pyo3_stub_gen::PyStubType for #type_name {
            fn type_output() -> ::pyo3_stub_gen::TypeInfo {
                ::pyo3_stub_gen::TypeInfo::builtin("str")
            }
            fn type_input() -> ::pyo3_stub_gen::TypeInfo {
                ::pyo3_stub_gen::TypeInfo::builtin("str")
            }
        }
    }
}
