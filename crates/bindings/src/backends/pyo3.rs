use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::{
    backends::Backend,
    contexts::{
        ClassContext, EnumerationContext, EnumerationShape, ErrorContext, ImplementationContext, MethodMetadata,
        StructureContext,
    },
    types::MethodFlavor,
};

pub struct Pyo3;

impl Backend for Pyo3 {
    fn structure_attributes(_context: &StructureContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(get_all, from_py_object))]
        }
    }

    fn structure_companions(context: &StructureContext) -> TokenStream {
        let registration_block = registration(&context.item.ident);
        let constructor_block = struct_constructor(context.item);
        quote! {
            #registration_block
            #constructor_block
        }
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass)]
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(from_py_object))]
        }
    }

    fn class_companions(context: &ClassContext) -> TokenStream {
        registration(&context.item.ident)
    }

    fn enumeration_attributes(context: &EnumerationContext) -> TokenStream {
        let stub_gen_attribute = match context.shape {
            EnumerationShape::Unit => quote! {
                #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
            },
            EnumerationShape::Data => quote! {
                #[cfg_attr(feature = "bindings-pyo3", pyo3_stub_gen::derive::gen_stub_pyclass_complex_enum)]
            },
        };
        quote! {
            #stub_gen_attribute
            #[cfg_attr(feature = "bindings-pyo3", pyo3::pyclass(eq, from_py_object))]
        }
    }

    fn enumeration_companions(context: &EnumerationContext) -> TokenStream {
        registration(&context.item.ident)
    }

    fn implementation_companions(context: &ImplementationContext) -> TokenStream {
        implementation_expansion(context)
    }

    fn method_attributes(metadata: &MethodMetadata) -> TokenStream {
        match metadata.flavor {
            MethodFlavor::Constructor => quote! {
                #[cfg_attr(feature = "bindings-pyo3", new)]
            },
            MethodFlavor::Factory => quote! {
                #[cfg_attr(feature = "bindings-pyo3", staticmethod)]
            },
            _ => quote! {},
        }
    }

    fn method_companions(
        context: &ImplementationContext,
        metadata: &MethodMetadata,
    ) -> TokenStream {
        match metadata.flavor {
            MethodFlavor::Factory => factory_expansion(&context.self_type, metadata),
            MethodFlavor::FactoryWithCallback => factory_with_callback_expansion(&context.self_type, metadata),
            _ => quote! {},
        }
    }

    fn error_companions(context: &ErrorContext) -> TokenStream {
        error_implementations(&context.item.ident)
    }
}

fn implementation_expansion(context: &ImplementationContext) -> TokenStream {
    let mut wrappers: Vec<TokenStream> = Vec::new();
    for metadata in context.all_methods() {
        if matches!(metadata.flavor, MethodFlavor::Factory | MethodFlavor::FactoryWithCallback) {
            continue;
        }
        wrappers.push(build_method_wrapper(metadata));
    }

    if let Some(metadata) = context.methods_of(MethodFlavor::StreamNext).next() {
        wrappers.push(build_stream_protocol(&metadata.method));
    }

    if wrappers.is_empty() {
        return quote! {};
    }

    let self_type = &context.self_type;
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

fn build_stream_protocol(next_method: &syn::ImplItemFn) -> TokenStream {
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
    let return_type = match output {
        syn::ReturnType::Default => return None,
        syn::ReturnType::Type(_, return_type) => return_type.as_ref(),
    };
    let path = match return_type {
        syn::Type::Path(type_path) => &type_path.path,
        _ => return None,
    };
    let last = path.segments.last()?;
    if last.ident != "Option" {
        return None;
    }
    let arguments = match &last.arguments {
        syn::PathArguments::AngleBracketed(angle) => &angle.args,
        _ => return None,
    };
    let inner = arguments.iter().find_map(|argument| match argument {
        syn::GenericArgument::Type(inner_type) => Some(inner_type),
        _ => None,
    })?;
    let inner_path = match inner {
        syn::Type::Path(type_path) => &type_path.path,
        _ => return None,
    };
    Some(inner_path.segments.last()?.ident.to_string())
}

fn build_method_wrapper(metadata: &MethodMetadata) -> TokenStream {
    let method = &metadata.method;
    let flavor = metadata.flavor;
    let original_ident = &method.sig.ident;
    let original_name = original_ident.to_string();
    let wrapper_ident = format_ident!("__pyo3_{}", original_ident);
    let asyncness = metadata.is_async();

    let pyo3_attribute = if asyncness {
        match flavor {
            MethodFlavor::Constructor => quote! { #[new] },
            MethodFlavor::Factory => quote! { #[staticmethod] },
            _ => quote! {},
        }
    } else {
        match flavor {
            MethodFlavor::Constructor => quote! { #[new] },
            MethodFlavor::Factory => quote! { #[staticmethod] },
            MethodFlavor::Getter => quote! { #[getter] },
            MethodFlavor::Setter => quote! { #[setter] },
            MethodFlavor::Plain | MethodFlavor::FactoryWithCallback | MethodFlavor::StreamNext => {
                quote! {}
            },
        }
    };

    let receiver = metadata.receiver();
    let typed_args = metadata.typed_args();
    let arg_idents = metadata.arg_idents();
    let typed_arg_tokens: Vec<TokenStream> = typed_args
        .iter()
        .map(|pat_type| {
            let pattern = &pat_type.pat;
            let owned_type = strip_outer_ref(&pat_type.ty);
            quote! { #pattern: #owned_type }
        })
        .collect();
    let arg_call_tokens: Vec<TokenStream> = typed_args
        .iter()
        .zip(arg_idents.iter())
        .map(|(pat_type, ident)| {
            if matches!(pat_type.ty.as_ref(), syn::Type::Reference(_)) {
                quote! { &#ident }
            } else {
                quote! { #ident }
            }
        })
        .collect();

    let receiver_tokens = match receiver {
        Some(receiver_arg) if receiver_arg.mutability.is_some() => quote! { &mut self },
        Some(_) => quote! { &self },
        None => quote! {},
    };
    let receiver_separator = if receiver.is_some() && !typed_arg_tokens.is_empty() {
        quote! { , }
    } else {
        quote! {}
    };

    let pyo3_name_attribute = match flavor {
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
        let stub_attribute = quote! {
            #[gen_stub(override_return_type(type_repr = #awaitable_repr, imports = ("collections.abc")))]
        };
        let py_argument = quote! { py: ::pyo3::Python<'py> };
        let inputs_with_py = if receiver.is_some() {
            if typed_arg_tokens.is_empty() {
                quote! { #receiver_tokens, #py_argument }
            } else {
                quote! { #receiver_tokens, #py_argument, #( #typed_arg_tokens ),* }
            }
        } else if typed_arg_tokens.is_empty() {
            quote! { #py_argument }
        } else {
            quote! { #py_argument, #( #typed_arg_tokens ),* }
        };
        let capture = if receiver.is_some() {
            quote! { let __this = ::std::clone::Clone::clone(self); }
        } else {
            quote! {}
        };
        let invocation = if receiver.is_some() {
            quote! { __this.#original_ident( #( #arg_call_tokens ),* ).await }
        } else {
            quote! { Self::#original_ident( #( #arg_call_tokens ),* ).await }
        };
        let body = if metadata.returns_result() {
            quote! {
                let __result = #invocation;
                __result.map_err(::std::convert::Into::into)
            }
        } else {
            quote! {
                let __value = #invocation;
                ::std::result::Result::<_, ::pyo3::PyErr>::Ok(__value)
            }
        };
        quote! {
            #pyo3_attribute
            #pyo3_name_attribute
            #stub_attribute
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
        let output = &method.sig.output;
        let inputs = if receiver.is_some() {
            quote! { #receiver_tokens #receiver_separator #( #typed_arg_tokens ),* }
        } else {
            quote! { #( #typed_arg_tokens ),* }
        };
        quote! {
            #pyo3_attribute
            #pyo3_name_attribute
            pub fn #wrapper_ident( #inputs ) #output {
                #call_target( #( #arg_call_tokens ),* )
            }
        }
    }
}

fn strip_outer_ref(ty: &syn::Type) -> TokenStream {
    if let syn::Type::Reference(reference) = ty {
        let inner = &reference.elem;
        quote! { #inner }
    } else {
        quote! { #ty }
    }
}

fn rust_return_to_python_repr(output: &syn::ReturnType) -> String {
    let return_type = match output {
        syn::ReturnType::Default => return "None".to_string(),
        syn::ReturnType::Type(_, return_type) => return_type.as_ref(),
    };
    rust_type_to_python_repr(unwrap_result(return_type))
}

fn rust_return_to_python_repr_for_self(
    output: &syn::ReturnType,
    self_type: &syn::Type,
) -> String {
    let representation = rust_return_to_python_repr(output);
    let self_name =
        type_path_last_ident(self_type).map(|ident| ident.to_string()).unwrap_or_else(|| "typing.Any".to_string());
    representation.replace("Self", &self_name)
}

fn type_path_last_ident(ty: &syn::Type) -> Option<&Ident> {
    if let syn::Type::Path(type_path) = ty {
        return type_path.path.segments.last().map(|segment| &segment.ident);
    }
    None
}

fn unwrap_result(ty: &syn::Type) -> &syn::Type {
    if let syn::Type::Path(type_path) = ty {
        if let Some(last) = type_path.path.segments.last() {
            if last.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(arguments) = &last.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) =
                        arguments.args.iter().find(|argument| matches!(argument, syn::GenericArgument::Type(_)))
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
                    [inner] => format!("{} | None", rust_type_to_python_repr(inner)),
                    _ => "typing.Any".to_string(),
                },
                "Vec" => match generic_args(&last.arguments).as_slice() {
                    [inner] => format!("builtins.list[{}]", rust_type_to_python_repr(inner)),
                    _ => "typing.Any".to_string(),
                },
                "HashMap" | "BTreeMap" | "IndexMap" => match generic_args(&last.arguments).as_slice() {
                    [key, value] => {
                        format!("builtins.dict[{}, {}]", rust_type_to_python_repr(key), rust_type_to_python_repr(value))
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
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

fn factory_expansion(
    self_type: &syn::Type,
    metadata: &MethodMetadata,
) -> TokenStream {
    let method = &metadata.method;
    let method_ident = &method.sig.ident;
    let method_name = method_ident.to_string();
    let wrapper_ident = format_ident!("{}_bindings_pyo3", method_ident);
    let inputs = &method.sig.inputs;
    let output = &method.sig.output;
    let asyncness = metadata.is_async();
    let arg_idents = metadata.arg_idents();

    if asyncness {
        let body = if metadata.returns_result() {
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

fn factory_with_callback_expansion(
    self_type: &syn::Type,
    metadata: &MethodMetadata,
) -> TokenStream {
    let method = &metadata.method;
    let method_ident = &method.sig.ident;
    let method_name = method_ident.to_string();
    let body = &method.block;
    let wrapper_ident = format_ident!("{}_bindings_pyo3", method_ident);

    let callback_inputs = match metadata.callback_inputs() {
        Some(inputs) => inputs,
        None => {
            return syn::Error::new_spanned(
                &method.sig,
                "Bindings::export(Method(FactoryWithCallback)) requires a parameter of type \
                 `Box<dyn Fn(..) + Send + Sync>` (return type must be `()`)",
            )
            .to_compile_error();
        },
    };
    let synthetic_idents: Vec<Ident> = (0..callback_inputs.len()).map(|index| format_ident!("arg{index}")).collect();
    let py_call_arguments = if synthetic_idents.is_empty() {
        quote! { () }
    } else {
        quote! { ( #( #synthetic_idents ),* , ) }
    };
    let callback_arg_reprs: Vec<String> = callback_inputs.iter().map(rust_type_to_python_repr).collect();
    let callable_repr = format!("collections.abc.Callable[[{}], None]", callback_arg_reprs.join(", "));

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
                pub fn #wrapper_ident(
                    #[gen_stub(override_type(type_repr = #callable_repr, imports = ("collections.abc")))]
                    callback: ::pyo3::Py<::pyo3::PyAny>,
                ) -> ::pyo3::PyResult<Self> {
                    let callback: ::std::boxed::Box<
                        dyn ::std::ops::Fn( #( #callback_inputs ),* ) + ::std::marker::Send + ::std::marker::Sync,
                    > = ::std::boxed::Box::new(
                        move | #( #synthetic_idents: #callback_inputs ),* | {
                            ::pyo3::Python::attach(|py| {
                                let _ = callback.call1(py, #py_call_arguments);
                            });
                        },
                    );
                    ::std::result::Result::Ok(#body)
                }
            }
        };
    }
}

fn struct_constructor(item_struct: &syn::ItemStruct) -> TokenStream {
    let type_name = &item_struct.ident;
    let fields = match &item_struct.fields {
        syn::Fields::Named(named) => &named.named,
        _ => return quote! {},
    };
    let public_fields: Vec<&syn::Field> =
        fields.iter().filter(|field| matches!(field.vis, syn::Visibility::Public(_))).collect();
    if public_fields.is_empty() {
        return quote! {};
    }
    let parameters: Vec<TokenStream> = public_fields
        .iter()
        .map(|field| {
            let ident = field.ident.as_ref().unwrap();
            let ty = &field.ty;
            quote! { #ident: #ty }
        })
        .collect();
    let assignments: Vec<TokenStream> = public_fields
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
                fn __pyo3_new(#( #parameters ),*) -> Self {
                    Self { #( #assignments ),* }
                }
            }
        };
    }
}

fn registration(type_name: &Ident) -> TokenStream {
    let type_name_string = type_name.to_string();
    quote! {
        #[cfg(feature = "bindings-pyo3")]
        ::inventory::submit! {
            ::bindings_types::PyClassRegistration {
                register: |module| {
                    use ::pyo3::types::{PyAnyMethods, PyModuleMethods};
                    module.add_class::<#type_name>()?;
                    let cls = module.as_any().getattr(#type_name_string)?;
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

fn error_implementations(type_name: &Ident) -> TokenStream {
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
