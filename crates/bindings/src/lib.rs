mod napi;
mod pyo3;
mod uniffi;
mod wasm;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Ident, ImplItem, ItemImpl, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

pub(crate) enum BindingKind {
    Enum,
    Struct,
    Class,
    ClassCloneable,
    Stream,
    Alias,
    Implementation,
    Method,
    Constructor,
    Factory,
    FactoryWithCallback,
    Getter,
    Setter,
    StreamNext,
    Error,
}

pub(crate) enum MethodFlavor {
    Plain,
    Constructor,
    Factory,
    Getter,
    Setter,
}

struct ExportArguments {
    kind: BindingKind,
}

impl Parse for ExportArguments {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let identifier: Ident = input.parse()?;
        let kind = match identifier.to_string().as_str() {
            "Enum" => BindingKind::Enum,
            "Struct" => BindingKind::Struct,
            "Class" => BindingKind::Class,
            "ClassCloneable" => BindingKind::ClassCloneable,
            "Stream" => BindingKind::Stream,
            "Alias" => BindingKind::Alias,
            "Implementation" => BindingKind::Implementation,
            "Method" => BindingKind::Method,
            "Constructor" => BindingKind::Constructor,
            "Factory" => BindingKind::Factory,
            "FactoryWithCallback" => BindingKind::FactoryWithCallback,
            "Getter" => BindingKind::Getter,
            "Setter" => BindingKind::Setter,
            "StreamNext" => BindingKind::StreamNext,
            "Error" => BindingKind::Error,
            other => {
                return Err(syn::Error::new(identifier.span(), format!("Unknown binding kind: {other}")));
            },
        };
        if input.peek(Token![,]) {
            return Err(input.error("bindings::export accepts a single kind argument"));
        }
        Ok(ExportArguments {
            kind,
        })
    }
}

#[proc_macro_attribute]
pub fn export(
    arguments: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let ExportArguments {
        kind,
    } = parse_macro_input!(arguments as ExportArguments);

    match kind {
        BindingKind::Enum => {
            let item = parse_macro_input!(item as syn::Item);
            let item_enum = match &item {
                syn::Item::Enum(item_enum) => item_enum,
                _ => {
                    return syn::Error::new_spanned(&item, "bindings::export(Enum) must be applied to an enum")
                        .to_compile_error()
                        .into();
                },
            };

            let mut has_unit_fields = false;
            let mut has_named_fields = false;
            for variant in &item_enum.variants {
                match variant.fields {
                    syn::Fields::Unit => has_unit_fields = true,
                    syn::Fields::Named(_) => has_named_fields = true,
                    syn::Fields::Unnamed(_) => {
                        return syn::Error::new_spanned(
                            &variant.fields,
                            "bindings::export(Enum) variants must use named fields (use `{}` for empty variants), tuple variants are not supported",
                        )
                        .to_compile_error()
                        .into();
                    },
                }
            }
            if has_unit_fields && has_named_fields {
                return syn::Error::new_spanned(
                    &item_enum.ident,
                    "bindings::export(Enum) variants must all be unit (e.g. `Foo`) or all be named (e.g. `Foo {}`, `Foo { x: i64 }`), mixing is not supported",
                )
                .to_compile_error()
                .into();
            }

            let enum_is_data = has_named_fields;
            let uniffi = uniffi::attributes(&kind);
            let pyo3 = pyo3::attributes(&kind);
            let pyo3_stub_gen = pyo3::enum_stub_gen_attribute(enum_is_data);
            let pyo3_registration = pyo3::registration(&item_enum.ident);
            let wasm = wasm::attributes(&kind);

            if enum_is_data {
                let variant_classes = napi::enum_variant_classes(&item_enum.ident, &item_enum.variants);
                quote! {
                    #uniffi
                    #pyo3_stub_gen
                    #pyo3
                    #wasm
                    #item
                    #variant_classes
                    #pyo3_registration
                }
                .into()
            } else {
                let napi = napi::attributes(&kind);
                quote! {
                    #napi
                    #uniffi
                    #pyo3_stub_gen
                    #pyo3
                    #wasm
                    #item
                    #pyo3_registration
                }
                .into()
            }
        },
        BindingKind::Struct | BindingKind::Class | BindingKind::ClassCloneable | BindingKind::Stream => {
            let item = parse_macro_input!(item as syn::Item);
            let item_struct = match &item {
                syn::Item::Struct(item_struct) => Some(item_struct),
                _ => None,
            };
            let type_name = item_struct.map(|s| &s.ident);
            let napi = napi::attributes(&kind);
            let uniffi = uniffi::attributes(&kind);
            let pyo3 = pyo3::attributes(&kind);
            let wasm = wasm::attributes(&kind);
            let napi_value = match (&kind, type_name) {
                (BindingKind::Struct, Some(ident)) => napi::struct_value_implementations(ident),
                (BindingKind::ClassCloneable, Some(ident)) => napi::class_value_implementations(ident),
                _ => quote! {},
            };
            let pyo3_registration = match type_name {
                Some(ident) => pyo3::registration(ident),
                None => quote! {},
            };
            let pyo3_constructor = match item_struct {
                Some(item_struct) => pyo3::struct_constructor(item_struct, &kind),
                None => quote! {},
            };
            quote! {
                #napi
                #uniffi
                #pyo3
                #wasm
                #item
                #napi_value
                #pyo3_registration
                #pyo3_constructor
            }
            .into()
        },
        BindingKind::Alias => {
            let item = parse_macro_input!(item as syn::ItemType);
            let napi = napi::attributes(&kind);
            let uniffi = uniffi::attributes(&kind);
            let pyo3 = pyo3::attributes(&kind);
            let wasm = wasm::attributes(&kind);
            quote! {
                #napi
                #uniffi
                #pyo3
                #wasm
                #item
            }
            .into()
        },
        BindingKind::Implementation => {
            let mut item_implementation = match syn::parse::<ItemImpl>(item) {
                Ok(item_implementation) => item_implementation,
                Err(error) => return error.to_compile_error().into(),
            };
            let self_type = item_implementation.self_ty.clone();

            let mut callback_factories: Vec<syn::ImplItemFn> = Vec::new();
            item_implementation.items.retain(|item| match item {
                ImplItem::Fn(method) if method_has_factory_with_callback(method) => {
                    callback_factories.push(method.clone());
                    false
                },
                _ => true,
            });

            let mut factories: Vec<syn::ImplItemFn> = Vec::new();
            item_implementation.items.retain(|item| match item {
                ImplItem::Fn(method) if method_has_factory(method) => {
                    let mut stripped = method.clone();
                    strip_bindings_export_attrs(&mut stripped.attrs);
                    factories.push(stripped);
                    false
                },
                _ => true,
            });

            let mut stream_next_methods: Vec<syn::ImplItemFn> = Vec::new();
            for item in &item_implementation.items {
                if let ImplItem::Fn(method) = item {
                    if method_has_stream_next(method) {
                        stream_next_methods.push(method.clone());
                    }
                }
            }

            // Capture method flavors before rewriting strips the `bindings::export(...)` attrs.
            let pyo3_methods: Vec<(syn::ImplItemFn, MethodFlavor, bool)> = item_implementation
                .items
                .iter()
                .filter_map(|item| match item {
                    ImplItem::Fn(method) => {
                        let flavor = method.attrs.iter().find_map(method_flavor).unwrap_or(MethodFlavor::Plain);
                        let is_stream_next = method_has_stream_next(method);
                        let mut cleaned = method.clone();
                        strip_bindings_export_attrs(&mut cleaned.attrs);
                        Some((cleaned, flavor, is_stream_next))
                    },
                    _ => None,
                })
                .collect();

            for item in &mut item_implementation.items {
                if let ImplItem::Fn(method) = item {
                    rewrite_method_attributes(&mut method.attrs);
                }
            }

            let callback_expansions: Vec<proc_macro2::TokenStream> = callback_factories
                .iter()
                .map(|method| {
                    let napi_expansion = napi::factory_with_callback(&self_type, method);
                    let uniffi_expansion = uniffi::factory_with_callback(&self_type, method);
                    quote! {
                        #napi_expansion
                        #uniffi_expansion
                    }
                })
                .collect();

            let stream_expansions: Vec<proc_macro2::TokenStream> =
                stream_next_methods.iter().map(|method| napi::stream_next_generator(&self_type, method)).collect();

            let factory_expansions: Vec<proc_macro2::TokenStream> = factories
                .iter()
                .map(|method| {
                    let plain_impl = quote! {
                        impl #self_type {
                            #method
                        }
                    };
                    let napi_expansion = napi::factory(&self_type, method);
                    let uniffi_expansion = uniffi::factory(&self_type, method);
                    let pyo3_expansion = pyo3::factory(&self_type, method);
                    quote! {
                        #plain_impl
                        #napi_expansion
                        #uniffi_expansion
                        #pyo3_expansion
                    }
                })
                .collect();

            let has_async_methods = item_implementation
                .items
                .iter()
                .any(|item| matches!(item, ImplItem::Fn(method) if method.sig.asyncness.is_some()));
            let napi = napi::attributes(&kind);
            let uniffi = uniffi::implementation_attribute(has_async_methods);
            let wasm = wasm::attributes(&kind);
            let pyo3_impl = pyo3::implementation_expansion(&self_type, &pyo3_methods);
            quote! {
                #napi
                #uniffi
                #wasm
                #item_implementation
                #pyo3_impl
                #( #callback_expansions )*
                #( #stream_expansions )*
                #( #factory_expansions )*
            }
            .into()
        },
        BindingKind::Method
        | BindingKind::Constructor
        | BindingKind::Factory
        | BindingKind::FactoryWithCallback
        | BindingKind::Getter
        | BindingKind::Setter
        | BindingKind::StreamNext => {
            let tokens = proc_macro2::TokenStream::from(item);
            quote! { #tokens }.into()
        },
        BindingKind::Error => {
            let item = parse_macro_input!(item as syn::Item);
            let item_enum = match &item {
                syn::Item::Enum(item_enum) => item_enum,
                _ => {
                    return syn::Error::new_spanned(&item, "Error can only be applied to enums")
                        .to_compile_error()
                        .into();
                },
            };
            let type_name = &item_enum.ident;
            let all_unit_variants =
                item_enum.variants.iter().all(|variant| matches!(variant.fields, syn::Fields::Unit));
            let napi_attribute = if all_unit_variants {
                quote! {
                    #[cfg_attr(feature = "bindings-napi", napi_derive::napi(string_enum))]
                }
            } else {
                quote! {
                    #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
                }
            };
            let napi_value = if all_unit_variants {
                quote! {}
            } else {
                napi::struct_value_implementations(type_name)
            };
            let uniffi = uniffi::error_attribute();
            let napi_implementations = napi::error_implementations(type_name);
            let pyo3_implementations = pyo3::error_implementations(type_name);
            let wasm_implementations = wasm::error_implementations(type_name);
            quote! {
                #napi_attribute
                #uniffi
                #item
                #napi_value
                #napi_implementations
                #pyo3_implementations
                #wasm_implementations
            }
            .into()
        },
    }
}

fn rewrite_method_attributes(attributes: &mut Vec<Attribute>) {
    let mut rewritten_attributes: Vec<Attribute> = Vec::with_capacity(attributes.len());
    for attribute in attributes.drain(..) {
        match method_flavor(&attribute) {
            Some(flavor) => {
                rewritten_attributes.push(napi::method_attribute(&flavor));
                if let Some(attribute) = uniffi::method_attribute(&flavor) {
                    rewritten_attributes.push(attribute);
                }
                if let Some(attribute) = pyo3::method_attribute(&flavor) {
                    rewritten_attributes.push(attribute);
                }
            },
            None => rewritten_attributes.push(attribute),
        }
    }
    *attributes = rewritten_attributes;
}

fn method_flavor(attribute: &Attribute) -> Option<MethodFlavor> {
    if !is_exportable_attribute(attribute) {
        return None;
    }
    let arguments: ExportArguments = attribute.parse_args().ok()?;
    match arguments.kind {
        BindingKind::Method => Some(MethodFlavor::Plain),
        BindingKind::Constructor => Some(MethodFlavor::Constructor),
        BindingKind::Factory => Some(MethodFlavor::Factory),
        BindingKind::Getter => Some(MethodFlavor::Getter),
        BindingKind::Setter => Some(MethodFlavor::Setter),
        BindingKind::StreamNext => Some(MethodFlavor::Plain),
        _ => None,
    }
}

fn method_has_factory(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| {
        if !is_exportable_attribute(attribute) {
            return false;
        }
        let Ok(arguments): syn::Result<ExportArguments> = attribute.parse_args() else {
            return false;
        };
        matches!(arguments.kind, BindingKind::Factory)
    })
}

fn strip_bindings_export_attrs(attrs: &mut Vec<Attribute>) {
    attrs.retain(|attribute| !is_exportable_attribute(attribute));
}

fn method_has_stream_next(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| {
        if !is_exportable_attribute(attribute) {
            return false;
        }
        let Ok(arguments): syn::Result<ExportArguments> = attribute.parse_args() else {
            return false;
        };
        matches!(arguments.kind, BindingKind::StreamNext)
    })
}

fn method_has_factory_with_callback(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| {
        if !is_exportable_attribute(attribute) {
            return false;
        }
        let Ok(arguments): syn::Result<ExportArguments> = attribute.parse_args() else {
            return false;
        };
        matches!(arguments.kind, BindingKind::FactoryWithCallback)
    })
}

fn is_exportable_attribute(attribute: &Attribute) -> bool {
    let path = attribute.path();
    let Some(last) = path.segments.last() else {
        return false;
    };
    if last.ident != "export" {
        return false;
    }
    match path.segments.first() {
        Some(first) if first.ident == "bindings" || first.ident == "export" => true,
        _ => false,
    }
}
