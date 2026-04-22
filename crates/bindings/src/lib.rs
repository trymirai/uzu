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
    Alias,
    Implementation,
    Method,
    Constructor,
    Factory,
    Getter,
    Setter,
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
            "Alias" => BindingKind::Alias,
            "Implementation" => BindingKind::Implementation,
            "Method" => BindingKind::Method,
            "Constructor" => BindingKind::Constructor,
            "Factory" => BindingKind::Factory,
            "Getter" => BindingKind::Getter,
            "Setter" => BindingKind::Setter,
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
        BindingKind::Enum | BindingKind::Struct | BindingKind::Class | BindingKind::ClassCloneable => {
            let item = parse_macro_input!(item as syn::Item);
            let type_name = match &item {
                syn::Item::Struct(item_struct) => Some(&item_struct.ident),
                syn::Item::Enum(item_enum) => Some(&item_enum.ident),
                _ => None,
            };
            let enum_has_data_variants = match &item {
                syn::Item::Enum(item_enum) => {
                    item_enum.variants.iter().any(|variant| !matches!(variant.fields, syn::Fields::Unit))
                },
                _ => false,
            };
            let napi = napi::attributes(&kind);
            let uniffi = uniffi::attributes(&kind);
            let pyo3 = pyo3::attributes(&kind);
            let wasm = wasm::attributes(&kind);
            let napi_value = match (&kind, type_name) {
                (BindingKind::Struct, Some(ident)) => napi::struct_value_implementations(ident),
                (BindingKind::Enum, Some(ident)) if enum_has_data_variants => napi::struct_value_implementations(ident),
                (BindingKind::ClassCloneable, Some(ident)) => napi::class_value_implementations(ident),
                _ => quote! {},
            };
            quote! {
                #napi
                #uniffi
                #pyo3
                #wasm
                #item
                #napi_value
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
            for item in &mut item_implementation.items {
                if let ImplItem::Fn(method) = item {
                    rewrite_method_attributes(&mut method.attrs);
                }
            }
            let napi = napi::attributes(&kind);
            let uniffi = uniffi::attributes(&kind);
            let pyo3 = pyo3::attributes(&kind);
            let wasm = wasm::attributes(&kind);
            quote! {
                #napi
                #uniffi
                #pyo3
                #wasm
                #item_implementation
            }
            .into()
        },
        BindingKind::Method
        | BindingKind::Constructor
        | BindingKind::Factory
        | BindingKind::Getter
        | BindingKind::Setter => {
            let tokens = proc_macro2::TokenStream::from(item);
            quote! { #tokens }.into()
        },
        BindingKind::Error => {
            let item = parse_macro_input!(item as syn::Item);
            let type_name = match &item {
                syn::Item::Enum(item_enum) => &item_enum.ident,
                _ => {
                    return syn::Error::new_spanned(&item, "Error can only be applied to enums")
                        .to_compile_error()
                        .into();
                },
            };
            let uniffi = uniffi::error_attribute();
            let napi_implementations = napi::error_implementations(type_name);
            let pyo3_implementations = pyo3::error_implementations(type_name);
            let wasm_implementations = wasm::error_implementations(type_name);
            quote! {
                #uniffi
                #item
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
        _ => None,
    }
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
