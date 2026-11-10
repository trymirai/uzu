use convert_case::{Case, Casing};
use heck::ToShoutySnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::backends::typescript::{option_inner, optional_class_field_type, typescript_type};

/// The JavaScript name of a field, parameter, or method, converted exactly as napi-derive does
/// (same `convert_case` version), so that e.g. `crc32c` becomes `crc32C` in both packages.
pub fn js_name(rust_name: &str) -> String {
    rust_name.trim_start_matches("r#").trim_start_matches('_').to_case(Case::Camel)
}

/// A member of an object type literal or interface, e.g. `name?: string`.
pub fn object_member(
    name: &str,
    ty: &syn::Type,
) -> String {
    match option_inner(ty) {
        Some(inner) => format!("{name}?: {}", typescript_type(inner, "")),
        None => format!("{name}: {}", typescript_type(ty, "")),
    }
}

/// A field of a class, e.g. `name?: string | null`, see `optional_class_field_type`.
pub fn class_member(
    name: &str,
    ty: &syn::Type,
) -> String {
    match option_inner(ty) {
        Some(inner) => format!("{name}?: {}", optional_class_field_type(inner, "")),
        None => object_member(name, ty),
    }
}

/// The `#[cfg(...)]` attributes of an item, to repeat on the code generated for it.
pub fn cfg_attributes(attributes: &[syn::Attribute]) -> Vec<&syn::Attribute> {
    attributes.iter().filter(|attribute| attribute.path().is_ident("cfg")).collect()
}

pub fn doc_attributes(attributes: &[syn::Attribute]) -> Vec<&syn::Attribute> {
    attributes.iter().filter(|attribute| attribute.path().is_ident("doc")).collect()
}

/// The combined `#[cfg(...)]` predicate of an item, or `None` when it is unconditional.
pub fn cfg_predicate(attributes: &[syn::Attribute]) -> Option<TokenStream> {
    let predicates: Vec<TokenStream> = cfg_attributes(attributes)
        .into_iter()
        .filter_map(|attribute| attribute.meta.require_list().ok().map(|list| list.tokens.clone()))
        .collect();
    match predicates.len() {
        0 => None,
        1 => predicates.into_iter().next(),
        _ => Some(quote! { all(#(#predicates),*) }),
    }
}

/// Emits a TypeScript declaration whose content depends on which entries (variants or fields) are
/// compiled in. `render` receives, per entry, whether it is included. One custom section is emitted
/// per combination of the entries' distinct `cfg` predicates, each gated on that combination, so the
/// d.ts always matches what the wasm build contains.
pub fn custom_sections(
    type_name: &str,
    predicates: &[Option<TokenStream>],
    render: impl Fn(&[bool]) -> String,
) -> TokenStream {
    let mut distinct: Vec<(String, TokenStream)> = Vec::new();
    for predicate in predicates.iter().flatten() {
        let key = predicate.to_string();
        if !distinct.iter().any(|(existing, _)| *existing == key) {
            distinct.push((key, predicate.clone()));
        }
    }
    if distinct.len() > 4 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("{type_name} has more than 4 distinct cfg predicates, which bindings::export does not support"),
        )
        .to_compile_error();
    }

    let sections = (0..1usize << distinct.len()).map(|mask| {
        let enabled = |key: &str| {
            let index = distinct.iter().position(|(existing, _)| existing == key).expect("collected above");
            mask & (1 << index) != 0
        };
        let gates = distinct.iter().map(|(key, predicate)| {
            if enabled(key) {
                quote! { #predicate }
            } else {
                quote! { not(#predicate) }
            }
        });
        let included: Vec<bool> =
            predicates.iter().map(|predicate| predicate.as_ref().is_none_or(|p| enabled(&p.to_string()))).collect();
        let text = render(&included);
        let ident = format_ident!("__UZU_TYPESCRIPT_{}_{}", type_name.to_shouty_snake_case(), mask);
        quote! {
            #[cfg(all(feature = "bindings-wasm", #(#gates),*))]
            #[allow(non_upper_case_globals)]
            #[wasm_bindgen::prelude::wasm_bindgen(typescript_custom_section)]
            const #ident: &'static str = #text;
        }
    });
    quote! { #(#sections)* }
}

/// Emits one TypeScript declaration, gated like the item it belongs to.
pub fn custom_section(
    name: &str,
    cfgs: &[&syn::Attribute],
    text: &str,
) -> TokenStream {
    let ident = format_ident!("__UZU_TYPESCRIPT_{}", name.to_shouty_snake_case());
    quote! {
        #[cfg(feature = "bindings-wasm")]
        #(#cfgs)*
        #[allow(non_upper_case_globals)]
        #[wasm_bindgen::prelude::wasm_bindgen(typescript_custom_section)]
        const #ident: &'static str = #text;
    }
}
