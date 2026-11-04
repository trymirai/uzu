use convert_case::{Case, Casing};
use heck::ToShoutySnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// The JavaScript name of a field, parameter, or method, converted exactly as napi-derive does
/// (same `convert_case` version), so that e.g. `crc32c` becomes `crc32C` in both packages.
pub fn js_name(rust_name: &str) -> String {
    rust_name.trim_start_matches("r#").trim_start_matches('_').to_case(Case::Camel)
}

/// The TypeScript type napi declares for a Rust type. `Option` maps to `T | null`, as for return
/// values; fields and parameters handle a top-level `Option` themselves.
pub fn typescript_type(
    ty: &syn::Type,
    self_name: &str,
) -> String {
    match ty {
        syn::Type::Reference(reference) => typescript_type(&reference.elem, self_name),
        syn::Type::Paren(paren) => typescript_type(&paren.elem, self_name),
        syn::Type::Group(group) => typescript_type(&group.elem, self_name),
        syn::Type::Tuple(tuple) if tuple.elems.is_empty() => "void".to_string(),
        syn::Type::Slice(slice) => format!("Array<{}>", typescript_type(&slice.elem, self_name)),
        syn::Type::Path(path) => path_type(path, self_name),
        _ => "any".to_string(),
    }
}

fn path_type(
    path: &syn::TypePath,
    self_name: &str,
) -> String {
    let Some(segment) = path.path.segments.last() else {
        return "any".to_string();
    };
    let arguments = generic_arguments(segment);
    let argument = |index: usize| {
        arguments.get(index).map_or_else(|| "any".to_string(), |argument| typescript_type(argument, self_name))
    };
    match segment.ident.to_string().as_str() {
        "bool" => "boolean".to_string(),
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" | "f32" | "f64" => {
            "number".to_string()
        },
        "String" | "str" | "char" => "string".to_string(),
        "Option" => format!("{} | null", argument(0)),
        "Vec" | "VecDeque" => format!("Array<{}>", argument(0)),
        "HashMap" | "BTreeMap" | "IndexMap" => format!("Record<{}, {}>", argument(0), argument(1)),
        "Box" | "Arc" | "Rc" | "Result" => argument(0),
        "JsValue" => "any".to_string(),
        "Self" => self_name.to_string(),
        name => name.to_string(),
    }
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

fn generic_arguments(segment: &syn::PathSegment) -> Vec<&syn::Type> {
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Vec::new();
    };
    arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

fn wrapped_inner<'ty>(
    ty: &'ty syn::Type,
    wrapper: &str,
) -> Option<&'ty syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != wrapper {
        return None;
    }
    generic_arguments(segment).into_iter().next()
}

pub fn option_inner(ty: &syn::Type) -> Option<&syn::Type> {
    wrapped_inner(ty, "Option")
}

pub fn result_inner(ty: &syn::Type) -> Option<&syn::Type> {
    wrapped_inner(ty, "Result")
}

pub fn is_unit(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Tuple(tuple) if tuple.elems.is_empty())
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
