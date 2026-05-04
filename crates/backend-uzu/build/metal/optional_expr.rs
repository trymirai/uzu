use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Expr, Path, PathSegment, parse_quote, punctuated::Punctuated};

use super::expr_rewrite::rewrite_paths_with;

const PLACEHOLDER_FN: &str = "__dsl_cpp_uint_cast";

pub fn rewrite_for_cpp(
    condition: &str,
    enum_names: &HashSet<Box<str>>,
) -> String {
    let mut parsed = match syn::parse_str::<Expr>(condition) {
        Ok(expr) => expr,
        Err(_) => return format!("static_cast<bool>({condition})"),
    };

    rewrite_paths_with(&mut parsed, |path| wrap_enum_path_with_uint_cast(path, enum_names));

    let body = clean_cpp_emission(parsed.into_token_stream());
    format!("static_cast<bool>({body})")
}

pub fn rewrite_for_rust(
    condition: &str,
    enum_paths: &HashMap<Box<str>, Box<str>>,
) -> String {
    let mut parsed = match syn::parse_str::<Expr>(condition) {
        Ok(expr) => expr,
        Err(_) => return condition.to_string(),
    };

    rewrite_paths_with(&mut parsed, |path| qualify_enum_path(path, enum_paths));

    parsed.into_token_stream().to_string().replace(" :: ", "::")
}

fn wrap_enum_path_with_uint_cast(
    path: &Path,
    enum_names: &HashSet<Box<str>>,
) -> Option<Expr> {
    if path.segments.len() < 2 {
        return None;
    }
    let first = path.segments.first()?;
    if !enum_names.contains(first.ident.to_string().as_str()) {
        return None;
    }
    let original = path.clone();
    Some(parse_quote! { __dsl_cpp_uint_cast(#original) })
}

fn qualify_enum_path(
    path: &Path,
    enum_paths: &HashMap<Box<str>, Box<str>>,
) -> Option<Expr> {
    if path.segments.len() < 2 {
        return None;
    }
    let first = path.segments.first()?;
    let full_path_str = enum_paths.get(first.ident.to_string().as_str())?;
    let full_path = syn::parse_str::<Path>(full_path_str).ok()?;

    let mut new_segments: Punctuated<PathSegment, syn::Token![::]> = full_path.segments.clone();
    for segment in path.segments.iter().skip(1) {
        new_segments.push(segment.clone());
    }
    let mut new_path = path.clone();
    new_path.leading_colon = full_path.leading_colon;
    new_path.segments = new_segments;
    Some(Expr::Path(syn::ExprPath {
        attrs: vec![],
        qself: None,
        path: new_path,
    }))
}

fn clean_cpp_emission(tokens: TokenStream) -> String {
    tokens
        .to_string()
        .replace(" :: ", "::")
        .replace(&format!("{PLACEHOLDER_FN} ("), "static_cast<uint>(")
        .replace(&format!("{PLACEHOLDER_FN}("), "static_cast<uint>(")
}
