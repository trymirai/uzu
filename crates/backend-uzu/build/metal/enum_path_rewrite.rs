use anyhow::Context;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Expr, Path, parse_quote};

use crate::common::{enum_paths::EnumPaths, expr_rewrite::rewrite_paths_with};

pub fn is_enum_c_type(
    enum_paths: &EnumPaths,
    c_type: &str,
) -> bool {
    let trimmed = c_type.trim_start_matches("const ").trim();
    let Some(uzu_path) = trimmed.strip_prefix("uzu::") else {
        return false;
    };
    let Some(short_name) = uzu_path.rsplit("::").next() else {
        return false;
    };
    let expected_rust_path = format!("crate::backends::common::gpu_types::{uzu_path}");
    enum_paths.full_path_for(short_name) == Some(expected_rust_path.as_str())
}

pub fn rewrite_for_rust(
    enum_paths: &EnumPaths,
    condition: &str,
) -> anyhow::Result<TokenStream> {
    let mut parsed: Expr =
        syn::parse_str(condition).with_context(|| format!("cannot parse rust expression `{condition}`"))?;
    rewrite_paths_with(&mut parsed, |path| qualify_enum_path(path, enum_paths));
    Ok(parsed.into_token_stream())
}

fn qualify_enum_path(
    path: &Path,
    enum_paths: &EnumPaths,
) -> Option<Expr> {
    if path.segments.len() < 2 {
        return None;
    }
    let head = path.segments.first()?;
    let full_path_text = enum_paths.full_path_for(&head.ident.to_string())?;
    let prefix: Path = syn::parse_str(full_path_text).ok()?;
    let suffix: Vec<_> = path.segments.iter().skip(1).cloned().collect();
    Some(parse_quote! { #prefix #( :: #suffix )* })
}
