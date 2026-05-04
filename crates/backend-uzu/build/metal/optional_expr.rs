use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Expr, Path, PathSegment, parse_quote, punctuated::Punctuated};

use super::expr_rewrite::rewrite_paths_with;
use crate::common::gpu_types::{GpuType, GpuTypes};

const PLACEHOLDER_FN: &str = "__dsl_metal_uint_cast";

pub enum RewriteTarget<'a> {
    /// Emit a Metal C++ expression. Wraps each enum path with
    /// `static_cast<uint>(...)` so the underlying value can be
    /// compared as an integer, and wraps the whole result in
    /// `static_cast<bool>(...)`.
    Metal {
        enum_paths: &'a HashMap<Box<str>, Box<str>>,
    },
    /// Emit a Rust expression. Qualifies each enum path with its
    /// full crate path so the binding compiles outside the gpu_types module.
    Rust {
        enum_paths: &'a HashMap<Box<str>, Box<str>>,
    },
}

pub struct OptionalExprRewriter {
    enum_paths: HashMap<Box<str>, Box<str>>,
}

impl OptionalExprRewriter {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> Self {
        let mut enum_paths = HashMap::new();
        for file in gpu_types.files.iter() {
            for ty in file.types.iter() {
                if let GpuType::Enum(e) = ty {
                    let full_path =
                        format!("crate::backends::common::gpu_types::{}::{}", file.name, e.name).into_boxed_str();
                    enum_paths.insert(e.name.clone(), full_path);
                }
            }
        }
        Self {
            enum_paths,
        }
    }

    pub fn metal_target(&self) -> RewriteTarget<'_> {
        RewriteTarget::Metal {
            enum_paths: &self.enum_paths,
        }
    }

    pub fn rust_target(&self) -> RewriteTarget<'_> {
        RewriteTarget::Rust {
            enum_paths: &self.enum_paths,
        }
    }
}

pub fn rewrite(
    condition: &str,
    target: RewriteTarget<'_>,
) -> String {
    let mut parsed = match syn::parse_str::<Expr>(condition) {
        Ok(expr) => expr,
        Err(_) => {
            return match target {
                RewriteTarget::Metal {
                    ..
                } => format!("static_cast<bool>({condition})"),
                RewriteTarget::Rust {
                    ..
                } => condition.to_string(),
            };
        },
    };

    match target {
        RewriteTarget::Metal {
            enum_paths,
        } => {
            rewrite_paths_with(&mut parsed, |path| wrap_enum_path_with_uint_cast(path, enum_paths));
            let body = clean_metal_emission(parsed.into_token_stream());
            format!("static_cast<bool>({body})")
        },
        RewriteTarget::Rust {
            enum_paths,
        } => {
            rewrite_paths_with(&mut parsed, |path| qualify_enum_path(path, enum_paths));
            parsed.into_token_stream().to_string().replace(" :: ", "::")
        },
    }
}

fn wrap_enum_path_with_uint_cast(
    path: &Path,
    enum_paths: &HashMap<Box<str>, Box<str>>,
) -> Option<Expr> {
    if path.segments.len() < 2 {
        return None;
    }
    let first = path.segments.first()?;
    if !enum_paths.contains_key(first.ident.to_string().as_str()) {
        return None;
    }
    let original = path.clone();
    Some(parse_quote! { __dsl_metal_uint_cast(#original) })
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

fn clean_metal_emission(tokens: TokenStream) -> String {
    tokens
        .to_string()
        .replace(" :: ", "::")
        .replace(&format!("{PLACEHOLDER_FN} ("), "static_cast<uint>(")
        .replace(&format!("{PLACEHOLDER_FN}("), "static_cast<uint>(")
}
