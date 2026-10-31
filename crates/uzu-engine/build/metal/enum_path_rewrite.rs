use anyhow::Context;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::Expr;

use crate::common::{
    expr_rewrite::{qualify_enum_path, rewrite_paths_with},
    gpu_type_paths::{GpuTypeKind, GpuTypePaths},
};

pub fn gpu_type_kind_for_c_type(
    gpu_type_paths: &GpuTypePaths,
    c_type: &str,
) -> Option<GpuTypeKind> {
    let trimmed = c_type.trim_start_matches("const ").trim();
    super::gpu_types::rust_path(gpu_type_paths, trimmed)?;
    let short_name = trimmed.rsplit("::").next()?;
    gpu_type_paths.kind_for(short_name)
}

pub fn is_enum_c_type(
    gpu_type_paths: &GpuTypePaths,
    c_type: &str,
) -> bool {
    matches!(gpu_type_kind_for_c_type(gpu_type_paths, c_type), Some(GpuTypeKind::Enum | GpuTypeKind::OptionSet))
}

pub fn rewrite_for_rust(
    gpu_type_paths: &GpuTypePaths,
    condition: &str,
) -> anyhow::Result<TokenStream> {
    let mut parsed: Expr =
        syn::parse_str(condition).with_context(|| format!("cannot parse rust expression `{condition}`"))?;
    rewrite_paths_with(&mut parsed, |path| qualify_enum_path(path, gpu_type_paths));
    Ok(parsed.into_token_stream())
}
