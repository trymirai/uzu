use anyhow::bail;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{BinOp, Expr, ExprBinary, ExprLit, ExprParen, ExprPath, ExprUnary, Lit, Path, UnOp, parse_quote};

use crate::common::{enum_paths::EnumPaths, expr_rewrite::rewrite_paths_with, gpu_types::GpuTypes};

pub struct EnumPathRewriter {
    enum_paths: EnumPaths,
}

impl EnumPathRewriter {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> Self {
        Self {
            enum_paths: EnumPaths::from_gpu_types(gpu_types),
        }
    }

    pub fn is_enum_c_type(
        &self,
        c_type: &str,
    ) -> bool {
        let trimmed = c_type.trim_start_matches("const ").trim();
        let type_name = trimmed.rsplit("::").next().unwrap_or(trimmed);
        self.enum_paths.contains(type_name)
    }

    pub fn rewrite_for_metal(
        &self,
        condition: &str,
    ) -> String {
        let parsed = match syn::parse_str::<Expr>(condition) {
            Ok(expr) => expr,
            Err(_) => return format!("static_cast<bool>({condition})"),
        };
        match emit_metal_expr(&parsed, self) {
            Ok(body) => format!("static_cast<bool>({body})"),
            Err(_) => format!("static_cast<bool>({condition})"),
        }
    }

    pub fn rewrite_for_rust(
        &self,
        condition: &str,
    ) -> anyhow::Result<TokenStream> {
        let mut parsed: Expr = syn::parse_str(condition)
            .map_err(|error| anyhow::anyhow!("cannot parse rust expression `{}`: {}", condition, error))?;
        let enum_paths = &self.enum_paths;
        rewrite_paths_with(&mut parsed, |path| qualify_enum_path(path, enum_paths));
        Ok(parsed.into_token_stream())
    }
}

fn emit_metal_expr(
    expr: &Expr,
    rewriter: &EnumPathRewriter,
) -> anyhow::Result<String> {
    match expr {
        Expr::Path(expr_path) => Ok(emit_metal_path(expr_path, rewriter)),
        Expr::Binary(ExprBinary {
            op,
            left,
            right,
            ..
        }) => {
            let operator = match op {
                BinOp::Eq(_) => "==",
                BinOp::Ne(_) => "!=",
                BinOp::And(_) => "&&",
                BinOp::Or(_) => "||",
                other => bail!("unsupported Metal binary operator: {:?}", other),
            };
            let left_emitted = emit_metal_expr(left, rewriter)?;
            let right_emitted = emit_metal_expr(right, rewriter)?;
            Ok(format!("{left_emitted} {operator} {right_emitted}"))
        },
        Expr::Unary(ExprUnary {
            op,
            expr: inner,
            ..
        }) => {
            let operator = match op {
                UnOp::Not(_) => "!",
                other => bail!("unsupported Metal unary operator: {:?}", other),
            };
            let inner_emitted = emit_metal_expr(inner, rewriter)?;
            Ok(format!("{operator}{inner_emitted}"))
        },
        Expr::Paren(ExprParen {
            expr: inner,
            ..
        }) => Ok(format!("({})", emit_metal_expr(inner, rewriter)?)),
        Expr::Lit(ExprLit {
            lit,
            ..
        }) => match lit {
            Lit::Bool(boolean) => Ok(boolean.value.to_string()),
            Lit::Int(integer) => Ok(integer.base10_digits().to_string()),
            other => bail!("unsupported Metal literal: {:?}", other),
        },
        other => bail!("unsupported Metal expression kind: {:?}", other),
    }
}

fn emit_metal_path(
    expr_path: &ExprPath,
    rewriter: &EnumPathRewriter,
) -> String {
    let path_text = path_to_metal_string(&expr_path.path);
    let head_is_enum =
        expr_path.path.segments.first().is_some_and(|segment| rewriter.enum_paths.contains(&segment.ident.to_string()));
    if head_is_enum && expr_path.path.segments.len() >= 2 {
        format!("static_cast<uint>({path_text})")
    } else {
        path_text
    }
}

fn path_to_metal_string(path: &Path) -> String {
    let leading = if path.leading_colon.is_some() {
        "::"
    } else {
        ""
    };
    let segments = path.segments.iter().map(|segment| segment.ident.to_string()).collect::<Vec<_>>().join("::");
    format!("{leading}{segments}")
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
