use anyhow::bail;
use quote::ToTokens;
use syn::{Expr, ExprLit, ItemConst, Lit, Type, TypePath};

#[derive(Debug)]
pub struct GpuTypeConstant {
    pub name: Box<str>,
    pub ty: Box<str>,
    pub value_expression: Box<str>,
}

impl GpuTypeConstant {
    pub fn parse(item: ItemConst) -> anyhow::Result<Self> {
        let Type::Path(TypePath {
            qself: None,
            path,
        }) = *item.ty
        else {
            bail!("GPU constant type must be a path, found {:?}", item.ty);
        };

        let value_expression = match *item.expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => value.base10_digits().into(),
            Expr::Lit(ExprLit {
                lit: Lit::Float(value),
                ..
            }) => value.base10_digits().into(),
            Expr::Lit(ExprLit {
                lit: Lit::Bool(value),
                ..
            }) => value.value.to_string().into(),
            expression => bail!("GPU constant value must be a numeric or boolean literal, found {expression:?}"),
        };

        Ok(Self {
            name: item.ident.to_string().into(),
            ty: path.into_token_stream().to_string().into(),
            value_expression,
        })
    }
}
