use anyhow::bail;
use quote::ToTokens;
use syn::{
    BinOp, Expr, ExprBinary, ExprGroup, ExprLit, ExprParen, ExprPath, ExprUnary, ItemConst, Lit, Type, TypePath, UnOp,
};

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

        let is_bool = path.is_ident("bool");
        let value_expression = render_const_expr(&item.expr, is_bool)?.into();

        Ok(Self {
            name: item.ident.to_string().into(),
            ty: path.into_token_stream().to_string().into(),
            value_expression,
        })
    }
}

fn render_const_expr(
    expr: &Expr,
    is_bool: bool,
) -> anyhow::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) => Ok(value.base10_digits().to_string()),
        Expr::Lit(ExprLit {
            lit: Lit::Float(value),
            ..
        }) => Ok(value.base10_digits().to_string()),
        Expr::Lit(ExprLit {
            lit: Lit::Bool(value),
            ..
        }) => Ok(value.value.to_string()),
        Expr::Path(ExprPath {
            qself: None,
            path,
            ..
        }) => {
            let Some(ident) = path.get_ident() else {
                bail!(
                    "GPU constant expression may only reference a bare constant name, found `{}`",
                    path.to_token_stream()
                );
            };
            Ok(ident.to_string())
        },
        Expr::Unary(ExprUnary {
            op,
            expr,
            ..
        }) => {
            let operand = render_const_expr(expr, is_bool)?;
            let op = match op {
                UnOp::Neg(_) => "-",
                UnOp::Not(_) => {
                    if is_bool {
                        "!"
                    } else {
                        "~"
                    }
                },
                other => bail!("Unsupported unary operator in GPU constant expression: `{}`", other.to_token_stream()),
            };
            Ok(format!("{op}{operand}"))
        },
        Expr::Binary(ExprBinary {
            left,
            op,
            right,
            ..
        }) => {
            let left = render_const_expr(left, is_bool)?;
            let right = render_const_expr(right, is_bool)?;
            Ok(format!("{left} {} {right}", binary_op(op)?))
        },
        Expr::Paren(ExprParen {
            expr,
            ..
        }) => Ok(format!("({})", render_const_expr(expr, is_bool)?)),
        Expr::Group(ExprGroup {
            expr,
            ..
        }) => render_const_expr(expr, is_bool),
        other => bail!("Unsupported GPU constant expression: `{}`", other.to_token_stream()),
    }
}

fn binary_op(op: &BinOp) -> anyhow::Result<&'static str> {
    Ok(match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::BitAnd(_) => "&",
        BinOp::BitOr(_) => "|",
        BinOp::BitXor(_) => "^",
        BinOp::Shl(_) => "<<",
        BinOp::Shr(_) => ">>",
        other => bail!("Unsupported binary operator in GPU constant expression: `{}`", other.to_token_stream()),
    })
}
