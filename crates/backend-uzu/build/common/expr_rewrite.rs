use syn::{
    Expr, ExprPath, LitInt, Path,
    visit_mut::{self, VisitMut},
};

pub fn rewrite_paths_with<F>(
    expr: &mut Expr,
    rewriter: F,
) where
    F: FnMut(&Path) -> Option<Expr>,
{
    Walker {
        rewriter,
    }
    .visit_expr_mut(expr);
}

struct Walker<F: FnMut(&Path) -> Option<Expr>> {
    rewriter: F,
}

impl<F: FnMut(&Path) -> Option<Expr>> VisitMut for Walker<F> {
    fn visit_expr_mut(
        &mut self,
        expr: &mut Expr,
    ) {
        visit_mut::visit_expr_mut(self, expr);

        if let Expr::Path(ExprPath {
            path,
            ..
        }) = expr
            && let Some(replacement) = (self.rewriter)(path)
        {
            *expr = replacement;
        }
    }

    fn visit_lit_int_mut(
        &mut self,
        literal: &mut LitInt,
    ) {
        let suffix = literal.suffix();
        let rust_suffix = match suffix {
            "u" | "U" => "u32",
            _ => return,
        };

        let text = literal.to_string();
        let digits = &text[..text.len() - suffix.len()];
        *literal = LitInt::new(&format!("{digits}{rust_suffix}"), literal.span());
    }
}
