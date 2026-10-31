use crate::slang::{SlangArgumentType, SlangKernel};

/// `expr`, a host expression over a kernel's arguments, as Slang: an optional constant's `unwrap_or` becomes its
/// condition's select, and `div_ceil`, `saturating_mul` and `saturating_sub` become their arithmetic.
pub fn slang_axis_expr(
    expr: &str,
    kernel: &SlangKernel,
) -> String {
    let expr = rewrite_slang_method(expr, "unwrap_or", |receiver, fallback| {
        let condition = kernel
            .arguments
            .iter()
            .find_map(|argument| {
                if argument.name != receiver {
                    return None;
                }
                match &argument.argument_type {
                    SlangArgumentType::Uniform {
                        condition,
                    } => condition.as_deref(),
                    _ => None,
                }
            })
            .expect("unwrap_or receiver must be an optional constant");
        format!("(({condition}) ? ({receiver}) : ({fallback}))")
    });
    let expr = rewrite_slang_method(&expr, "div_ceil", |receiver, argument| {
        format!("(({receiver}) / ({argument}) + uint(({receiver}) % ({argument}) != 0))")
    });
    let expr =
        rewrite_slang_method(&expr, "saturating_mul", |receiver, argument| format!("(({receiver}) * ({argument}))"));

    rewrite_slang_method(&expr, "saturating_sub", |receiver, argument| {
        format!("(({receiver}) > ({argument}) ? ({receiver}) - ({argument}) : 0)")
    })
}

fn rewrite_slang_method(
    expr: &str,
    method: &str,
    rewrite: impl Fn(&str, &str) -> String,
) -> String {
    let pattern = format!(".{method}(");
    let Some(dot) = expr.find(&pattern) else {
        return expr.to_string();
    };

    let receiver_start = expr[..dot].rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |idx| idx + 1);
    let argument_start = dot + pattern.len();
    let mut depth = 1;
    let mut argument_end = argument_start;
    for (offset, ch) in expr[argument_start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    argument_end = argument_start + offset;
                    break;
                }
            },
            _ => (),
        }
    }

    let receiver = &expr[receiver_start..dot];
    let argument = &expr[argument_start..argument_end];
    let rewritten = rewrite(receiver, argument);
    format!("{}{}{}", &expr[..receiver_start], rewritten, &expr[argument_end + 1..])
}
