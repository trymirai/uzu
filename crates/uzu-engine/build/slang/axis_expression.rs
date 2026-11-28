use crate::slang::{SlangArgumentType, SlangKernel};

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

    let expr = rewrite_slang_method(&expr, "saturating_sub", |receiver, argument| {
        format!("(({receiver}) > ({argument}) ? ({receiver}) - ({argument}) : 0)")
    });

    rewrite_slang_if(&expr)
}

fn rewrite_slang_if(expr: &str) -> String {
    let mut out = expr.to_string();
    loop {
        let Some(if_pos) = find_keyword(&out, "if") else {
            return out;
        };
        let cond_start = if_pos + 2;
        let Some(then_open_rel) = out[cond_start..].find('{') else {
            return out;
        };
        let then_open = cond_start + then_open_rel;
        let Some(then_close) = find_matching_brace(&out, then_open) else {
            return out;
        };
        let after_then = out[then_close + 1..].trim_start();
        let Some(after_else) = after_then.strip_prefix("else") else {
            return out;
        };
        if after_else.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
            return out;
        }
        let after_else_offset = out.len() - after_else.len();
        let Some(else_open_rel) = out[after_else_offset..].find('{') else {
            return out;
        };
        let else_open = after_else_offset + else_open_rel;
        let Some(else_close) = find_matching_brace(&out, else_open) else {
            return out;
        };

        let cond = out[cond_start..then_open].trim();
        let then_branch = rewrite_slang_if(out[then_open + 1..then_close].trim());
        let else_branch = rewrite_slang_if(out[else_open + 1..else_close].trim());
        let rewritten = format!("(({cond}) ? ({then_branch}) : ({else_branch}))");
        out = format!("{}{}{}", &out[..if_pos], rewritten, &out[else_close + 1..]);
    }
}

fn find_keyword(
    s: &str,
    kw: &str,
) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut start = 0;
    while let Some(rel) = s[start..].find(kw) {
        let pos = start + rel;
        let before_ok = pos == 0 || !(bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_');
        let after = pos + kw.len();
        let after_ok = after == bytes.len() || !(bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_');
        if before_ok && after_ok {
            return Some(pos);
        }
        start = pos + kw.len();
    }
    None
}

fn find_matching_brace(
    s: &str,
    open_pos: usize,
) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in s[open_pos..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_pos + offset);
                }
            },
            _ => (),
        }
    }
    None
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
