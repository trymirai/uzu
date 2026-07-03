use std::ops::Range;

use gpui::{
    AnyElement, ClipboardItem, CursorStyle, FontStyle, FontWeight, HighlightStyle, InteractiveText, IntoElement,
    SharedString, StyledText, div, prelude::*, px,
};

use crate::{
    components::icon::{Icon, IconEl},
    theme::{FONT_MONO, Theme},
    tokens,
};

#[derive(Clone)]
pub struct ParsedMarkdown {
    blocks: Vec<Block>,
}

#[derive(Clone)]
enum Block {
    Prose(Vec<Line>),
    Code {
        lang: String,
        code: String,
    },
}

#[derive(Clone)]
enum Line {
    Blank,
    Heading {
        level: u8,
        inline: Inline,
    },
    Quote(Inline),
    Bullet(Inline),
    Ordered {
        num: String,
        inline: Inline,
    },
    Plain(Inline),
}

#[derive(Clone, Default)]
struct Inline {
    text: String,
    runs: Vec<(Range<usize>, RunKind)>,
    links: Vec<(Range<usize>, String)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunKind {
    Code,
    Bold,
    Italic,
    Link,
}

pub fn parse(text: &str) -> ParsedMarkdown {
    let mut blocks = Vec::new();
    let mut in_code = false;
    let mut lang = String::new();
    let mut buf = String::new();

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if in_code {
                blocks.push(Block::Code {
                    lang: std::mem::take(&mut lang),
                    code: buf.trim_end_matches('\n').to_string(),
                });
                buf.clear();
                in_code = false;
            } else {
                if !buf.trim().is_empty() {
                    blocks.push(parse_prose(&buf));
                }
                buf.clear();
                lang = line.trim_start().trim_start_matches('`').trim().to_string();
                in_code = true;
            }
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    if in_code {
        blocks.push(Block::Code {
            lang,
            code: buf.trim_end_matches('\n').to_string(),
        });
    } else if !buf.trim().is_empty() {
        blocks.push(parse_prose(&buf));
    }
    ParsedMarkdown {
        blocks,
    }
}

fn parse_prose(prose: &str) -> Block {
    let lines = prose.trim_matches('\n').split('\n').map(parse_line).collect();
    Block::Prose(lines)
}

fn parse_line(line: &str) -> Line {
    if line.trim().is_empty() {
        return Line::Blank;
    }
    let trimmed = line.trim_start();

    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
        return Line::Heading {
            level: hashes as u8,
            inline: parse_inline(trimmed[hashes + 1..].trim_start()),
        };
    }

    if let Some(rest) = trimmed.strip_prefix("> ").or_else(|| trimmed.strip_prefix(">")) {
        return Line::Quote(parse_inline(rest.trim_start()));
    }

    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return Line::Bullet(parse_inline(rest));
    }

    let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && trimmed[digits..].starts_with(". ") {
        return Line::Ordered {
            num: trimmed[..digits].to_string(),
            inline: parse_inline(&trimmed[digits + 2..]),
        };
    }

    Line::Plain(parse_inline(line))
}

pub fn render(
    parsed: &ParsedMarkdown,
    theme: &Theme,
    id_seed: usize,
) -> AnyElement {
    let mut col = div().flex().flex_col().w_full().min_w_0().gap_2();
    let mut line_no = 0usize;

    for (bi, block) in parsed.blocks.iter().enumerate() {
        col = col.child(match block {
            Block::Prose(lines) => {
                let mut p = div().flex().flex_col().w_full().min_w_0().gap_1();
                for line in lines {
                    if matches!(line, Line::Blank) {
                        p = p.child(div().h(px(6.)));
                    } else {
                        let lid = id_seed.wrapping_mul(100_000).wrapping_add(line_no);
                        line_no += 1;
                        p = p.child(render_line(line, theme, lid));
                    }
                }
                p.into_any_element()
            },
            Block::Code {
                lang,
                code,
            } => code_block(lang, code, theme, id_seed * 64 + bi),
        });
    }
    col.into_any_element()
}

pub fn markdown(
    text: &str,
    theme: &Theme,
    id_seed: usize,
) -> AnyElement {
    render(&parse(text), theme, id_seed)
}

fn code_block(
    lang: &str,
    code: &str,
    theme: &Theme,
    uid: usize,
) -> AnyElement {
    let label = if lang.trim().is_empty() {
        "code".to_string()
    } else {
        lang.trim().to_string()
    };
    let code_for_copy = code.to_string();

    let header = div()
        .flex()
        .items_center()
        .justify_between()
        .px_3()
        .py_1()
        .border_b_1()
        .border_color(theme.border)
        .child(div().text_size(tokens::font::CAPTION).text_color(theme.text_muted).child(label))
        .child(
            div()
                .id(SharedString::from(format!("md-copy-{uid}")))
                .flex()
                .items_center()
                .gap_1()
                .px_2()
                .py_0p5()
                .rounded_md()
                .cursor(CursorStyle::PointingHand)
                .text_size(tokens::font::CAPTION)
                .text_color(theme.text_muted)
                .hover(|s| s.text_color(theme.text))
                .child(IconEl::new(Icon::Copy, theme.text_muted).size(tokens::icon::XS))
                .child("Copy")
                .on_click(move |_, _, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(code_for_copy.clone()));
                }),
        );

    let mut body = div()
        .font_family(FONT_MONO)
        .text_size(tokens::font::SMALL)
        .text_color(theme.text)
        .p_3()
        .flex()
        .flex_col()
        .w_full()
        .min_w_0()
        .overflow_hidden();
    for line in code.split('\n') {
        body = body.child(div().child(line.to_string()));
    }

    div()
        .flex()
        .flex_col()
        .w_full()
        .min_w_0()
        .bg(theme.bg_sub)
        .border_1()
        .border_color(theme.border)
        .rounded_lg()
        .overflow_hidden()
        .child(header)
        .child(body)
        .into_any_element()
}

fn prose_wrap(el: AnyElement) -> AnyElement {
    div().w_full().min_w_0().overflow_hidden().child(el).into_any_element()
}

fn render_line(
    line: &Line,
    theme: &Theme,
    id: usize,
) -> AnyElement {
    match line {
        Line::Blank => div().h(px(6.)).into_any_element(),
        Line::Heading {
            level,
            inline,
        } => {
            let size = match level {
                1 => tokens::font::H1,
                2 => tokens::font::H2,
                3 => tokens::font::H3,
                _ => tokens::font::BODY,
            };
            let weight = if *level <= 3 {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::MEDIUM
            };
            prose_wrap(
                div()
                    .text_size(size)
                    .font_weight(weight)
                    .text_color(theme.text)
                    .child(inline_el(inline, theme, id))
                    .into_any_element(),
            )
        },
        Line::Quote(inline) => prose_wrap(
            div()
                .border_l_2()
                .border_color(theme.border)
                .pl_3()
                .text_color(theme.text_muted)
                .child(inline_el(inline, theme, id))
                .into_any_element(),
        ),
        Line::Bullet(inline) => div()
            .flex()
            .w_full()
            .min_w_0()
            .gap_2()
            .child(div().flex_none().text_color(theme.text_muted).child("•"))
            .child(div().flex_1().min_w_0().overflow_hidden().child(inline_el(inline, theme, id)))
            .into_any_element(),
        Line::Ordered {
            num,
            inline,
        } => div()
            .flex()
            .w_full()
            .min_w_0()
            .gap_2()
            .child(div().flex_none().text_color(theme.text_muted).child(format!("{num}.")))
            .child(div().flex_1().min_w_0().overflow_hidden().child(inline_el(inline, theme, id)))
            .into_any_element(),
        Line::Plain(inline) => prose_wrap(inline_el(inline, theme, id)),
    }
}

fn style_for(
    kind: RunKind,
    theme: &Theme,
) -> HighlightStyle {
    match kind {
        RunKind::Code => HighlightStyle {
            background_color: Some(theme.bg_sub),
            color: Some(theme.text),
            ..Default::default()
        },
        RunKind::Bold => HighlightStyle {
            font_weight: Some(FontWeight::BOLD),
            ..Default::default()
        },
        RunKind::Italic => HighlightStyle {
            font_style: Some(FontStyle::Italic),
            ..Default::default()
        },
        RunKind::Link => HighlightStyle {
            color: Some(theme.info),
            ..Default::default()
        },
    }
}

fn inline_el(
    inline: &Inline,
    theme: &Theme,
    id: usize,
) -> AnyElement {
    let runs: Vec<(Range<usize>, HighlightStyle)> =
        inline.runs.iter().map(|(r, k)| (r.clone(), style_for(*k, theme))).collect();
    let styled = StyledText::new(inline.text.clone()).with_highlights(runs);
    if inline.links.is_empty() {
        return styled.into_any_element();
    }
    let urls: Vec<String> = inline.links.iter().map(|(_, u)| u.clone()).collect();
    let ranges: Vec<Range<usize>> = inline.links.iter().map(|(r, _)| r.clone()).collect();
    InteractiveText::new(SharedString::from(format!("md-link-{id}")), styled)
        .on_click(ranges, move |ix, _, cx| {
            if let Some(url) = urls.get(ix) {
                cx.open_url(url);
            }
        })
        .into_any_element()
}

fn parse_inline(line: &str) -> Inline {
    let mut out = String::new();
    let mut runs: Vec<(Range<usize>, RunKind)> = Vec::new();
    let mut links: Vec<(Range<usize>, String)> = Vec::new();
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '`' => {
                let start = out.len();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if nc == '`' {
                        break;
                    }
                    out.push(nc);
                }
                runs.push((start..out.len(), RunKind::Code));
            },
            '[' => {
                let mut link_text = String::new();
                let mut found_close = false;
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if nc == ']' {
                        found_close = true;
                        break;
                    }
                    link_text.push(nc);
                }
                if found_close && chars.peek() == Some(&'(') {
                    chars.next();
                    let mut url = String::new();
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == ')' {
                            break;
                        }
                        url.push(nc);
                    }
                    let start = out.len();
                    out.push_str(&link_text);
                    runs.push((start..out.len(), RunKind::Link));
                    links.push((start..out.len(), url));
                } else {
                    out.push('[');
                    out.push_str(&link_text);
                    if found_close {
                        out.push(']');
                    }
                }
            },
            '*' if chars.peek() == Some(&'*') => {
                chars.next();
                let start = out.len();
                loop {
                    match chars.next() {
                        Some('*') if chars.peek() == Some(&'*') => {
                            chars.next();
                            break;
                        },
                        Some(ch) => out.push(ch),
                        None => break,
                    }
                }
                runs.push((start..out.len(), RunKind::Bold));
            },
            '*' => {
                let start = out.len();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if nc == '*' {
                        break;
                    }
                    out.push(nc);
                }
                runs.push((start..out.len(), RunKind::Italic));
            },
            '_' => {
                if out.chars().last().is_some_and(char::is_alphanumeric) {
                    out.push('_');
                } else {
                    let mut content = String::new();
                    let mut closed = false;
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == '_' && !chars.peek().is_some_and(|n| n.is_alphanumeric()) {
                            closed = true;
                            break;
                        }
                        content.push(nc);
                    }
                    if closed {
                        let start = out.len();
                        out.push_str(&content);
                        runs.push((start..out.len(), RunKind::Italic));
                    } else {
                        out.push('_');
                        out.push_str(&content);
                    }
                }
            },
            _ => out.push(c),
        }
    }

    Inline {
        text: out,
        runs,
        links,
    }
}
