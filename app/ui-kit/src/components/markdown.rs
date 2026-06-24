//! Markdown rendering for chat message bodies, mirroring ui-kit's
//! `MarkdownRenderer`. Renders fenced code blocks (language label + copy button)
//! and prose: headings, unordered/ordered lists, blockquotes, and inline
//! bold/italic/code/links. Tables, math, and syntax highlighting are not yet
//! supported (a later refinement). Links are styled but not yet clickable.

use std::ops::Range;

use gpui::{
    AnyElement, ClipboardItem, CursorStyle, FontStyle, FontWeight, HighlightStyle, IntoElement,
    SharedString, StyledText, div, prelude::*, px,
};

use crate::{
    components::icon::{Icon, IconEl},
    theme::{FONT_MONO, Theme},
};

enum Block {
    Text(String),
    Code { lang: String, code: String },
}

fn parse_blocks(text: &str) -> Vec<Block> {
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
                    blocks.push(Block::Text(buf.trim_matches('\n').to_string()));
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
        blocks.push(Block::Text(buf.trim_matches('\n').to_string()));
    }
    blocks
}

/// Renders `text` as a vertical stack of prose + code blocks. `id_seed` must be
/// unique per message so code-block copy buttons get stable, non-colliding ids.
pub fn markdown(text: &str, theme: &Theme, id_seed: usize) -> AnyElement {
    let mut col = div().flex().flex_col().gap_2();

    for (bi, block) in parse_blocks(text).into_iter().enumerate() {
        col = col.child(match block {
            // Prose: one element per line so single newlines (lists, breaks)
            // survive, while each line still wraps.
            Block::Text(prose) => {
                let mut p = div().flex().flex_col().gap_1();
                for line in prose.split('\n') {
                    if line.trim().is_empty() {
                        p = p.child(div().h(px(6.)));
                    } else {
                        p = p.child(render_line(line, theme));
                    }
                }
                p.into_any_element()
            }
            Block::Code { lang, code } => code_block(&lang, &code, theme, id_seed * 64 + bi),
        });
    }
    col.into_any_element()
}

/// A fenced code block: header (language label + copy button) over a monospace
/// body, mirroring ui-kit's code-block styling.
fn code_block(lang: &str, code: &str, theme: &Theme, uid: usize) -> AnyElement {
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
        .child(div().text_size(px(11.)).text_color(theme.text_muted).child(label))
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
                .text_size(px(11.))
                .text_color(theme.text_muted)
                .hover(|s| s.text_color(theme.text))
                .child(IconEl::new(Icon::Copy, theme.text_muted).size(12.))
                .child("Copy")
                .on_click(move |_, _, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(code_for_copy.clone()));
                }),
        );

    let mut body = div()
        .font_family(FONT_MONO)
        .text_size(px(12.))
        .text_color(theme.text)
        .p_3()
        .flex()
        .flex_col();
    for line in code.split('\n') {
        body = body.child(div().child(line.to_string()));
    }

    div()
        .flex()
        .flex_col()
        .bg(theme.bg_sub)
        .border_1()
        .border_color(theme.border)
        .rounded_lg()
        .overflow_hidden()
        .child(header)
        .child(body)
        .into_any_element()
}

/// Render one prose line: headings (`#`), unordered (`- `/`* `) and ordered
/// (`N. `) list items, blockquotes (`> `), and inline styling.
fn render_line(line: &str, theme: &Theme) -> AnyElement {
    let trimmed = line.trim_start();

    // Heading: leading #'s followed by a space.
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
        let (text, runs) = inline_runs(trimmed[hashes + 1..].trim_start(), theme);
        return div()
            .text_lg()
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(theme.text)
            .child(StyledText::new(text).with_highlights(runs))
            .into_any_element();
    }

    // Blockquote.
    if let Some(rest) = trimmed.strip_prefix("> ").or_else(|| trimmed.strip_prefix(">")) {
        let (text, runs) = inline_runs(rest.trim_start(), theme);
        return div()
            .border_l_2()
            .border_color(theme.border)
            .pl_3()
            .text_color(theme.text_muted)
            .child(StyledText::new(text).with_highlights(runs))
            .into_any_element();
    }

    // Unordered list item.
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        let (text, runs) = inline_runs(rest, theme);
        return div()
            .flex()
            .gap_2()
            .child(div().text_color(theme.text_muted).child("•"))
            .child(StyledText::new(text).with_highlights(runs))
            .into_any_element();
    }

    // Ordered list item: `N. `.
    let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && trimmed[digits..].starts_with(". ") {
        let num = trimmed[..digits].to_string();
        let (text, runs) = inline_runs(&trimmed[digits + 2..], theme);
        return div()
            .flex()
            .gap_2()
            .child(div().text_color(theme.text_muted).child(format!("{num}.")))
            .child(StyledText::new(text).with_highlights(runs))
            .into_any_element();
    }

    let (text, runs) = inline_runs(line, theme);
    StyledText::new(text).with_highlights(runs).into_any_element()
}

/// Parse `**bold**`, `*italic*`, `` `code` ``, and `[text](url)` links into plain
/// text + styled ranges. Unclosed markers (common mid-stream) consume to end.
fn inline_runs(line: &str, theme: &Theme) -> (String, Vec<(Range<usize>, HighlightStyle)>) {
    let mut out = String::new();
    let mut runs: Vec<(Range<usize>, HighlightStyle)> = Vec::new();
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
                runs.push((
                    start..out.len(),
                    HighlightStyle {
                        background_color: Some(theme.bg_sub),
                        color: Some(theme.text),
                        ..Default::default()
                    },
                ));
            }
            '[' => {
                // [text](url) → blue link text (url dropped; not yet clickable).
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
                    chars.next(); // '('
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == ')' {
                            break;
                        }
                    }
                    let start = out.len();
                    out.push_str(&link_text);
                    runs.push((
                        start..out.len(),
                        HighlightStyle {
                            color: Some(theme.info),
                            ..Default::default()
                        },
                    ));
                } else {
                    // Not a link: emit literally.
                    out.push('[');
                    out.push_str(&link_text);
                    if found_close {
                        out.push(']');
                    }
                }
            }
            '*' if chars.peek() == Some(&'*') => {
                chars.next(); // second '*'
                let start = out.len();
                loop {
                    match chars.next() {
                        Some('*') if chars.peek() == Some(&'*') => {
                            chars.next();
                            break;
                        }
                        Some(ch) => out.push(ch),
                        None => break,
                    }
                }
                runs.push((
                    start..out.len(),
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        ..Default::default()
                    },
                ));
            }
            '*' | '_' => {
                let marker = c;
                let start = out.len();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if nc == marker {
                        break;
                    }
                    out.push(nc);
                }
                runs.push((
                    start..out.len(),
                    HighlightStyle {
                        font_style: Some(FontStyle::Italic),
                        ..Default::default()
                    },
                ));
            }
            _ => out.push(c),
        }
    }

    (out, runs)
}
