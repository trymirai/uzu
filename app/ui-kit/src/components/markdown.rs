//! Minimal markdown rendering for chat message bodies. Splits text into fenced
//! code blocks (rendered monospace in a card) and prose (line-preserving). Inline
//! styling (bold/italic/links) is a later refinement.

use std::ops::Range;

use gpui::{
    AnyElement, FontStyle, FontWeight, HighlightStyle, IntoElement, StyledText, div, prelude::*, px,
};

use crate::theme::{FONT_MONO, Theme};

enum Block {
    Text(String),
    Code(String),
}

fn parse_blocks(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut in_code = false;
    let mut buf = String::new();

    let flush = |blocks: &mut Vec<Block>, buf: &mut String, in_code: bool| {
        if in_code {
            blocks.push(Block::Code(buf.trim_end_matches('\n').to_string()));
        } else if !buf.trim().is_empty() {
            blocks.push(Block::Text(buf.trim_matches('\n').to_string()));
        }
        buf.clear();
    };

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            flush(&mut blocks, &mut buf, in_code);
            in_code = !in_code;
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush(&mut blocks, &mut buf, in_code);
    blocks
}

/// Renders `text` as a vertical stack of prose + code blocks.
pub fn markdown(text: &str, theme: &Theme) -> AnyElement {
    let mut col = div().flex().flex_col().gap_2();

    for block in parse_blocks(text) {
        col = col.child(match block {
            // Prose: one element per line so single newlines (lists, breaks)
            // survive, while each line still wraps. Inline bold/italic/code +
            // headings + bullets are rendered per line.
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
            Block::Code(code) => {
                let mut body = div()
                    .font_family(FONT_MONO)
                    .text_size(px(12.))
                    .text_color(theme.text)
                    .bg(theme.bg_sub)
                    .border_1()
                    .border_color(theme.border)
                    .rounded_lg()
                    .p_3()
                    .flex()
                    .flex_col();
                for line in code.split('\n') {
                    body = body.child(div().child(line.to_string()));
                }
                body.into_any_element()
            }
        });
    }
    col.into_any_element()
}

/// Render one prose line: headings (`#`), bullets (`- `/`* `), and inline
/// bold/italic/code.
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

    // Bullet list item.
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        let (text, runs) = inline_runs(rest, theme);
        return div()
            .flex()
            .gap_2()
            .child(div().text_color(theme.text_muted).child("•"))
            .child(StyledText::new(text).with_highlights(runs))
            .into_any_element();
    }

    let (text, runs) = inline_runs(line, theme);
    StyledText::new(text).with_highlights(runs).into_any_element()
}

/// Parse `**bold**`, `*italic*`, and `` `code` `` into plain text + styled
/// ranges (byte offsets into the returned string). Unclosed markers (common
/// mid-stream) consume to end of line.
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
