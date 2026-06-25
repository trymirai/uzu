//! Markdown for chat bodies, mirroring ui-kit's `MarkdownRenderer`: code blocks
//! (with copy), headings, lists, blockquotes, inline bold/italic/code/links
//! (clickable). No tables/math/syntax-highlight yet.

use std::ops::Range;

use gpui::{
    AnyElement, ClipboardItem, CursorStyle, FontStyle, FontWeight, HighlightStyle, InteractiveText,
    IntoElement, SharedString, StyledText, div, prelude::*, px,
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
    let mut col = div().flex().flex_col().w_full().min_w_0().gap_2();
    let mut line_no = 0usize;

    for (bi, block) in parse_blocks(text).into_iter().enumerate() {
        col = col.child(match block {
            // Prose: one element per line so single newlines (lists, breaks)
            // survive, while each line still wraps.
            Block::Text(prose) => {
                let mut p = div().flex().flex_col().w_full().min_w_0().gap_1();
                for line in prose.split('\n') {
                    if line.trim().is_empty() {
                        p = p.child(div().h(px(6.)));
                    } else {
                        let lid = id_seed.wrapping_mul(100_000).wrapping_add(line_no);
                        line_no += 1;
                        p = p.child(render_line(line, theme, lid));
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

/// Wrap prose so long lines wrap inside the chat column instead of overflowing.
fn prose_wrap(el: AnyElement) -> AnyElement {
    div()
        .w_full()
        .min_w_0()
        .overflow_hidden()
        .child(el)
        .into_any_element()
}

/// Render one prose line: headings (`#`), unordered (`- `/`* `) and ordered
/// (`N. `) list items, blockquotes (`> `), and inline styling. `id` keeps link
/// hit-targets unique.
fn render_line(line: &str, theme: &Theme, id: usize) -> AnyElement {
    let trimmed = line.trim_start();

    // Heading: leading #'s followed by a space.
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
        return prose_wrap(
            div()
                .text_lg()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(theme.text)
                .child(text_el(trimmed[hashes + 1..].trim_start(), theme, id))
                .into_any_element(),
        );
    }

    // Blockquote.
    if let Some(rest) = trimmed.strip_prefix("> ").or_else(|| trimmed.strip_prefix(">")) {
        return prose_wrap(
            div()
                .border_l_2()
                .border_color(theme.border)
                .pl_3()
                .text_color(theme.text_muted)
                .child(text_el(rest.trim_start(), theme, id))
                .into_any_element(),
        );
    }

    // Unordered list item.
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return div()
            .flex()
            .w_full()
            .min_w_0()
            .gap_2()
            .child(div().flex_none().text_color(theme.text_muted).child("•"))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .child(text_el(rest, theme, id)),
            )
            .into_any_element();
    }

    // Ordered list item: `N. `.
    let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && trimmed[digits..].starts_with(". ") {
        let num = trimmed[..digits].to_string();
        return div()
            .flex()
            .w_full()
            .min_w_0()
            .gap_2()
            .child(div().flex_none().text_color(theme.text_muted).child(format!("{num}.")))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .child(text_el(&trimmed[digits + 2..], theme, id)),
            )
            .into_any_element();
    }

    prose_wrap(text_el(line, theme, id))
}

/// Inline text for one line: a `StyledText`, upgraded to an `InteractiveText`
/// (clickable links opening in the browser) when it contains `[text](url)`.
fn text_el(line: &str, theme: &Theme, id: usize) -> AnyElement {
    let (text, runs, links) = inline_runs(line, theme);
    let styled = StyledText::new(text).with_highlights(runs);
    if links.is_empty() {
        return styled.into_any_element();
    }
    let urls: Vec<String> = links.iter().map(|(_, u)| u.clone()).collect();
    let ranges: Vec<Range<usize>> = links.into_iter().map(|(r, _)| r).collect();
    InteractiveText::new(SharedString::from(format!("md-link-{id}")), styled)
        .on_click(ranges, move |ix, _, cx| {
            if let Some(url) = urls.get(ix) {
                cx.open_url(url);
            }
        })
        .into_any_element()
}

/// Parse `**bold**`, `*italic*`, `` `code` ``, and `[text](url)` links into plain
/// text + styled ranges. Unclosed markers (common mid-stream) consume to end.
#[allow(clippy::type_complexity)]
fn inline_runs(
    line: &str,
    theme: &Theme,
) -> (String, Vec<(Range<usize>, HighlightStyle)>, Vec<(Range<usize>, String)>) {
    let mut out = String::new();
    let mut runs: Vec<(Range<usize>, HighlightStyle)> = Vec::new();
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
                    runs.push((
                        start..out.len(),
                        HighlightStyle {
                            color: Some(theme.info),
                            ..Default::default()
                        },
                    ));
                    links.push((start..out.len(), url));
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

    (out, runs, links)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classify a highlight run by its style alone, so snapshots stay
    /// theme- and color-independent.
    fn kind(style: &HighlightStyle) -> &'static str {
        if style.background_color.is_some() {
            "code"
        } else if style.font_weight == Some(FontWeight::BOLD) {
            "bold"
        } else if style.font_style == Some(FontStyle::Italic) {
            "italic"
        } else if style.color.is_some() {
            "link"
        } else {
            "plain"
        }
    }

    /// Readable structural dump of one inline line.
    fn describe_inline(line: &str) -> String {
        let (text, runs, links) = inline_runs(line, &Theme::dark());
        let mut out = format!("text: {text:?}\n");
        for (range, style) in &runs {
            out += &format!(
                "  run {}..{} {} {:?}\n",
                range.start,
                range.end,
                kind(style),
                &text[range.clone()]
            );
        }
        for (range, url) in &links {
            out += &format!("  link {}..{} -> {url}\n", range.start, range.end);
        }
        out
    }

    fn describe_blocks(text: &str) -> String {
        let mut out = String::new();
        for block in parse_blocks(text) {
            match block {
                Block::Text(t) => out += &format!("[text] {t:?}\n"),
                Block::Code { lang, code } => out += &format!("[code:{lang}] {code:?}\n"),
            }
        }
        out
    }

    #[test]
    fn inline_styles_parse() {
        insta::assert_snapshot!(describe_inline(
            "A **bold** word, *italic*, `code`, and a [link](https://example.com)."
        ));
    }

    #[test]
    fn fenced_code_block_splits_from_prose() {
        insta::assert_snapshot!(describe_blocks(
            "Here is code:\n```rust\nfn main() {}\n```\nDone."
        ));
    }

    // Mid-stream tokens commonly leave a marker open; it must still style to EOL.
    #[test]
    fn unclosed_bold_consumes_to_end() {
        let (text, runs, _) = inline_runs("half **bold", &Theme::dark());
        assert_eq!(text, "half bold");
        assert_eq!(runs.len(), 1);
        assert_eq!(kind(&runs[0].1), "bold");
    }

    #[test]
    fn link_url_is_captured() {
        let (text, _, links) = inline_runs("[uzu](https://github.com/trymirai/uzu)", &Theme::dark());
        assert_eq!(text, "uzu");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].1, "https://github.com/trymirai/uzu");
    }

    // A `[text]` with no `(url)` is not a link: emit the brackets literally.
    #[test]
    fn bracket_without_paren_is_literal() {
        let (text, _, links) = inline_runs("see [note] here", &Theme::dark());
        assert_eq!(text, "see [note] here");
        assert!(links.is_empty());
    }
}
