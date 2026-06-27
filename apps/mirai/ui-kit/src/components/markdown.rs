//! Markdown for chat bodies, mirroring ui-kit's `MarkdownRenderer`: code blocks
//! (with copy), headings, lists, blockquotes, inline bold/italic/code/links
//! (clickable). No tables/math/syntax-highlight yet.
//!
//! Parsing is split from rendering: [`parse`] produces a theme-independent
//! [`ParsedMarkdown`] (cheap to clone/cache), and [`render`] builds elements
//! from it. Callers that render the same text every frame (e.g. chat messages)
//! cache the parse and only rebuild elements, avoiding a per-frame text scan.

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

/// A parsed markdown document — theme-independent and cheap to clone, so callers
/// can cache it and rebuild elements each frame without re-scanning the text.
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

/// Inline text with theme-independent style runs (colors applied at render).
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

/// Parse markdown into a reusable, theme-independent structure.
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

/// Classify one prose line: heading (`#`), blockquote (`> `), unordered
/// (`- `/`* `) or ordered (`N. `) list item, blank, or plain — and parse its
/// inline styling.
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

    // Plain lines keep their original (un-trimmed) text, matching prior behavior.
    Line::Plain(parse_inline(line))
}

/// Renders a parsed document as a vertical stack of prose + code blocks.
/// `id_seed` must be unique per message so code-block copy buttons and link
/// hit-targets get stable, non-colliding ids.
pub fn render(
    parsed: &ParsedMarkdown,
    theme: &Theme,
    id_seed: usize,
) -> AnyElement {
    let mut col = div().flex().flex_col().w_full().min_w_0().gap_2();
    let mut line_no = 0usize;

    for (bi, block) in parsed.blocks.iter().enumerate() {
        col = col.child(match block {
            // Prose: one element per line so single newlines (lists, breaks)
            // survive, while each line still wraps.
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

/// Convenience: parse + render in one call (for callers that don't cache).
pub fn markdown(
    text: &str,
    theme: &Theme,
    id_seed: usize,
) -> AnyElement {
    render(&parse(text), theme, id_seed)
}

/// A fenced code block: header (language label + copy button) over a monospace
/// body, mirroring ui-kit's code-block styling.
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

/// Wrap prose so long lines wrap inside the chat column instead of overflowing.
fn prose_wrap(el: AnyElement) -> AnyElement {
    div().w_full().min_w_0().overflow_hidden().child(el).into_any_element()
}

/// Build one parsed prose line into an element. `id` keeps link hit-targets unique.
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
            // h1=24px h2=20px h3=18px h4–h6=14px (body size), matching Electron.
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

/// Map a parsed inline run kind to a themed highlight style.
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

/// Build inline text: a `StyledText`, upgraded to an `InteractiveText`
/// (clickable links opening in the browser) when it contains `[text](url)`.
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

/// Parse `**bold**`, `*italic*`, `` `code` ``, and `[text](url)` links into plain
/// text + theme-independent style runs. Unclosed markers (common mid-stream)
/// consume to end.
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
                chars.next(); // second '*'
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
                // Underscores only delimit emphasis at word boundaries, so
                // identifiers like `foo_bar` stay literal (CommonMark intraword
                // rule). `*` keeps its intraword behaviour above.
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
                        // No word-boundary close — keep the opener literal.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn kind_name(kind: RunKind) -> &'static str {
        match kind {
            RunKind::Code => "code",
            RunKind::Bold => "bold",
            RunKind::Italic => "italic",
            RunKind::Link => "link",
        }
    }

    /// Readable structural dump of one inline line.
    fn describe_inline(line: &str) -> String {
        let inline = parse_inline(line);
        let mut out = format!("text: {:?}\n", inline.text);
        for (range, kind) in &inline.runs {
            out += &format!(
                "  run {}..{} {} {:?}\n",
                range.start,
                range.end,
                kind_name(*kind),
                &inline.text[range.clone()]
            );
        }
        for (range, url) in &inline.links {
            out += &format!("  link {}..{} -> {url}\n", range.start, range.end);
        }
        out
    }

    fn describe_blocks(text: &str) -> String {
        let mut out = String::new();
        for block in parse(text).blocks {
            match block {
                Block::Prose(lines) => {
                    let joined: Vec<String> = lines
                        .iter()
                        .map(|l| match l {
                            Line::Blank => String::new(),
                            Line::Heading {
                                inline,
                                ..
                            }
                            | Line::Quote(inline)
                            | Line::Bullet(inline)
                            | Line::Ordered {
                                inline,
                                ..
                            }
                            | Line::Plain(inline) => inline.text.clone(),
                        })
                        .collect();
                    out += &format!("[text] {:?}\n", joined.join("\n"));
                },
                Block::Code {
                    lang,
                    code,
                } => out += &format!("[code:{lang}] {code:?}\n"),
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
        insta::assert_snapshot!(describe_blocks("Here is code:\n```rust\nfn main() {}\n```\nDone."));
    }

    // Mid-stream tokens commonly leave a marker open; it must still style to EOL.
    #[test]
    fn unclosed_bold_consumes_to_end() {
        let inline = parse_inline("half **bold");
        assert_eq!(inline.text, "half bold");
        assert_eq!(inline.runs.len(), 1);
        assert_eq!(inline.runs[0].1, RunKind::Bold);
    }

    #[test]
    fn link_url_is_captured() {
        let inline = parse_inline("[uzu](https://github.com/trymirai/uzu)");
        assert_eq!(inline.text, "uzu");
        assert_eq!(inline.links.len(), 1);
        assert_eq!(inline.links[0].1, "https://github.com/trymirai/uzu");
    }

    // A `[text]` with no `(url)` is not a link: emit the brackets literally.
    #[test]
    fn bracket_without_paren_is_literal() {
        let inline = parse_inline("see [note] here");
        assert_eq!(inline.text, "see [note] here");
        assert!(inline.links.is_empty());
    }
}
