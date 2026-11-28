use dom_query::Document;
use dom_smoothie::{Config, Readability, TextMode};
use reqwest::Url;

use crate::chat::{messages::collapse_whitespace, web_source::PageContent};

const SKIPPED: [&str; 12] =
    ["script", "style", "noscript", "template", "svg", "iframe", "nav", "header", "footer", "aside", "form", "button"];
const MEDIA: [&str; 7] = ["img", "picture", "figure", "video", "audio", "canvas", "svg"];
const DIALOGS: &str = "[role=\"dialog\"], [aria-modal=\"true\"]";
/// Consent banners by name. The names also match real articles (a cookie
/// recipe, a GDPR explainer), so only a plain container with no article inside counts.
const BANNER_NAMES: &str = "[id*=\"cookie\" i], [class*=\"cookie\" i], [id*=\"consent\" i], [class*=\"consent\" i], \
                           [id*=\"gdpr\" i], [class*=\"gdpr\" i]";
const BANNER_CONTAINERS: &str = "div, section, aside";
const PAGE_BODY: &str = "article, main";

pub(super) fn extract(
    page: &str,
    url: &Url,
) -> PageContent {
    if is_nested_too_deeply(page) {
        return PageContent::default();
    }
    let config = Config {
        text_mode: TextMode::Markdown,
        ..Config::default()
    };
    let page = without_overlays(page);
    let article = Readability::new(page.as_str(), Some(url.as_str()), Some(config))
        .ok()
        .filter(Readability::is_probably_readable)
        .and_then(|mut readability| readability.parse().ok());
    if let Some(article) = article {
        let text = tidy(&Document::from(article.content.as_ref()).md(Some(&MEDIA)));
        if !text.is_empty() {
            return PageContent {
                title: collapse_whitespace(&article.title),
                text,
                published_at: article
                    .published_time
                    .map(|time| collapse_whitespace(&time))
                    .filter(|time| !time.is_empty()),
                cut_short: false,
            };
        }
    }
    whole_body(&page, url)
}

/// Parsing time grows with the square of the nesting and the Markdown writer recurses
/// once per level, so a long chain of elements would hang the read or overflow the stack.
const MAX_NESTING: usize = 256;
const VOID_ELEMENTS: [&str; 13] =
    ["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"];
/// A parser closes these when the next of their kind opens, so leaving them open nests nothing.
const IMPLICITLY_CLOSED: [&str; 17] = [
    "p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot", "colgroup", "option", "optgroup", "a",
    "button", "nobr", "form",
];
const RAW_TEXT_ELEMENTS: [&str; 4] = ["script", "style", "textarea", "title"];

fn is_nested_too_deeply(page: &str) -> bool {
    let is_one_of = |name: &str, kinds: &[&str]| kinds.iter().any(|kind| kind.eq_ignore_ascii_case(name));
    let mut open: Vec<&str> = Vec::new();
    let mut rest = page;
    while let Some(start) = rest.find('<') {
        rest = &rest[start + 1..];
        let closes = rest.starts_with('/');
        let named = rest.trim_start_matches('/');
        let name = &named[..named.bytes().take_while(u8::is_ascii_alphanumeric).count()];
        if name.is_empty() || is_one_of(name, &VOID_ELEMENTS) || is_one_of(name, &IMPLICITLY_CLOSED) {
            continue;
        }
        if closes {
            if let Some(opened) = open.iter().rposition(|kind| kind.eq_ignore_ascii_case(name)) {
                open.truncate(opened);
            }
            continue;
        }
        let tag_end = named.find('>').unwrap_or(named.len());
        if named[..tag_end].ends_with('/') {
            continue;
        }
        if is_one_of(name, &RAW_TEXT_ELEMENTS) {
            // Its content is text; a `<` in a script is not a tag.
            let mut after = &named[tag_end..];
            while let Some(candidate) = after.find("</") {
                after = &after[candidate + 2..];
                if after.get(..name.len()).is_some_and(|closing| closing.eq_ignore_ascii_case(name)) {
                    break;
                }
            }
            rest = after;
            continue;
        }
        open.push(name);
        if open.len() > MAX_NESTING {
            return true;
        }
    }
    false
}

fn without_overlays(page: &str) -> String {
    let document = Document::from(page);
    let dialogs = document.select(DIALOGS);
    let banners: Vec<_> = document
        .select(BANNER_NAMES)
        .iter()
        .filter(|banner| banner.is(BANNER_CONTAINERS) && !banner.select(PAGE_BODY).exists())
        .collect();
    if !dialogs.exists() && banners.is_empty() {
        return page.to_string();
    }
    dialogs.remove();
    for banner in banners {
        banner.remove();
    }
    document.html().to_string()
}

fn whole_body(
    page: &str,
    url: &Url,
) -> PageContent {
    let document = Document::from(page);
    // The page's title is the first one; icons drawn in SVG carry titles of their own.
    let title =
        document.select("title").nodes().first().map(|title| collapse_whitespace(&title.text())).unwrap_or_default();
    document.select(&SKIPPED.join(", ")).remove();
    for link in document.select("a[href]").iter() {
        if let Some(href) = link.attr("href")
            && let Ok(absolute) = url.join(&href)
        {
            link.set_attr("href", absolute.as_str());
        }
    }
    PageContent {
        title,
        text: document.select("body").nodes().first().map(|body| tidy(&body.md(Some(&MEDIA)))).unwrap_or_default(),
        published_at: None,
        cut_short: false,
    }
}

fn tidy(markdown: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in markdown.lines().map(str::trim_end) {
        if line.is_empty() && lines.last().is_none_or(|last| last.is_empty()) {
            continue;
        }
        lines.push(line);
    }
    lines.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(page: &str) -> PageContent {
        super::extract(page, &Url::parse("https://example.com/blog/post").unwrap())
    }

    fn paragraphs() -> String {
        (1..=8)
            .map(|index| {
                format!("<p>Paragraph {index} of the release notes explains a change in some detail for readers.</p>")
            })
            .collect()
    }

    #[test]
    fn the_main_part_is_read_as_markdown_without_navigation_scripts_or_forms() {
        let page = format!(
            r#"<html><head><title> Rust  1.100 </title><meta property="article:published_time" content="2026-10-15T00:00:00Z">
            <style>p {{ color: red }}</style></head><body>
            <header>Site name</header><nav><a href="/">Home</a></nav>
            <article><h1>Announcing Rust</h1><p>The team <b>shipped</b> a release, see <a href="/notes">the notes</a>.</p>
              <img src="/hero.png" alt="hero"><script>track()</script>{}
              <ul><li>One</li><li>Two</li></ul>
              <form><input value="email"><button>Subscribe</button></form></article>
            <aside>Related</aside><footer>Copyright</footer></body></html>"#,
            paragraphs()
        );

        let extracted = extract(&page);

        assert_eq!(extracted.title, "Rust 1.100");
        assert_eq!(extracted.published_at.as_deref(), Some("2026-10-15T00:00:00Z"));
        let text = &extracted.text;
        assert!(text.contains("# Announcing Rust"), "{text}");
        assert!(text.contains("**shipped**"), "{text}");
        assert!(text.contains("(https://example.com/notes)"), "{text}");
        assert!(text.contains("- One\n- Two"), "{text}");
        assert!(text.contains("Paragraph 8"), "{text}");
        for left_out in ["Site name", "Home", "track()", "Subscribe", "Related", "Copyright", "hero", "color: red"] {
            assert!(!text.contains(left_out), "{left_out} in {text}");
        }
        assert!(!text.contains("\n\n\n"), "{text}");
    }

    #[test]
    fn a_page_the_reader_view_cannot_read_falls_back_to_its_body() {
        let page = "<html><head><title>Index</title></head><body><nav>Menu</nav><div>Plain page</div><p>Second line</p>\
             <table><tr><th>Name</th><th>Size</th></tr><tr><td><a href=\"/files/a.txt\">a.txt</a></td><td>1 KB</td></tr></table>\
             <svg><title>Chevron</title></svg></body></html>";

        let extracted = extract(page);

        assert_eq!(extracted.title, "Index");
        assert_eq!(extracted.published_at, None);
        let text = &extracted.text;
        assert!(text.starts_with("Plain page"), "{text}");
        assert!(text.contains("Second line"), "{text}");
        assert!(text.contains("| Name | Size |"), "{text}");
        assert!(text.contains("| [a.txt](https://example.com/files/a.txt) | 1 KB |"), "{text}");
        assert!(!text.contains("Menu"), "{text}");
        assert!(!text.contains("Chevron"), "{text}");
    }

    #[test]
    fn consent_dialogs_are_left_out() {
        let page = format!(
            r#"<html><body><div role="dialog" class="fc-consent-root"><p>We and our vendors use cookies to store and access information on a device, and process personal data such as unique identifiers.</p><p>Accept all cookies or manage your choices below.</p></div>
            <div id="onetrust-banner-sdk">Willkommen! Wir freuen uns, dass Sie hier sind.</div>
            <article><h1>The article</h1>{}</article></body></html>"#,
            paragraphs()
        );
        let text = extract(&page).text;
        assert!(text.contains("# The article"), "{text}");
        for left_out in ["cookies", "Willkommen"] {
            assert!(!text.contains(left_out), "{left_out} in {text}");
        }

        let recipe = format!(
            r#"<html><body><div class="cookie-recipe"><article><h1>Oatmeal cookies</h1>{}</article></div></body></html>"#,
            paragraphs()
        );
        assert!(extract(&recipe).text.contains("# Oatmeal cookies"));
    }

    #[test]
    fn a_page_with_only_scripts_has_no_text() {
        let extracted = extract("<body><div id=\"root\"></div><script>render()</script></body>");
        assert_eq!(extracted.text, "");
        assert_eq!(extracted.title, "");
    }

    #[test]
    fn a_page_nested_too_deeply_to_walk_is_left_unread() {
        for element in ["<blockquote>", "<ul><li>", "<div>", "<b>"] {
            let page = format!("<html><body>{}text</body></html>", element.repeat(200_000));
            let read = std::thread::Builder::new()
                .stack_size(2 * 1024 * 1024)
                .spawn(move || extract(&page).text)
                .unwrap()
                .join()
                .expect("extraction must not overflow the stack");
            assert_eq!(read, "", "{element}");
        }
    }

    #[test]
    fn untidy_pages_people_really_write_are_not_taken_for_deep_ones() {
        let unclosed_items = format!("<ul>{}</ul>", "<li>item<p>text".repeat(2_000));
        let unclosed_links = "<a href=/x>link ".repeat(1_000);
        let crossed = "<div><span>text</div>".repeat(1_000);
        let script = format!("<script>{}</script>", "if (a<b) { x = '<div><div>' }".repeat(1_000));
        let self_closed = "<svg/><custom-element />".repeat(1_000);
        for page in [unclosed_items, unclosed_links, crossed, script, self_closed] {
            assert!(!is_nested_too_deeply(&page), "{}", &page[..40]);
        }
        assert!(is_nested_too_deeply(&"<DIV>".repeat(MAX_NESTING + 1)));
        assert!(!is_nested_too_deeply(&"<div>".repeat(MAX_NESTING)));
    }
}
