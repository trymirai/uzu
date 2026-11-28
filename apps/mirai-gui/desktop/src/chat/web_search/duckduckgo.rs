use std::time::{Duration, Instant};

use dom_query::{Document, Selection};

use super::{HTTP_CLIENT, Provider, SNIPPET_MAX_CHARS, SearchError, SearchResult};
use crate::chat::{
    messages::{clip_chars, collapse_whitespace},
    web_view::{BROWSER_USER_AGENT, WebView, WebViewError},
};

const ENDPOINT: &str = "https://html.duckduckgo.com/html/";
const AD_CLASS: &str = "result--ad";
/// Outbound links may be wrapped in a redirect that carries the target here.
const REDIRECT_TARGET_PARAMETER: &str = "uddg";
const OWN_HOST_SUFFIX: &str = "duckduckgo.com";
/// A model often asks for several searches at once, and a burst is answered with the bot check.
const REQUEST_SPACING: Duration = Duration::from_secs(1);

const RESULT: &str = ".result";
const TITLE_LINK: &str = "a.result__a";
const SNIPPET: &str = ".result__snippet";
const NO_RESULTS: &str = ".result--no-result";
const BOT_CHECK: &str = ".anomaly-modal, #challenge-form";

pub(super) async fn search(
    query: &str,
    web_view: Option<&WebView>,
    pacing: &tokio::sync::Mutex<Option<Instant>>,
) -> Result<Vec<SearchResult>, SearchError> {
    match (search_plain(query, pacing).await, web_view) {
        (Err(SearchError::Blocked(_)), Some(web_view)) => search_in_browser(web_view, query, pacing).await,
        (outcome, _) => outcome,
    }
}

async fn search_plain(
    query: &str,
    pacing: &tokio::sync::Mutex<Option<Instant>>,
) -> Result<Vec<SearchResult>, SearchError> {
    let mut last_request = pacing.lock().await;
    if let Some(finished) = *last_request {
        tokio::time::sleep_until((finished + REQUEST_SPACING).into()).await;
    }
    let request = HTTP_CLIENT.get(ENDPOINT).query(&[("q", query)]).header("user-agent", BROWSER_USER_AGENT);
    let response = request.send().await;
    *last_request = Some(Instant::now());
    drop(last_request);
    let response = response.map_err(|error| SearchError::request(Provider::DuckDuckGo, error))?;
    let status = response.status();
    let page = super::response_text(response, Provider::DuckDuckGo).await?;
    // The bot check is served as 202 Accepted with a challenge page.
    if status == reqwest::StatusCode::ACCEPTED {
        return Err(SearchError::Blocked(Provider::DuckDuckGo));
    }
    match SearchError::from_status(Provider::DuckDuckGo, status, false) {
        Some(refusal) => Err(refusal),
        None => parse_page(&page),
    }
}

async fn search_in_browser(
    web_view: &WebView,
    query: &str,
    pacing: &tokio::sync::Mutex<Option<Instant>>,
) -> Result<Vec<SearchResult>, SearchError> {
    let mut last_request = pacing.lock().await;
    let url = reqwest::Url::parse_with_params(ENDPOINT, [("q", query)])
        .map_err(|error| SearchError::failed(Provider::DuckDuckGo, error))?;
    let rendered = web_view.render(&url, BOT_CHECK).await;
    *last_request = Some(Instant::now());
    drop(last_request);
    match rendered {
        Ok(html) => parse_page(&html),
        Err(WebViewError::Challenged) => Err(SearchError::Blocked(Provider::DuckDuckGo)),
        Err(error) => Err(SearchError::failed(Provider::DuckDuckGo, format!("in the browser window, {error}"))),
    }
}

fn parse_page(page: &str) -> Result<Vec<SearchResult>, SearchError> {
    let document = Document::from(page);
    if document.select(BOT_CHECK).exists() {
        return Err(SearchError::Blocked(Provider::DuckDuckGo));
    }
    let results: Vec<_> =
        document.select(RESULT).iter().filter(|result| !result.has_class(AD_CLASS)).filter_map(parse_result).collect();
    if results.is_empty() && !document.select(NO_RESULTS).exists() {
        return Err(SearchError::failed(Provider::DuckDuckGo, "unrecognized results page"));
    }
    Ok(results)
}

fn parse_result(result: Selection) -> Option<SearchResult> {
    let link = result.select(TITLE_LINK).first();
    let url = target_url(&link.attr("href")?)?;
    let snippet = result.select(SNIPPET).first();
    Some(SearchResult {
        title: collapse_whitespace(&link.text()),
        url,
        snippet: clip_chars(&collapse_whitespace(&snippet.text()), SNIPPET_MAX_CHARS),
        published_at: None,
    })
}

fn target_url(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    let url = reqwest::Url::parse(&absolute).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let on_search_service =
        url.host_str().is_some_and(|host| host == OWN_HOST_SUFFIX || host.ends_with(&format!(".{OWN_HOST_SUFFIX}")));
    if !on_search_service {
        return Some(url.into());
    }
    let target = url.query_pairs().find(|(key, _)| key == REDIRECT_TARGET_PARAMETER)?.1.into_owned();
    target_url(&target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::web_search::live;

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_duckduckgo_answers_a_plain_search() {
        let pacing = tokio::sync::Mutex::const_new(None);
        live::assert_results(&search(live::QUERY, None, &pacing).await.unwrap());
    }

    const NO_RESULTS_BLOCK: &str = r#"<div class="result results_links web-result result--no-result">
        <div class="no-results__message">No results found</div></div>"#;

    fn result_block(
        classes: &str,
        href: &str,
        title: &str,
        snippet: &str,
    ) -> String {
        format!(
            r#"<div class="result results_links results_links_deep {classes}">
                 <div class="links_main links_deep result__body">
                   <h2 class="result__title"><a rel="nofollow" class="result__a" href="{href}">{title}</a></h2>
                   <div class="result__extras"><a class="result__url" href="{href}">example</a></div>
                   <a class="result__snippet" href="{href}">{snippet}</a>
                 </div>
               </div>"#
        )
    }

    fn page(blocks: &[String]) -> String {
        format!(r#"<html><body><div class="results">{}</div></body></html>"#, blocks.concat())
    }

    #[test]
    fn results_keep_page_order_and_markup_inside_text_is_flattened() {
        let page = page(&[
            result_block(
                "web-result",
                "https://releases.rs/",
                "Rust Versions | Rust Changelogs",
                "<b>Rust</b> <b>Versions</b>\n   # Stable: 1.100.0",
            ),
            result_block("web-result", "https://example.com/a?b=1&amp;c=2", "Second &amp; last", ""),
        ]);

        assert_eq!(
            parse_page(&page).unwrap(),
            [
                SearchResult {
                    title: "Rust Versions | Rust Changelogs".into(),
                    url: "https://releases.rs/".into(),
                    snippet: "Rust Versions # Stable: 1.100.0".into(),
                    published_at: None,
                },
                SearchResult {
                    title: "Second & last".into(),
                    url: "https://example.com/a?b=1&c=2".into(),
                    snippet: String::new(),
                    published_at: None,
                },
            ]
        );
    }

    #[test]
    fn redirect_wrappers_are_unwrapped_and_ads_are_dropped() {
        let wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3D1&amp;rut=abc";
        let ad_click = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3Dshop.example";
        let page = page(&[
            result_block("result--ad", "https://shop.example/", "Sponsored", "ad"),
            result_block("web-result", ad_click, "Sponsored link", "ad"),
            result_block("web-result", "https://duckduckgo.com/y.js?ad_domain=shop.example", "Ad script", "ad"),
            result_block("web-result", "javascript:void(0)", "Not a page", ""),
            result_block("web-result", wrapped, "Wrapped", "kept"),
        ]);

        let results = parse_page(&page).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Wrapped");
        assert_eq!(results[0].url, "https://example.com/path?q=1");
    }

    #[test]
    fn the_nothing_found_page_is_an_empty_answer_but_unknown_markup_is_a_failure() {
        assert!(parse_page(&page(&[NO_RESULTS_BLOCK.to_string()])).unwrap().is_empty());
        assert_eq!(
            parse_page(&page(&[])).unwrap_err().to_string(),
            "DuckDuckGo search failed: unrecognized results page"
        );
    }

    #[test]
    fn a_challenge_page_is_a_bot_check_but_the_same_words_in_a_result_are_not() {
        let challenge = r#"<html><body><div class="anomaly-modal__mask"><div class="anomaly-modal">
            <form id="challenge-form"></form></div></div></body></html>"#;
        assert_eq!(parse_page(challenge).unwrap_err(), SearchError::Blocked(Provider::DuckDuckGo));

        let mentions = page(&[result_block("web-result", "https://example.com/", "anomaly-modal", "challenge-form")]);
        assert_eq!(parse_page(&mentions).unwrap().len(), 1);
    }
}
