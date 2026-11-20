mod brave;
mod duckduckgo;
mod exa;
mod keenable;
mod keys;
mod mcp;
mod permission;
mod tool;

use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};

pub use keys::StoredKey;
pub(super) use keys::{key_endings, remove_key, save_key};
pub(super) use permission::{Decision, Permission, PermissionRequest, ShowRequest, Subject};
use reqwest::{StatusCode, Url, redirect};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
pub use tool::SearchFailure;
pub(super) use tool::{WEB_SEARCH_TOOL_NAME, search_view, searched_query, web_search_tool};

use super::{
    known_urls::MAX_URL_BYTES,
    messages::clip_chars,
    web_source::{PageContent, title_line},
    web_tools::WebTools,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const RESULT_COUNT: u32 = 5;
const SNIPPET_MAX_CHARS: usize = 500;
const PROVIDER_SETTING_KEY: &str = "webSearchProvider";
const PROVIDER_MESSAGE_MAX_CHARS: usize = 200;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    published_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Provider {
    DuckDuckGo,
    Exa,
    Keenable,
    Brave,
}

impl Provider {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::DuckDuckGo => "DuckDuckGo",
            Self::Exa => "Exa",
            Self::Keenable => "Keenable",
            Self::Brave => "Brave",
        }
    }

    pub(super) fn selected() -> Self {
        crate::storage::settings_load().map_or(Self::DuckDuckGo, |settings| Self::selected_in(&settings))
    }

    fn selected_in(settings: &serde_json::Value) -> Self {
        settings.get(PROVIDER_SETTING_KEY).and_then(|value| Self::deserialize(value).ok()).unwrap_or(Self::DuckDuckGo)
    }

    fn keyed(self) -> Option<KeyedProvider> {
        match self {
            Self::DuckDuckGo => None,
            Self::Exa => Some(KeyedProvider::Exa),
            Self::Keenable => Some(KeyedProvider::Keenable),
            Self::Brave => Some(KeyedProvider::Brave),
        }
    }

    fn page_tool(self) -> Option<&'static mcp::PageTool> {
        match self {
            Self::Exa => Some(&exa::PAGE_TOOL),
            Self::Keenable => Some(&keenable::PAGE_TOOL),
            Self::DuckDuckGo | Self::Brave => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyedProvider {
    Exa,
    Keenable,
    Brave,
}

impl KeyedProvider {
    fn provider(self) -> Provider {
        match self {
            Self::Exa => Provider::Exa,
            Self::Keenable => Provider::Keenable,
            Self::Brave => Provider::Brave,
        }
    }

    async fn search(
        self,
        query: &str,
        key: Option<&str>,
    ) -> Result<Vec<SearchResult>, SearchError> {
        match self {
            Self::Exa => exa::search(query, key).await,
            Self::Keenable => keenable::search(query, key).await,
            Self::Brave => brave::search(query, key).await,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(super) enum SearchError {
    #[error("{} search limit reached", .0.label())]
    RateLimited(Provider),
    #[error("{} blocked the request with a bot check", .0.label())]
    Blocked(Provider),
    #[error("{} needs an API key, which is added in Settings", .0.label())]
    NoKey(Provider),
    #[error("{} did not accept the API key; it can be changed in Settings", .0.label())]
    KeyRejected(Provider),
    #[error("the {} API key could not be read from the macOS Keychain", .0.label())]
    KeyUnreadable(Provider),
    #[error("{} rejected the search: {message}", .provider.label())]
    Rejected {
        provider: Provider,
        message: String,
    },
    #[error("{} search failed: {message}", .provider.label())]
    Failed {
        provider: Provider,
        message: String,
    },
}

impl SearchError {
    fn failed(
        provider: Provider,
        message: impl std::fmt::Display,
    ) -> Self {
        Self::Failed {
            provider,
            message: message.to_string(),
        }
    }

    fn rejected(
        provider: Provider,
        message: &str,
    ) -> Self {
        Self::Rejected {
            provider,
            message: clip_chars(message, PROVIDER_MESSAGE_MAX_CHARS),
        }
    }

    fn from_status(
        provider: Provider,
        status: StatusCode,
        keyed: bool,
    ) -> Option<Self> {
        if status == StatusCode::TOO_MANY_REQUESTS {
            Some(Self::RateLimited(provider))
        } else if keyed && matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
            Some(Self::KeyRejected(provider))
        } else if status.is_success() {
            None
        } else {
            Some(Self::failed(provider, format!("HTTP {}", status.as_u16())))
        }
    }

    fn request(
        provider: Provider,
        error: reqwest::Error,
    ) -> Self {
        Self::failed(provider, error.without_url())
    }

    pub(super) fn log_entry(&self) -> serde_json::Value {
        match self {
            Self::RateLimited(_) => serde_json::json!({ "kind": "rateLimited" }),
            Self::Blocked(_) => serde_json::json!({ "kind": "blocked" }),
            Self::NoKey(_) => serde_json::json!({ "kind": "noKey" }),
            Self::KeyRejected(_) => serde_json::json!({ "kind": "keyRejected" }),
            Self::KeyUnreadable(_) => serde_json::json!({ "kind": "keyUnreadable" }),
            Self::Rejected {
                ..
            } => serde_json::json!({ "kind": "rejected" }),
            Self::Failed {
                message,
                ..
            } => serde_json::json!({ "kind": "failed", "message": message }),
        }
    }
}

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

async fn response_text(
    mut response: reqwest::Response,
    provider: Provider,
) -> Result<String, SearchError> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| SearchError::request(provider, error))? {
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(SearchError::failed(provider, "response too large"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Redirects are not followed: the request can carry the user's key.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .redirect(redirect::Policy::none())
        .build()
        .expect("web search HTTP client")
});

pub(super) struct ReplyState {
    pub(super) permission: Arc<Permission>,
    pub(super) provider: Provider,
    /// Read once per reply; searches running side by side wait for the same read.
    key: Arc<tokio::sync::OnceCell<Result<Option<String>, SearchError>>>,
    pub(super) result_chars: usize,
}

impl ReplyState {
    pub(super) fn new(
        provider: Provider,
        asks_before_searching: bool,
        show_request: ShowRequest,
    ) -> Self {
        Self {
            permission: Arc::new(Permission::new(asks_before_searching, show_request)),
            provider,
            key: Arc::default(),
            result_chars: 0,
        }
    }
}

async fn search(
    provider: Provider,
    query: &str,
    tools: &WebTools,
) -> Result<Vec<SearchResult>, SearchError> {
    let results = match provider.keyed() {
        Some(keyed) => keyed.search(query, reply_key(provider, tools).await?.as_deref()).await,
        None => duckduckgo::search(query, tools.web_view.as_ref(), &tools.scraped_search_pacing).await,
    };
    results.map(presentable)
}

async fn reply_key(
    provider: Provider,
    tools: &WebTools,
) -> Result<Option<String>, SearchError> {
    match tools.with_reply(|reply| reply.search.key.clone()) {
        Ok(key) => key.get_or_init(|| stored_key(provider)).await.clone(),
        Err(_) => stored_key(provider).await,
    }
}

/// The Keychain is read only for a provider whose key the settings show: reading it can make
/// macOS ask the user, which a search without a key must never do.
async fn stored_key(provider: Provider) -> Result<Option<String>, SearchError> {
    let read = tokio::task::spawn_blocking(move || {
        if !keys::has_ending(provider) {
            return Ok(None);
        }
        match keys::read_for_search() {
            Ok(keys) => Ok(keys.get(provider).map(str::to_string)),
            Err(error) => {
                crate::logger::warn(
                    "web-search:key:unreadable",
                    Some(serde_json::json!({ "provider": provider, "error": error })),
                );
                Err(SearchError::KeyUnreadable(provider))
            },
        }
    });
    read.await.unwrap_or(Ok(None))
}

pub(super) async fn read_page(
    tools: &WebTools,
    url: &Url,
    max_chars: usize,
) -> Option<Result<PageContent, SearchError>> {
    let (run_id, provider, permission) = tools
        .with_reply(|reply| (reply.run_id.clone(), reply.search.provider, reply.search.permission.clone()))
        .ok()?;
    let tool = provider.page_tool()?;
    // The provider found the other addresses itself or they are public links;
    // one the user wrote can be private, so the user is asked first.
    if tools.with_known(|known| known.was_written_by_user(url))
        && !permission.allows(&run_id, provider.label(), Subject::Page(url.as_str())).await
    {
        return None;
    }
    let key = match reply_key(provider, tools).await {
        Ok(key) => key,
        Err(error) => return Some(Err(error)),
    };
    Some(mcp::read_page(tool, url.as_str(), max_chars, key.as_deref()).await)
}

fn presentable(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut kept: Vec<SearchResult> = Vec::new();
    for result in results {
        let is_web_page = result.url.starts_with("https://") || result.url.starts_with("http://");
        let is_new = kept.iter().all(|other| other.url != result.url);
        if is_web_page && is_new && result.url.len() <= MAX_URL_BYTES && kept.len() < RESULT_COUNT as usize {
            kept.push(SearchResult {
                title: title_line(&result.title),
                ..result
            });
        }
    }
    kept
}

#[cfg(test)]
pub(in crate::chat) mod live {
    pub(in crate::chat) const QUERY: &str = "current stable Rust version";
    pub(in crate::chat) const SCRIPT_DRAWN_PAGE: &str = "https://quotes.toscrape.com/js/";
    pub(in crate::chat) const SCRIPT_DRAWN_TEXT: &str = "The world as we have created it";

    pub(super) fn assert_results(results: &[super::SearchResult]) {
        assert!(!results.is_empty());
        for result in results {
            assert!(result.url.starts_with("http"));
            assert!(!result.snippet.is_empty(), "an excerpt comes with every result");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(
        title: &str,
        url: &str,
    ) -> SearchResult {
        SearchResult {
            title: title.to_string(),
            url: url.to_string(),
            snippet: String::new(),
            published_at: None,
        }
    }

    #[test]
    fn settings_name_one_provider_and_default_to_duckduckgo() {
        let selected = |settings: serde_json::Value| Provider::selected_in(&settings);
        for (value, provider) in [
            ("duckduckgo", Provider::DuckDuckGo),
            ("exa", Provider::Exa),
            ("keenable", Provider::Keenable),
            ("brave", Provider::Brave),
        ] {
            assert_eq!(selected(serde_json::json!({ PROVIDER_SETTING_KEY: value })), provider);
        }
        for unreadable in [
            serde_json::json!({}),
            serde_json::json!({ PROVIDER_SETTING_KEY: "Exa" }),
            serde_json::json!({ PROVIDER_SETTING_KEY: "nobody" }),
            serde_json::json!({ PROVIDER_SETTING_KEY: 3 }),
        ] {
            assert_eq!(selected(unreadable), Provider::DuckDuckGo);
        }
    }

    #[test]
    fn every_provider_but_duckduckgo_takes_a_key_and_two_read_pages() {
        assert_eq!(Provider::DuckDuckGo.keyed(), None);
        for keyed in [KeyedProvider::Exa, KeyedProvider::Keenable, KeyedProvider::Brave] {
            assert_eq!(keyed.provider().keyed(), Some(keyed));
        }
        let reads_pages = |provider: Provider| provider.page_tool().is_some();
        assert!(reads_pages(Provider::Exa) && reads_pages(Provider::Keenable));
        assert!(!reads_pages(Provider::DuckDuckGo) && !reads_pages(Provider::Brave));
    }

    #[test]
    fn an_http_status_is_a_spent_quota_a_refused_key_or_a_failure() {
        let status =
            |code: u16, keyed| SearchError::from_status(Provider::Exa, StatusCode::from_u16(code).unwrap(), keyed);
        assert_eq!(status(200, true), None);
        assert_eq!(status(429, false), Some(SearchError::RateLimited(Provider::Exa)));
        for refusal in [401, 403] {
            assert_eq!(status(refusal, true), Some(SearchError::KeyRejected(Provider::Exa)));
            assert_eq!(status(refusal, false), Some(SearchError::failed(Provider::Exa, format!("HTTP {refusal}"))));
        }
        assert_eq!(status(502, true), Some(SearchError::failed(Provider::Exa, "HTTP 502")));
    }

    #[test]
    fn a_providers_wording_is_shortened_for_the_model_and_kept_out_of_the_log() {
        let ours = SearchError::failed(Provider::Exa, "HTTP 502");
        assert_eq!(ours.log_entry(), serde_json::json!({ "kind": "failed", "message": "HTTP 502" }));

        let theirs = SearchError::rejected(Provider::Keenable, "No results for query \"private question\"");
        assert_eq!(theirs.log_entry(), serde_json::json!({ "kind": "rejected" }));
        assert!(theirs.to_string().contains("private question"));

        let long = SearchError::rejected(Provider::Keenable, &"word ".repeat(200));
        assert!(
            matches!(long, SearchError::Rejected { message, .. } if message.chars().count() == PROVIDER_MESSAGE_MAX_CHARS)
        );
    }

    #[test]
    fn results_are_limited_to_web_pages_listed_once() {
        let mut results = vec![
            page("First", "https://example.com/a"),
            page("Script", "javascript:alert(1)"),
            page("Again", "https://example.com/a"),
            page("Plain", "http://example.com/b"),
        ];
        results.extend((0..RESULT_COUNT).map(|index| page("More", &format!("https://example.com/more/{index}"))));

        let kept = presentable(results);

        assert_eq!(kept.len(), RESULT_COUNT as usize);
        assert_eq!(kept[0].title, "First");
        assert_eq!(presentable(vec![page(" Two\n lines ", "https://example.com/")])[0].title, "Two lines");
        assert_eq!(kept[1].title, "Plain");
        assert!(kept[2..].iter().all(|result| result.title == "More"));
    }
}
