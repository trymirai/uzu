mod download;
mod extract;

use std::{collections::HashMap, sync::Arc};

use download::{ReadPage, fetch};
use reqwest::Url;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uzu::{
    session::tool::{
        func_def::{ErrorFuture, ToolDescriptor},
        uzu_tool_closure,
    },
    types::basic::ToolCall,
};

use super::{
    known_urls::Origin,
    web_address::public_web_url,
    web_source::WebSource,
    web_tools::{WebTools, millis_since},
};

pub(super) const WEB_FETCH_TOOL_NAME: &str = "web_fetch";

pub(super) const TEXT_MAX_CHARS: usize = 16_000;
const TEXT_PER_REPLY_CHARS: usize = 3 * TEXT_MAX_CHARS;

type PageSlot = Arc<tokio::sync::Mutex<Option<Arc<ReadPage>>>>;

#[derive(Default)]
pub(super) struct ReplyState {
    pages: HashMap<String, PageSlot>,
    pub(super) result_chars: usize,
}

impl ReplyState {
    fn slot(
        &mut self,
        key: String,
    ) -> PageSlot {
        self.pages.entry(key).or_default().clone()
    }

    /// Set aside before the page is read, so calls that run together cannot pass
    /// the limit between them. None when the reply has had its text.
    fn reserve_part(&mut self) -> Option<usize> {
        let reserved = TEXT_MAX_CHARS.min(TEXT_PER_REPLY_CHARS.saturating_sub(self.result_chars));
        self.result_chars += reserved;
        (reserved > 0).then_some(reserved)
    }

    fn settle_part(
        &mut self,
        reserved: usize,
        handed_over: usize,
    ) {
        self.result_chars = self.result_chars - reserved + handed_over;
    }
}

pub(super) fn web_fetch_tool(tools: Arc<WebTools>) -> ToolDescriptor {
    uzu_tool_closure! {
        /// Returns a web page as Markdown.
        /// A long page comes in parts: if truncated is true, call again with start set to nextStart.
        web_fetch: async move |
            url: String,
            start: Option<usize>
        | -> Result<PagePart, ErrorFuture> {
            let requested = requested_page(&url, &tools).map_err(|error| error.to_string())?;
            let (run_id, reserved) = tools
                .with_reply(|reply| (reply.run_id.clone(), reply.fetch.reserve_part()))
                .map_err(|error| error.to_string())?;
            let started = std::time::Instant::now();
            let start = start.unwrap_or(0);
            let outcome = if let Some(reserved) = reserved {
                let read = read(requested.url.clone(), &tools).await;
                let outcome = read.and_then(|page| Ok((part(&page, start, reserved)?, page)));
                let handed_over = outcome.as_ref().map_or(0, |(part, _)| part.text.chars().count());
                let _ = tools.with_reply(|reply| reply.fetch.settle_part(reserved, handed_over));
                outcome
            } else {
                Err(FetchError::TextAllowanceSpent)
            };
            if let Ok((part, page)) = &outcome {
                remember_read(&tools, &requested, &page.url, &part.text);
            }
            // The address stays out of the log: the log is exportable.
            crate::logger::info(
                "web-fetch:done",
                Some(serde_json::json!({
                    "runId": run_id,
                    "start": start,
                    "chars": outcome.as_ref().ok().map(|(part, _)| part.text.chars().count()),
                    "truncated": outcome.as_ref().ok().map(|(part, _)| part.truncated),
                    "reader": outcome.as_ref().ok().map(|(_, page)| page.reader),
                    "error": outcome.as_ref().err().map(FetchError::kind),
                    "durationMs": millis_since(started),
                })),
            );
            outcome.map(|(part, _)| part).map_err(|error| error.to_string().into())
        }
    }
}

struct RequestedPage {
    url: Url,
    written_by_user: bool,
}

fn requested_page(
    address: &str,
    tools: &WebTools,
) -> Result<RequestedPage, FetchError> {
    let spelled = if address.contains("://") {
        address.to_string()
    } else {
        format!("https://{address}")
    };
    let mut url = public_web_url(&spelled).map_err(|_| FetchError::Address)?;
    // A fragment never reaches a server with a plain request, but a reader that runs the
    // page's scripts would hand it to them; a page must not get text through it.
    url.set_fragment(None);
    let (openable, shown_as_http, written_by_user) = tools
        .with_known(|known| (known.is_openable(&url), known.was_shown_as_http(&url), known.was_written_by_user(&url)));
    if !openable {
        return Err(FetchError::NotShown);
    }
    if url.scheme() == "http" && !shown_as_http {
        let _ = url.set_scheme("https");
    }
    Ok(RequestedPage {
        url,
        written_by_user,
    })
}

fn remember_read(
    tools: &WebTools,
    requested: &RequestedPage,
    read_from: &Url,
    text: &str,
) {
    let origin = if requested.written_by_user {
        Origin::User
    } else {
        Origin::Trusted
    };
    tools.with_known(|known| {
        known.remember(read_from, origin);
        if let Some(host) = read_from.host_str() {
            known.remember_text(
                text,
                &Origin::Page {
                    host: host.to_lowercase(),
                },
            );
        }
    });
}

pub(super) fn requested_address(call: &ToolCall) -> String {
    #[derive(Deserialize)]
    struct Arguments {
        url: String,
    }
    serde_json::from_str::<Arguments>(&call.arguments.json)
        .ok()
        .and_then(|arguments| public_web_url(&arguments.url).ok())
        .map(String::from)
        .unwrap_or_default()
}

pub(super) fn page_source(result: &serde_json::Value) -> Option<WebSource> {
    let page = PagePart::deserialize(result).ok()?;
    Some(WebSource {
        title: if page.title.is_empty() {
            page.url.clone()
        } else {
            page.title
        },
        url: page.url,
    })
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct PagePart {
    url: String,
    title: String,
    text: String,
    truncated: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    start: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next_start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    published_at: Option<String>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, PartialEq, thiserror::Error)]
enum FetchError {
    #[error("The URL must be a full http:// or https:// URL of a public web page")]
    Address,
    #[error("Only a URL from the conversation, a search result or a page already read can be opened.")]
    NotShown,
    #[error("The site refused automated access (HTTP {0}). Try another page.")]
    Refused(u16),
    #[error("The page could not be loaded (HTTP {0})")]
    Status(u16),
    #[error("The URL does not hold a text page ({0})")]
    NotText(String),
    #[error("The page has no readable text. Try another page.")]
    Empty,
    #[error("The page could not be loaded: {0}")]
    Request(String),
    #[error("The page has {0} characters, so start must be smaller than that")]
    PastEnd(usize),
    #[error("This reply has already read as much page text as it may. Answer with what you have.")]
    TextAllowanceSpent,
    #[error("{0}")]
    NoReply(super::web_tools::NoReply),
}

impl FetchError {
    fn kind(&self) -> &'static str {
        match self {
            Self::Address => "address",
            Self::NotShown => "notShown",
            Self::Refused(_) => "refused",
            Self::Status(_) => "status",
            Self::NotText(_) => "notText",
            Self::Empty => "empty",
            Self::Request(_) => "request",
            Self::PastEnd(_) => "pastEnd",
            Self::TextAllowanceSpent => "textAllowanceSpent",
            Self::NoReply(_) => "noReply",
        }
    }
}

fn download_key(url: &Url) -> String {
    url.as_str().split_once("://").map_or_else(|| url.to_string(), |(_, address)| address.to_string())
}

async fn read(
    url: Url,
    tools: &WebTools,
) -> Result<Arc<ReadPage>, FetchError> {
    let requested = download_key(&url);
    let slot = tools.with_reply(|reply| reply.fetch.slot(requested.clone())).map_err(FetchError::NoReply)?;
    let mut slot = slot.lock().await;
    if let Some(page) = slot.as_ref() {
        return Ok(page.clone());
    }
    let page = Arc::new(fetch(url, tools).await?);
    *slot = Some(page.clone());
    keep_under_final_address(tools, &requested, &page);
    Ok(page)
}

fn keep_under_final_address(
    tools: &WebTools,
    requested: &str,
    page: &Arc<ReadPage>,
) {
    let led_to = download_key(&page.url);
    if led_to != requested {
        let _ = tools.with_reply(|reply| {
            reply.fetch.pages.entry(led_to).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(Some(page.clone()))));
        });
    }
}

fn part(
    page: &ReadPage,
    start: usize,
    max_chars: usize,
) -> Result<PagePart, FetchError> {
    if start >= page.length {
        return Err(FetchError::PastEnd(page.length));
    }
    let end = page.length.min(start + max_chars);
    Ok(PagePart {
        url: page.url.to_string(),
        title: page.content.title.clone(),
        text: page.content.text.chars().skip(start).take(end - start).collect(),
        truncated: end < page.length,
        start,
        next_start: (end < page.length).then_some(end),
        published_at: page.content.published_at.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            payloads::test_message,
            web_search::{Provider, live},
            web_source::PageContent,
        },
        *,
    };

    fn page_at(
        address: &str,
        text: &str,
    ) -> Result<ReadPage, FetchError> {
        let content = PageContent {
            title: "Title".into(),
            text: text.into(),
            published_at: None,
        };
        ReadPage::new(Url::parse(address).unwrap(), content, download::Reader::Plain)
    }

    fn page(text: &str) -> Result<ReadPage, FetchError> {
        page_at("https://example.com/", text)
    }

    #[test]
    fn a_long_page_comes_in_parts() {
        let short = part(&page("Short text").unwrap(), 0, TEXT_MAX_CHARS).unwrap();
        assert_eq!(
            (short.text.as_str(), short.truncated, short.start, short.next_start),
            ("Short text", false, 0, None)
        );
        let serialized = serde_json::to_value(&short).unwrap();
        assert!(serialized.get("start").is_none() && serialized.get("nextStart").is_none());

        let long = page(&"é".repeat(TEXT_MAX_CHARS + 1)).unwrap();
        let first = part(&long, 0, TEXT_MAX_CHARS).unwrap();
        assert_eq!(first.text.chars().count(), TEXT_MAX_CHARS);
        assert_eq!((first.truncated, first.next_start), (true, Some(TEXT_MAX_CHARS)));
        let second = part(&long, first.next_start.unwrap(), TEXT_MAX_CHARS).unwrap();
        assert_eq!(
            (second.text.as_str(), second.truncated, second.start, second.next_start),
            ("é", false, TEXT_MAX_CHARS, None)
        );
        assert_eq!(
            part(&long, TEXT_MAX_CHARS + 1, TEXT_MAX_CHARS).unwrap_err(),
            FetchError::PastEnd(TEXT_MAX_CHARS + 1)
        );

        let rest_of_allowance = part(&long, 0, 100).unwrap();
        assert_eq!(rest_of_allowance.text.chars().count(), 100);
        assert_eq!((rest_of_allowance.truncated, rest_of_allowance.next_start), (true, Some(100)));
    }

    #[test]
    fn a_reply_reads_three_parts_of_text_however_many_pages_they_come_from() {
        let mut reply = ReplyState::default();
        for _ in 0..3 {
            assert_eq!(reply.reserve_part(), Some(TEXT_MAX_CHARS));
            reply.settle_part(TEXT_MAX_CHARS, TEXT_MAX_CHARS);
        }
        assert_eq!(reply.reserve_part(), None);
        assert_eq!(reply.result_chars, TEXT_PER_REPLY_CHARS);

        let mut reply = ReplyState::default();
        for _ in 0..3 {
            assert_eq!(reply.reserve_part(), Some(TEXT_MAX_CHARS));
        }
        assert_eq!(reply.reserve_part(), None);
        reply.settle_part(TEXT_MAX_CHARS, 500);
        reply.settle_part(TEXT_MAX_CHARS, 0);
        assert_eq!(reply.reserve_part(), Some(TEXT_MAX_CHARS));
        assert_eq!(reply.reserve_part(), Some(TEXT_MAX_CHARS - 500));
        assert_eq!(reply.result_chars, TEXT_PER_REPLY_CHARS);
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_a_page_that_draws_itself_with_scripts_is_read_by_the_replys_provider() {
        let read_with = |provider: Provider| async move {
            let tools = WebTools::new(None);
            let _reply = tools.begin_reply("live", provider, true, Box::new(|_| true));
            download::fetch(Url::parse(live::SCRIPT_DRAWN_PAGE).unwrap(), &tools).await.unwrap()
        };
        for provider in [Provider::Exa, Provider::Keenable] {
            let page = read_with(provider).await;
            assert_eq!(page.reader, download::Reader::Provider, "{provider:?}");
            assert!(page.content.text.contains(live::SCRIPT_DRAWN_TEXT), "{provider:?}");
        }
        assert_eq!(read_with(Provider::DuckDuckGo).await.reader, download::Reader::Plain);
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_an_address_the_user_wrote_goes_to_the_provider_only_when_they_allow_it() {
        use super::super::web_search::Decision;
        let drawn_by_scripts = Url::parse(live::SCRIPT_DRAWN_PAGE).unwrap();
        for (decision, reader) in
            [(Decision::Deny, download::Reader::Plain), (Decision::Allow, download::Reader::Provider)]
        {
            let (shown, mut requests) = tokio::sync::mpsc::unbounded_channel();
            let tools = WebTools::new(None);
            let searches_without_asking = false;
            let show = Box::new(move |request| shown.send(request).is_ok());
            let _reply = tools.begin_reply("live", Provider::Exa, searches_without_asking, show);
            let wrote = test_message(
                uzu::types::session::chat::ChatRole::User {},
                &format!("Read {}", live::SCRIPT_DRAWN_PAGE),
                None,
            );
            tools.remember_messages(&[wrote], true);
            let user = async {
                let request = requests.recv().await.unwrap();
                assert_eq!(
                    (request.provider, request.pages.as_slice()),
                    ("Exa", [drawn_by_scripts.to_string()].as_slice())
                );
                let permission = tools.with_reply(|reply| reply.search.permission.clone()).unwrap();
                assert!(permission.answer(&request.request_id, 1, decision));
            };
            let (page, ()) = tokio::join!(download::fetch(drawn_by_scripts.clone(), &tools), user);
            assert_eq!(page.unwrap().reader, reader, "{decision:?}");
        }
    }

    #[tokio::test]
    async fn a_page_the_reply_has_read_is_not_downloaded_again() {
        let tools = WebTools::new(None);
        let _reply = tools.begin_reply("run", Provider::DuckDuckGo, true, Box::new(|_| true));
        let read_before = Url::parse("https://example.com/").unwrap();
        let slot = tools.with_reply(|reply| reply.fetch.slot(download_key(&read_before))).unwrap();
        *slot.lock().await = Some(Arc::new(page("Text").unwrap()));

        let again = read(Url::parse("http://Example.com").unwrap(), &tools).await.unwrap();

        assert_eq!(again.content.text, "Text");
    }

    #[tokio::test]
    async fn a_page_is_also_kept_under_the_address_a_redirect_led_to() {
        let tools = WebTools::new(None);
        let _reply = tools.begin_reply("run", Provider::DuckDuckGo, true, Box::new(|_| true));
        let requested = download_key(&Url::parse("http://example.com/old").unwrap());
        let moved = Arc::new(page_at("https://www.example.com/new/", "Text").unwrap());

        keep_under_final_address(&tools, &requested, &moved);

        let read_on = read(Url::parse("https://www.example.com/new/").unwrap(), &tools).await.unwrap();
        assert!(Arc::ptr_eq(&read_on, &moved));
    }

    fn known_from(
        role: uzu::types::session::chat::ChatRole,
        text: &str,
    ) -> Arc<WebTools> {
        let tools = WebTools::new(None);
        tools.remember_messages(&[test_message(role, text, None)], true);
        tools
    }

    #[test]
    fn only_an_address_the_conversation_showed_is_read_and_over_https_unless_shown_in_the_clear() {
        use uzu::types::session::chat::ChatRole;
        let tools = known_from(ChatRole::User {}, "Read https://docs.example/a#part and http://plain.example/b");
        let requested = |address: &str| requested_page(address, &tools).map(|page| page.url.to_string());

        assert_eq!(requested("https://docs.example/a#other"), Ok("https://docs.example/a".into()));
        assert_eq!(requested("http://docs.example/a"), Ok("https://docs.example/a".into()));
        assert_eq!(requested("http://plain.example/b"), Ok("http://plain.example/b".into()));
        assert_eq!(requested("https://docs.example/made-up"), Err(FetchError::NotShown));
        assert_eq!(requested("http://127.0.0.1/"), Err(FetchError::Address));
        assert_eq!(requested("docs.example/a"), Ok("https://docs.example/a".into()));
        assert_eq!(requested("docs.example/made-up"), Err(FetchError::NotShown));
        assert_eq!(requested("ftp://docs.example/a"), Err(FetchError::Address));
    }

    #[test]
    fn a_page_lends_its_final_address_and_its_links_and_the_users_address_stays_the_users() {
        use uzu::types::session::chat::ChatRole;
        let tools = known_from(ChatRole::User {}, "Read https://docs.example/private");
        let written_by_user =
            |address: &str| tools.with_known(|known| known.was_written_by_user(&Url::parse(address).unwrap()));
        let openable = |address: &str| tools.with_known(|known| known.is_openable(&Url::parse(address).unwrap()));

        let from_user = requested_page("https://docs.example/private", &tools).unwrap();
        assert!(from_user.written_by_user);
        let led_to = Url::parse("https://docs.example/private?token=secret").unwrap();
        remember_read(&tools, &from_user, &led_to, "See https://docs.example/next and https://other.example/x?id=1");
        assert!(written_by_user(led_to.as_str()), "where the user's address led can be as private");
        assert!(openable("https://docs.example/next"));
        assert!(!openable("https://other.example/x?id=1"), "a link to another host carries no query");
        assert!(!written_by_user("https://docs.example/next"));

        let found = requested_page("https://docs.example/next", &tools).unwrap();
        assert!(!found.written_by_user);
        let moved = Url::parse("https://docs.example/moved").unwrap();
        remember_read(&tools, &found, &moved, "");
        assert!(openable(moved.as_str()) && !written_by_user(moved.as_str()));
    }

    #[test]
    fn one_download_serves_both_schemes_of_an_address_and_no_other_address() {
        let key = |address: &str| download_key(&Url::parse(address).unwrap());
        assert_eq!(key("http://Example.com"), key("https://example.com/"));
        assert_ne!(key("https://example.com/docs"), key("https://example.com/docs/"));
        assert_ne!(key("https://example.com/docs"), key("https://example.com/docs?page=2"));
    }

    #[test]
    fn the_conversation_shows_the_requested_address_and_then_the_page_that_was_read() {
        let call = |arguments: serde_json::Value| ToolCall {
            identifier: None,
            name: WEB_FETCH_TOOL_NAME.to_string(),
            arguments: arguments.into(),
        };
        assert_eq!(
            requested_address(&call(serde_json::json!({ "url": " https://example.com/a " }))),
            "https://example.com/a"
        );
        assert_eq!(requested_address(&call(serde_json::json!({ "url": "http://127.0.0.1/" }))), "");
        assert_eq!(requested_address(&call(serde_json::json!({ "address": "https://example.com/" }))), "");

        let read =
            serde_json::json!({ "url": "https://example.com/b", "title": "", "text": "Text", "truncated": false });
        assert_eq!(
            page_source(&read),
            Some(WebSource {
                title: "https://example.com/b".into(),
                url: "https://example.com/b".into(),
            })
        );
        assert_eq!(page_source(&serde_json::json!({ "error": "The page could not be loaded (HTTP 500)" })), None);
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_pages_are_read_as_markdown_in_parts_from_known_addresses_only() {
        let tools = WebTools::new(None);
        let tool = web_fetch_tool(tools.clone());
        let read = |arguments: serde_json::Value| async {
            let tool = &tool;
            tool.execute(arguments.into()).await.map(|value| serde_json::to_value(value).unwrap())
        };
        let _reply = tools.begin_reply("live", Provider::DuckDuckGo, true, Box::new(|_| true));
        tools.remember_messages(
            &[test_message(
                uzu::types::session::chat::ChatRole::User {},
                "Read https://blog.rust-lang.org/2025/01/09/Rust-1.84.0/ and \
                 https://en.wikipedia.org/wiki/Rust_(programming_language)",
                None,
            )],
            true,
        );

        let unknown = read(serde_json::json!({ "url": "https://example.com/made-up" })).await;
        let Err(refusal) = unknown else {
            panic!("unknown address refused")
        };
        let message = refusal.to_string();
        assert!(message.contains("Only a URL"), "{message}");

        let blog =
            read(serde_json::json!({ "url": "https://blog.rust-lang.org/2025/01/09/Rust-1.84.0/" })).await.unwrap();
        assert!(blog["text"].as_str().unwrap().contains("## "), "headings kept as Markdown");
        assert!(blog["text"].as_str().unwrap().contains("](https://"), "links kept with addresses");
        assert_eq!(blog["truncated"], false);

        let wiki = read(serde_json::json!({ "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)" }))
            .await
            .unwrap();
        assert_eq!(wiki["truncated"], true);
        let next = wiki["nextStart"].as_u64().unwrap();
        let more = read(
            serde_json::json!({ "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)", "start": next }),
        )
        .await
        .unwrap();
        assert_eq!(more["start"].as_u64().unwrap(), next);
        assert!(!more["text"].as_str().unwrap().is_empty());

        let linked =
            read(serde_json::json!({ "url": "https://en.wikipedia.org/wiki/Programming_paradigm" })).await.unwrap();
        assert!(!linked["text"].as_str().unwrap().is_empty());
        let refused =
            read(serde_json::json!({ "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)" })).await;
        let Err(refusal) = refused else {
            panic!("the reply's text allowance is spent")
        };
        let message = refusal.to_string();
        assert!(message.contains("as much page text"), "{message}");
    }
}
