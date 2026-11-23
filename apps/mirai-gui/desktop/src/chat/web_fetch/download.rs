use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};

use encoding_rs::{Encoding, UTF_8};
use reqwest::{
    StatusCode, Url,
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, LOCATION},
    redirect,
};
use tokio::time::Instant;

use super::{FetchError, extract::extract};
use crate::chat::{
    messages::clip_chars,
    web_address::{NotPublic, PublicAddresses, is_public_web},
    web_search,
    web_source::{PageContent, title_line},
    web_tools::WebTools,
    web_view::BROWSER_USER_AGENT,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_REDIRECTS: usize = 5;
const MAX_DOWNLOAD_BYTES: usize = 5 * 1024 * 1024;
const SCRIPT_DRAWN_MAX_CHARS: usize = 400;
const PUBLISHED_AT_MAX_CHARS: usize = 40;

#[derive(Debug)]
pub(super) struct ReadPage {
    pub(super) url: Url,
    pub(super) content: PageContent,
    pub(super) length: usize,
    pub(super) reader: Reader,
}

impl ReadPage {
    pub(super) fn new(
        url: Url,
        mut content: PageContent,
        reader: Reader,
    ) -> Result<Self, FetchError> {
        content.title = title_line(&content.title);
        content.published_at = content.published_at.map(|date| clip_chars(&date, PUBLISHED_AT_MAX_CHARS));
        let length = content.text.chars().count();
        if length == 0 {
            return Err(FetchError::Empty);
        }
        Ok(Self {
            url,
            content,
            length,
            reader,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum Reader {
    Plain,
    Provider,
}

fn request_error(error: reqwest::Error) -> FetchError {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(current) = cause {
        if current.downcast_ref::<NotPublic>().is_some() {
            return FetchError::Address;
        }
        cause = current.source();
    }
    FetchError::Request(error.without_url().to_string())
}

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .user_agent(BROWSER_USER_AGENT)
        .dns_resolver(Arc::new(PublicAddresses))
        .no_proxy()
        .redirect(redirect::Policy::none())
        .build()
        .expect("web fetch HTTP client")
});

/// The page is never opened in the browser window on this device: there its
/// scripts could send requests into the user's network.
pub(super) async fn fetch(
    url: Url,
    tools: &WebTools,
) -> Result<ReadPage, FetchError> {
    let plain = fetch_plain(url.clone()).await;
    if !is_worth_another_reader(&plain) {
        return plain;
    }
    better_of(plain, read_by_provider(&url, tools).await)
}

fn is_worth_another_reader(plain: &Result<ReadPage, FetchError>) -> bool {
    match plain {
        Ok(page) => page.length < SCRIPT_DRAWN_MAX_CHARS,
        Err(FetchError::Refused(_) | FetchError::Empty | FetchError::Request(_)) => true,
        Err(FetchError::Status(status)) => *status >= 500,
        Err(
            FetchError::Address
            | FetchError::NotShown
            | FetchError::NotText(_)
            | FetchError::PastEnd(_)
            | FetchError::TextAllowanceSpent
            | FetchError::NoReply(_),
        ) => false,
    }
}

fn better_of(
    plain: Result<ReadPage, FetchError>,
    by_provider: Option<Result<ReadPage, FetchError>>,
) -> Result<ReadPage, FetchError> {
    match (plain, by_provider) {
        (Ok(plain), Some(Ok(other))) if other.length > plain.length => Ok(other),
        (Err(_), Some(Ok(other))) => Ok(other),
        (plain, _) => plain,
    }
}

async fn read_by_provider(
    url: &Url,
    tools: &WebTools,
) -> Option<Result<ReadPage, FetchError>> {
    let read = web_search::read_page(tools, url, super::TEXT_PER_REPLY_CHARS).await?;
    Some(match read {
        Ok(content) => ReadPage::new(url.clone(), content, Reader::Provider),
        Err(error) => {
            crate::logger::info("web-fetch:provider", Some(serde_json::json!({ "error": error.log_entry() })));
            Err(FetchError::Request("the search provider could not read it".into()))
        },
    })
}

async fn fetch_plain(url: Url) -> Result<ReadPage, FetchError> {
    if let Some(secure) = https_spelling(&url) {
        let answered = load(secure).await;
        if !has_no_https(&answered) {
            return answered;
        }
        crate::logger::info("web-fetch:https-first", None);
    }
    load(url).await
}

fn https_spelling(url: &Url) -> Option<Url> {
    let mut secure = url.clone();
    (url.scheme() == "http" && secure.set_scheme("https").is_ok()).then_some(secure)
}

fn has_no_https(answered: &Result<ReadPage, FetchError>) -> bool {
    matches!(answered, Err(FetchError::Request(_)))
}

async fn load(mut url: Url) -> Result<ReadPage, FetchError> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let mut redirects = 0;
    let mut response = loop {
        // Some sites answer with a Markdown rendering, which is shorter and needs no extraction.
        let request = HTTP_CLIENT
            .get(url.clone())
            .header(ACCEPT, "text/markdown,text/html;q=0.9,application/xhtml+xml;q=0.8,text/plain;q=0.7")
            .timeout(deadline.saturating_duration_since(Instant::now()));
        let response = request.send().await.map_err(request_error)?;
        let Some(location) = redirect_location(response.status(), response.headers()) else {
            break response;
        };
        if redirects == MAX_REDIRECTS {
            return Err(FetchError::Request("too many redirects".into()));
        }
        redirects += 1;
        url = next_hop(&url, location)?;
    };
    let content_type =
        response.headers().get(CONTENT_TYPE).and_then(|value| value.to_str().ok()).unwrap_or_default().to_lowercase();
    let is_html = is_html_page(response.status(), &content_type)?;
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(request_error)? {
        let room = MAX_DOWNLOAD_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if body.len() == MAX_DOWNLOAD_BYTES {
            break;
        }
    }
    let page_url = url.clone();
    let content = tokio::task::spawn_blocking(move || {
        let page = decode(&body, &content_type);
        if is_html {
            extract(&page, &page_url)
        } else {
            PageContent {
                text: page.trim().to_string(),
                ..PageContent::default()
            }
        }
    })
    .await
    .map_err(|error| FetchError::Request(error.to_string()))?;
    ReadPage::new(url, content, Reader::Plain)
}

fn redirect_location(
    status: StatusCode,
    headers: &HeaderMap,
) -> Option<&str> {
    let redirects = matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    );
    headers.get(LOCATION).filter(|_| redirects)?.to_str().ok()
}

fn next_hop(
    from: &Url,
    location: &str,
) -> Result<Url, FetchError> {
    let mut to = from.join(location).map_err(|_| FetchError::Request("unreadable redirect".into()))?;
    if from.scheme() == "https" && to.scheme() == "http" {
        let _ = to.set_scheme("https");
    }
    if is_public_web(&to) {
        Ok(to)
    } else {
        Err(FetchError::Address)
    }
}

fn is_html_page(
    status: StatusCode,
    content_type: &str,
) -> Result<bool, FetchError> {
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::TOO_MANY_REQUESTS) {
        return Err(FetchError::Refused(status.as_u16()));
    }
    if !status.is_success() {
        return Err(FetchError::Status(status.as_u16()));
    }
    // Without a header a browser sniffs the body; a page is the common case.
    let is_html = content_type.is_empty() || content_type.contains("html");
    if !is_html && !content_type.starts_with("text/") {
        let kind = content_type.split(';').next().unwrap_or_default().trim();
        return Err(FetchError::NotText(if kind.is_empty() {
            "unknown type".into()
        } else {
            kind.into()
        }));
    }
    Ok(is_html)
}

const CHARSET_SNIFF_BYTES: usize = 1024;

fn decode(
    body: &[u8],
    content_type: &str,
) -> String {
    let declared = |text: &[u8]| {
        let lower = text.to_ascii_lowercase();
        let start = lower.windows(8).position(|window| window == b"charset=")? + 8;
        let label: Vec<u8> = lower[start..]
            .iter()
            .skip_while(|byte| matches!(byte, b'"' | b'\'' | b' '))
            .take_while(|byte| !matches!(byte, b'"' | b'\'' | b';' | b'>' | b' ' | b'/' | b'\n' | b'\r' | b'\t'))
            .copied()
            .collect();
        Encoding::for_label(&label)
    };
    let encoding = Encoding::for_bom(body)
        .map(|(encoding, _)| encoding)
        .or_else(|| declared(content_type.as_bytes()))
        .or_else(|| declared(&body[..body.len().min(CHARSET_SNIFF_BYTES)]))
        .unwrap_or(UTF_8);
    encoding.decode_with_bom_removal(body).0.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(address: &str) -> Url {
        Url::parse(address).unwrap()
    }

    fn page(
        text: &str,
        reader: Reader,
    ) -> Result<ReadPage, FetchError> {
        let content = PageContent {
            title: "Title".into(),
            text: text.into(),
            published_at: None,
            cut_short: false,
        };
        ReadPage::new(url("https://example.com/"), content, reader)
    }

    #[test]
    fn only_a_redirect_status_with_a_location_is_followed() {
        let location = |status: u16, header: Option<&str>| {
            let mut headers = HeaderMap::new();
            if let Some(header) = header {
                headers.insert(LOCATION, header.parse().unwrap());
            }
            redirect_location(StatusCode::from_u16(status).unwrap(), &headers).map(str::to_string)
        };
        for redirect in [301, 302, 303, 307, 308] {
            assert_eq!(location(redirect, Some("/next")).as_deref(), Some("/next"), "{redirect}");
        }
        assert_eq!(location(302, None), None);
        for other in [200, 201, 300, 304, 404] {
            assert_eq!(location(other, Some("/next")), None, "{other}");
        }
    }

    #[test]
    fn a_redirect_leads_only_to_a_public_web_address() {
        let from = url("https://example.com/docs/page");
        assert_eq!(next_hop(&from, "/moved").unwrap(), url("https://example.com/moved"));
        assert_eq!(next_hop(&from, "other?x=1").unwrap(), url("https://example.com/docs/other?x=1"));
        assert_eq!(next_hop(&from, "https://www.example.org/").unwrap(), url("https://www.example.org/"));
        for local in [
            "http://127.0.0.1/admin",
            "https://192.168.1.1/",
            "//10.0.0.5/",
            "http://[::1]/",
            "https://user:secret@example.org/",
            "ftp://example.org/file",
            "file:///etc/hosts",
            "javascript:alert(1)",
        ] {
            assert_eq!(next_hop(&from, local).unwrap_err(), FetchError::Address, "{local}");
        }
    }

    #[test]
    fn a_redirect_does_not_take_an_https_page_into_the_clear() {
        let secure = url("https://example.com/");
        assert_eq!(next_hop(&secure, "http://example.com/plain").unwrap(), url("https://example.com/plain"));
        assert_eq!(next_hop(&secure, "http://other.example/").unwrap(), url("https://other.example/"));

        let plain = url("http://example.com/");
        assert_eq!(next_hop(&plain, "http://www.example.com/").unwrap(), url("http://www.example.com/"));
        assert_eq!(next_hop(&plain, "https://example.com/").unwrap(), url("https://example.com/"));
    }

    #[test]
    fn an_http_address_is_tried_over_https_first_and_in_the_clear_only_when_https_is_unreachable() {
        assert_eq!(https_spelling(&url("http://example.com/a?b=1")), Some(url("https://example.com/a?b=1")));
        assert_eq!(https_spelling(&url("https://example.com/")), None);

        assert!(has_no_https(&Err(FetchError::Request("connection refused".into()))));
        for answered in [
            page("Text", Reader::Plain),
            Err(FetchError::Status(404)),
            Err(FetchError::Refused(403)),
            Err(FetchError::Address),
            Err(FetchError::Empty),
        ] {
            assert!(!has_no_https(&answered), "{answered:?}");
        }
    }

    #[test]
    fn an_answer_is_a_page_only_with_a_success_status_and_a_text_type() {
        let kind = |status: u16, content_type: &str| is_html_page(StatusCode::from_u16(status).unwrap(), content_type);
        assert_eq!(kind(200, "text/html; charset=utf-8"), Ok(true));
        assert_eq!(kind(200, "application/xhtml+xml"), Ok(true));
        assert_eq!(kind(200, ""), Ok(true));
        assert_eq!(kind(200, "text/markdown"), Ok(false));
        assert_eq!(kind(200, "text/plain; charset=utf-8"), Ok(false));
        assert_eq!(kind(200, "application/pdf"), Err(FetchError::NotText("application/pdf".into())));
        assert_eq!(kind(200, "; charset=utf-8"), Err(FetchError::NotText("unknown type".into())));
        for refused in [401, 403, 429] {
            assert_eq!(kind(refused, "text/html"), Err(FetchError::Refused(refused)));
        }
        for failed in [404, 500, 304] {
            assert_eq!(kind(failed, "text/html"), Err(FetchError::Status(failed)));
        }
    }

    #[test]
    fn a_page_without_text_is_refused_and_its_title_is_one_line() {
        assert_eq!(page("", Reader::Plain).unwrap_err(), FetchError::Empty);

        let content = PageContent {
            title: " Two\n lines ".into(),
            text: "Text".into(),
            published_at: Some("x".repeat(100)),
            cut_short: false,
        };
        let read = ReadPage::new(url("https://example.com/"), content, Reader::Plain).unwrap();
        assert_eq!(read.content.title, "Two lines");
        assert_eq!(read.content.published_at.unwrap().chars().count(), PUBLISHED_AT_MAX_CHARS);
    }

    #[test]
    fn another_reader_is_tried_only_for_failures_it_could_get_past() {
        for worth in [
            FetchError::Refused(403),
            FetchError::Empty,
            FetchError::Request("timeout".into()),
            FetchError::Status(503),
        ] {
            assert!(is_worth_another_reader(&Err(worth)));
        }
        for pointless in [FetchError::Status(404), FetchError::Address, FetchError::NotText("application/pdf".into())] {
            assert!(!is_worth_another_reader(&Err(pointless)));
        }
        assert!(is_worth_another_reader(&page("Loading", Reader::Plain)));
        assert!(!is_worth_another_reader(&page(&"word ".repeat(SCRIPT_DRAWN_MAX_CHARS), Reader::Plain)));
    }

    #[test]
    fn the_providers_page_is_taken_when_it_has_more_text_or_the_plain_request_failed() {
        let reader = |outcome: Result<ReadPage, FetchError>| outcome.map(|page| page.reader);
        let short = || page("Loading", Reader::Plain);
        let refused = || Err(FetchError::Refused(403));
        let by_provider = |text: &str| Some(page(text, Reader::Provider));

        assert_eq!(reader(better_of(short(), by_provider("The whole article"))), Ok(Reader::Provider));
        assert_eq!(reader(better_of(short(), by_provider("Less"))), Ok(Reader::Plain));
        assert_eq!(reader(better_of(refused(), by_provider("Text"))), Ok(Reader::Provider));
        assert_eq!(reader(better_of(short(), None)), Ok(Reader::Plain));
        assert_eq!(reader(better_of(refused(), None)), Err(FetchError::Refused(403)));
        let unread = Some(Err(FetchError::Request("the search provider could not read it".into())));
        assert_eq!(reader(better_of(refused(), unread)), Err(FetchError::Refused(403)));
    }

    #[test]
    fn pages_are_decoded_by_bom_header_or_their_own_charset_and_utf8_otherwise() {
        let cyrillic_1251: &[u8] = &[0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2];
        assert_eq!(decode(cyrillic_1251, "text/html; charset=windows-1251"), "Привет");
        assert_eq!(decode(cyrillic_1251, "text/html; charset=\"Windows-1251\""), "Привет");

        let meta = [b"<html><head><meta charset='windows-1251'></head><body>".as_slice(), cyrillic_1251].concat();
        assert_eq!(decode(&meta, "text/html"), "<html><head><meta charset='windows-1251'></head><body>Привет");
        let http_equiv = [
            b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=windows-1251\">".as_slice(),
            cyrillic_1251,
        ]
        .concat();
        assert!(decode(&http_equiv, "text/html").ends_with("Привет"));

        let utf8 = [b"<meta charset=windows-1251>".as_slice(), "Привет".as_bytes()].concat();
        assert!(decode(&utf8, "text/html; charset=utf-8").ends_with("Привет"));

        let bom_utf16 =
            [&[0xff, 0xfe][..], &"Hi".encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>()].concat();
        assert_eq!(decode(&bom_utf16, "text/html; charset=windows-1251"), "Hi");

        assert_eq!(decode("Привет".as_bytes(), "text/html"), "Привет");
        assert_eq!(decode(b"plain", "text/plain; charset=unknown-thing"), "plain");
    }
}
