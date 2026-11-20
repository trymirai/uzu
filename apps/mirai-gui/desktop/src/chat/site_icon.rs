use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::{StatusCode, Url, header::LOCATION, redirect};

use crate::chat::{
    web_address::{PublicAddresses, Resolution, is_public_web, resolution},
    web_view::BROWSER_USER_AGENT,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_REDIRECTS: usize = 3;
const MAX_ICON_BYTES: usize = 64 * 1024;
const MISSING_RETRY_AFTER: Duration = Duration::from_secs(7 * 24 * 60 * 60);

// Sent without cookies or a referrer: the site learns only that its icon was asked for.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .user_agent(BROWSER_USER_AGENT)
        .dns_resolver(Arc::new(PublicAddresses))
        .no_proxy()
        .redirect(redirect::Policy::none())
        .timeout(REQUEST_TIMEOUT)
        .build()
        .expect("site icon HTTP client")
});

pub(super) async fn site_icon(host: &str) -> Option<String> {
    let url = icon_url(host)?;
    let cached = cache_path(host);
    if let Some(path) = &cached
        && let Some(bytes) = read_cache(path)
    {
        return data_url(&bytes);
    }
    let bytes = match download(url).await {
        Fetched::Icon(bytes) => bytes,
        Fetched::Missing => Vec::new(),
        Fetched::Unreachable => return None,
    };
    if let Some(path) = &cached
        && let Err(error) = crate::storage::write_atomic(path, &bytes)
    {
        crate::logger::warn("site-icon:cache", Some(serde_json::json!({ "error": error.to_string() })));
    }
    data_url(&bytes)
}

enum Fetched {
    Icon(Vec<u8>),
    Missing,
    Unreachable,
}

fn icon_url(host: &str) -> Option<Url> {
    let url = Url::parse(&format!("https://{host}/favicon.ico")).ok()?;
    let named = url.domain().is_some_and(|domain| domain == host.to_ascii_lowercase()) && url.port().is_none();
    (named && is_public_web(&url)).then_some(url)
}

fn cache_path(host: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library/Caches/mirai/site-icons").join(host.to_ascii_lowercase()))
}

/// An empty file records a site without an icon until it is due to be asked again.
fn read_cache(path: &PathBuf) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.is_empty() {
        return Some(bytes);
    }
    let age = std::fs::metadata(path).and_then(|meta| meta.modified()).ok()?.elapsed().ok()?;
    (age < MISSING_RETRY_AFTER).then_some(bytes)
}

async fn download(mut url: Url) -> Fetched {
    let mut redirects = 0;
    let mut response = loop {
        let resolved = url.clone();
        match tokio::task::spawn_blocking(move || resolution(&resolved)).await {
            Ok(Resolution::Public) => {},
            Ok(Resolution::NotPublic) => return Fetched::Missing,
            Ok(Resolution::Failed) | Err(_) => return Fetched::Unreachable,
        }
        let Ok(response) = HTTP_CLIENT.get(url.clone()).send().await else {
            return Fetched::Unreachable;
        };
        if !response.status().is_redirection() {
            break response;
        }
        let next = response.headers().get(LOCATION).and_then(|location| location.to_str().ok());
        let Some(next) = next.and_then(|location| url.join(location).ok()).filter(is_public_web) else {
            return Fetched::Missing;
        };
        if redirects == MAX_REDIRECTS {
            return Fetched::Missing;
        }
        redirects += 1;
        url = next;
    };
    let status = response.status();
    if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS {
        return Fetched::Unreachable;
    }
    if status != StatusCode::OK {
        return Fetched::Missing;
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => body.extend_from_slice(&chunk),
            Ok(None) => break,
            Err(_) => return Fetched::Unreachable,
        }
        if body.len() > MAX_ICON_BYTES {
            return Fetched::Missing;
        }
    }
    if image_type(&body).is_some() {
        Fetched::Icon(body)
    } else {
        Fetched::Missing
    }
}

/// Sites answer a missing icon with an HTML page as often as with a 404, so the bytes decide.
fn image_type(bytes: &[u8]) -> Option<&'static str> {
    match bytes {
        [0, 0, 1, 0, ..] => Some("image/x-icon"),
        [0x89, b'P', b'N', b'G', ..] => Some("image/png"),
        [b'G', b'I', b'F', b'8', ..] => Some("image/gif"),
        [0xFF, 0xD8, 0xFF, ..] => Some("image/jpeg"),
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => Some("image/webp"),
        _ => None,
    }
}

fn data_url(bytes: &[u8]) -> Option<String> {
    let mime = image_type(bytes)?;
    Some(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_public_domain_names_get_an_icon_address() {
        assert_eq!(icon_url("www.theguardian.com").unwrap().as_str(), "https://www.theguardian.com/favicon.ico");
        for host in ["127.0.0.1", "192.168.1.1", "example.com:8080", "example.com/path", "a b.com", ""] {
            assert!(icon_url(host).is_none(), "{host}");
        }
    }

    #[test]
    fn an_html_page_is_not_taken_for_an_icon() {
        assert_eq!(image_type(&[0, 0, 1, 0, 1, 0]), Some("image/x-icon"));
        assert_eq!(image_type(b"\x89PNG\r\n"), Some("image/png"));
        assert_eq!(image_type(b"<!doctype html>"), None);
        assert_eq!(data_url(b""), None);
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_a_site_icon_is_downloaded() {
        let url = icon_url("en.wikipedia.org").unwrap();
        assert!(matches!(download(url).await, Fetched::Icon(_)));
    }
}
