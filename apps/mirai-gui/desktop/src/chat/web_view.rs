use std::{
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use reqwest::Url;
use serde::{Deserialize, de::DeserializeOwned};
use tauri::{
    AppHandle, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
    utils::config::BackgroundThrottlingPolicy,
    webview::{NewWindowResponse, PageLoadEvent, PermissionResponse},
};
use tokio::sync::{mpsc, oneshot};

use super::{
    web_address::{is_public_web, resolves_to_public},
    web_tools::millis_since,
};

pub(super) const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
     (KHTML, like Gecko) Version/18.0 Safari/605.1.15";

const LOAD_TIMEOUT: Duration = Duration::from_secs(20);
const CHECK_TIMEOUT: Duration = Duration::from_secs(8);
const POLL: Duration = Duration::from_millis(500);
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);
const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 900.0;
const MAX_HTML_CHARS: usize = 5_000_000;

const NO_DIALOGS_SCRIPT: &str = "window.alert = window.confirm = window.prompt = window.print = function () {}; \
                            window.open = function () { return null; };";

static HTML_SCRIPT: LazyLock<String> =
    LazyLock::new(|| format!("document.documentElement.outerHTML.slice(0, {MAX_HTML_CHARS})"));

static NEXT_LABEL: AtomicU64 = AtomicU64::new(0);
static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Clone)]
pub(super) struct WebView(pub(super) AppHandle);

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum WebViewError {
    #[error("the address is not a public web page over HTTPS")]
    Address,
    #[error("the browser window could not be opened: {0}")]
    Window(String),
    #[error("the page did not load: {0}")]
    Load(String),
    #[error("the site's bot check did not let the page through")]
    Challenged,
}

impl WebViewError {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Address => "address",
            Self::Window(_) => "window",
            Self::Load(_) => "load",
            Self::Challenged => "challenged",
        }
    }
}

#[derive(Debug, Deserialize)]
struct Probe {
    url: String,
    loaded: bool,
    check: bool,
}

impl Probe {
    fn script(check_selector: &str) -> String {
        let selector = serde_json::Value::from(check_selector);
        format!(
            "({{ url: location.href, loaded: document.readyState === 'complete', \
             check: !!document.querySelector({selector}) }})"
        )
    }

    fn is_worth_another_look(
        &self,
        waited: Duration,
    ) -> bool {
        (self.check || !self.loaded) && waited < CHECK_TIMEOUT
    }
}

impl WebView {
    /// Only a search provider's own address is opened here: the window does not check the
    /// requests a page's scripts make.
    pub(super) async fn render(
        &self,
        url: &Url,
        check_selector: &str,
    ) -> Result<String, WebViewError> {
        let _turn = ONE_AT_A_TIME.lock().await;
        let started = Instant::now();
        let outcome = self.load_and_read(url, check_selector).await;
        crate::logger::info(
            "web-view:done",
            Some(serde_json::json!({
                "chars": outcome.as_ref().ok().map(String::len),
                "error": outcome.as_ref().err().map(WebViewError::kind),
                "durationMs": millis_since(started),
            })),
        );
        outcome
    }

    async fn load_and_read(
        &self,
        url: &Url,
        check_selector: &str,
    ) -> Result<String, WebViewError> {
        if !is_public_page(url).await {
            return Err(WebViewError::Address);
        }
        let (loaded, mut loads) = mpsc::unbounded_channel();
        let window = Closing(self.open(url, loaded)?);

        let _ = tokio::time::timeout(LOAD_TIMEOUT, loads.recv()).await;
        let waiting_since = Instant::now();
        let script = Probe::script(check_selector);
        let mut probe: Probe = evaluate(&window.0, &script).await?;
        while probe.is_worth_another_look(waiting_since.elapsed()) {
            tokio::time::sleep(POLL).await;
            probe = evaluate(&window.0, &script).await?;
        }
        if probe.check {
            return Err(WebViewError::Challenged);
        }
        if !probe.loaded {
            return Err(WebViewError::Load("the page did not finish loading".into()));
        }
        // Resolves the name, which `may_navigate` could not.
        let landed = Url::parse(&probe.url).map_err(|_| WebViewError::Load("the page has no address".into()))?;
        if !is_public_page(&landed).await {
            return Err(WebViewError::Address);
        }
        evaluate(&window.0, &HTML_SCRIPT).await
    }

    fn open(
        &self,
        url: &Url,
        loaded: mpsc::UnboundedSender<()>,
    ) -> Result<WebviewWindow, WebViewError> {
        let label = format!("web-view-{}", NEXT_LABEL.fetch_add(1, Ordering::Relaxed));
        WebviewWindowBuilder::new(&self.0, label, WebviewUrl::External(url.clone()))
            .title("Web view")
            .inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
            .visible(false)
            .focused(false)
            .incognito(true)
            .devtools(false)
            .user_agent(BROWSER_USER_AGENT)
            .background_throttling(BackgroundThrottlingPolicy::Disabled)
            .initialization_script(NO_DIALOGS_SCRIPT)
            .on_navigation(may_navigate)
            .on_new_window(|_, _| NewWindowResponse::Deny)
            .on_download(|_, _| false)
            .on_permission_request(|_, _| PermissionResponse::Deny)
            .on_page_load(move |_, payload| {
                if matches!(payload.event(), PageLoadEvent::Finished) {
                    let _ = loaded.send(());
                }
            })
            .build()
            .map_err(|error| WebViewError::Window(error.to_string()))
    }
}

async fn is_public_page(url: &Url) -> bool {
    let resolved = url.clone();
    may_navigate(url) && tokio::task::spawn_blocking(move || resolves_to_public(&resolved)).await.unwrap_or(false)
}

/// A navigation callback runs on the main thread and cannot resolve names; HTTPS keeps a
/// local service out until the landing address is resolved after the load.
fn may_navigate(target: &Url) -> bool {
    is_public_web(target) && target.scheme() == "https"
}

async fn evaluate<T: DeserializeOwned>(
    window: &WebviewWindow,
    script: &str,
) -> Result<T, WebViewError> {
    let (answer, answered) = oneshot::channel();
    let answer = Mutex::new(Some(answer));
    window
        .eval_with_callback(script, move |value| {
            if let Some(answer) = answer.lock().expect("web view answer mutex poisoned").take() {
                let _ = answer.send(value);
            }
        })
        .map_err(|error| WebViewError::Load(error.to_string()))?;
    let value = tokio::time::timeout(ANSWER_TIMEOUT, answered)
        .await
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| WebViewError::Load("the page did not answer".into()))?;
    serde_json::from_str(&value).map_err(|_| WebViewError::Load("the page could not be read".into()))
}

struct Closing(WebviewWindow);

impl Drop for Closing {
    fn drop(&mut self) {
        if let Err(error) = self.0.destroy() {
            crate::logger::warn("web-view:close", Some(serde_json::json!({ "error": error.to_string() })));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_redirect_may_only_lead_to_a_public_page_over_https() {
        assert!(may_navigate(&Url::parse("https://example.com/next").unwrap()));
        assert!(may_navigate(&Url::parse("https://8.8.8.8/").unwrap()));
        for refused in [
            "http://example.com/",
            "https://127.0.0.1/",
            "https://10.0.0.1/",
            "file:///etc/hosts",
            "https://user:pw@example.com/",
        ] {
            assert!(!may_navigate(&Url::parse(refused).unwrap()), "{refused}");
        }
    }

    #[test]
    fn the_window_waits_for_a_check_to_pass_and_the_page_to_arrive_but_not_for_long() {
        let probe = |loaded: bool, check: bool| Probe {
            url: String::new(),
            loaded,
            check,
        };
        assert!(!probe(true, false).is_worth_another_look(Duration::ZERO));
        assert!(probe(true, true).is_worth_another_look(Duration::from_secs(7)));
        assert!(probe(false, false).is_worth_another_look(Duration::from_secs(7)));
        assert!(!probe(true, true).is_worth_another_look(CHECK_TIMEOUT));

        let script = Probe::script(".anomaly-modal, #challenge-form");
        assert!(script.contains(r#"document.querySelector(".anomaly-modal, #challenge-form")"#), "{script}");
    }
}
