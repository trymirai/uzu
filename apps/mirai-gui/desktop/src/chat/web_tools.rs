use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use super::{
    known_urls::{KnownUrls, Origin},
    payloads::MsgIn,
    web_fetch, web_search,
    web_view::WebView,
};

pub(super) fn millis_since(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(super) struct WebTools {
    pub(super) web_view: Option<WebView>,
    reply: Mutex<Option<ReplyScope>>,
    known: Mutex<KnownUrls>,
    pub(super) scraped_search_pacing: tokio::sync::Mutex<Option<Instant>>,
}

pub(super) struct Reply(Arc<WebTools>);

impl Drop for Reply {
    fn drop(&mut self) {
        self.0.reply.lock().expect("web reply mutex poisoned").take();
    }
}

pub(super) struct ReplyScope {
    pub(super) run_id: String,
    pub(super) search: web_search::ReplyState,
    pub(super) fetch: web_fetch::ReplyState,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("No reply is in progress")]
pub(super) struct NoReply;

impl WebTools {
    pub(super) fn new(web_view: Option<WebView>) -> Arc<Self> {
        Arc::new(Self {
            web_view,
            reply: Mutex::new(None),
            known: Mutex::new(KnownUrls::default()),
            scraped_search_pacing: tokio::sync::Mutex::const_new(None),
        })
    }

    pub(super) fn with_reply<T>(
        &self,
        f: impl FnOnce(&mut ReplyScope) -> T,
    ) -> Result<T, NoReply> {
        let mut reply = self.reply.lock().expect("web reply mutex poisoned");
        reply.as_mut().map(f).ok_or(NoReply)
    }

    pub(super) fn with_known<T>(
        &self,
        f: impl FnOnce(&mut KnownUrls) -> T,
    ) -> T {
        let mut known = self.known.lock().expect("known addresses mutex poisoned");
        f(&mut known)
    }

    #[must_use]
    pub(super) fn begin_reply(
        self: &Arc<Self>,
        run_id: &str,
        provider: web_search::Provider,
        asks_before_searching: bool,
        show_request: web_search::ShowRequest,
    ) -> Reply {
        *self.reply.lock().expect("web reply mutex poisoned") = Some(ReplyScope {
            run_id: run_id.to_string(),
            search: web_search::ReplyState::new(provider, asks_before_searching, show_request),
            fetch: web_fetch::ReplyState::default(),
        });
        Reply(self.clone())
    }

    pub(super) fn reply_result_chars(&self) -> Option<usize> {
        self.with_reply(|reply| reply.search.result_chars + reply.fetch.result_chars).ok()
    }

    pub(super) fn remember_messages(
        &self,
        messages: &[MsgIn],
        replay: bool,
    ) {
        self.with_known(|known| {
            if replay {
                known.forget();
            }
            for message in messages {
                if message.role != (uzu::types::session::chat::ChatRole::Assistant {}) {
                    known.remember_text(&message.content, &Origin::User);
                }
                known.remember_addresses(message.openable_urls.iter().map(String::as_str), &Origin::Trusted);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use reqwest::Url;
    use uzu::types::session::chat::ChatRole;

    use super::{super::payloads::test_message, *};

    fn message(
        role: ChatRole,
        content: &str,
        openable_urls: &[&str],
    ) -> MsgIn {
        MsgIn {
            openable_urls: openable_urls.iter().map(ToString::to_string).collect(),
            ..test_message(role, content, None)
        }
    }

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn the_tools_work_only_while_a_run_holds_its_reply() {
        let tools = WebTools::new(None);
        assert_eq!(tools.with_reply(|_| ()), Err(NoReply));
        assert_eq!(tools.reply_result_chars(), None);

        let reply = tools.begin_reply("run-1", web_search::Provider::DuckDuckGo, true, Box::new(|_| true));
        assert_eq!(tools.with_reply(|reply| reply.run_id.clone()), Ok("run-1".into()));
        assert_eq!(tools.reply_result_chars(), Some(0));

        drop(reply);
        assert_eq!(tools.with_reply(|_| ()), Err(NoReply));
    }

    #[test]
    fn an_earlier_answer_lends_the_addresses_it_found_and_none_of_its_own_words() {
        let tools = WebTools::new(None);
        tools.remember_messages(
            &[
                message(ChatRole::User {}, "Compare https://shop.example/item?id=9 please.", &[]),
                message(
                    ChatRole::Assistant {},
                    "See https://example.com/cited?utm=1 and https://example.com/plain.",
                    &["https://example.com/found?page=2", "not an address"],
                ),
            ],
            true,
        );
        let openable = |address: &str| tools.with_known(|known| known.is_openable(&url(address)));
        assert!(openable("https://shop.example/item?id=9"), "the user's address, query and all");
        let written_by_user = |address: &str| tools.with_known(|known| known.was_written_by_user(&url(address)));
        assert!(written_by_user("https://shop.example/item?id=9"));
        assert!(!written_by_user("https://example.com/found?page=2"));
        assert!(openable("https://example.com/found?page=2"), "a search result the answer came from");
        assert!(!openable("https://example.com/plain"), "the model's own link");
        assert!(!openable("https://example.com/cited?utm=1"), "the model's own link with a query");
        assert!(!openable("https://example.com/other"));

        tools.remember_messages(&[message(ChatRole::User {}, "Open https://example.com/other", &[])], false);
        assert!(openable("https://example.com/other"), "a continuation adds without forgetting");
        assert!(openable("https://shop.example/item?id=9"));
    }

    #[test]
    fn the_tools_are_registered_under_the_names_the_transcript_looks_for() {
        let tools = WebTools::new(None);
        assert_eq!(web_search::web_search_tool(tools.clone()).name, web_search::WEB_SEARCH_TOOL_NAME);
        assert_eq!(web_fetch::web_fetch_tool(tools).name, web_fetch::WEB_FETCH_TOOL_NAME);
    }
}
