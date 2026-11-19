use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Decision {
    Deny,
    Allow,
    AlwaysAllow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequest {
    pub request_id: String,
    pub provider: &'static str,
    pub queries: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<String>,
}

impl PermissionRequest {
    fn items(&self) -> usize {
        self.queries.len() + self.pages.len()
    }
}

pub(in crate::chat) enum Subject<'a> {
    Search(&'a str),
    Page(&'a str),
}

pub(in crate::chat) type ShowRequest = Box<dyn Fn(PermissionRequest) -> bool + Send + Sync>;

enum Mode {
    Ask,
    Allowed,
    /// For the rest of the reply: a model that keeps trying does not bring the question back.
    Denied,
}

struct Waiting {
    request: PermissionRequest,
    allowed: watch::Sender<Option<bool>>,
}

struct State {
    mode: Mode,
    pages_denied: bool,
    waiting: Option<Waiting>,
    requests: u32,
}

pub(in crate::chat) struct Permission {
    show: ShowRequest,
    state: Mutex<State>,
}

impl Permission {
    pub(in crate::chat) fn new(
        asks: bool,
        show: ShowRequest,
    ) -> Self {
        Self {
            show,
            state: Mutex::new(State {
                mode: if asks {
                    Mode::Ask
                } else {
                    Mode::Allowed
                },
                pages_denied: false,
                waiting: None,
                requests: 0,
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("search permission mutex poisoned")
    }

    pub(in crate::chat) async fn allows(
        &self,
        run_id: &str,
        provider: &'static str,
        subject: Subject<'_>,
    ) -> bool {
        let mut allowed = {
            let mut state = self.state();
            let state = &mut *state;
            match (&state.mode, &subject) {
                (Mode::Denied, _) => return false,
                (_, Subject::Page(_)) if state.pages_denied => return false,
                (Mode::Allowed, Subject::Search(_)) => return true,
                (Mode::Ask, _) | (Mode::Allowed, Subject::Page(_)) => {},
            }
            let waiting = state.waiting.get_or_insert_with(|| {
                state.requests += 1;
                Waiting {
                    request: PermissionRequest {
                        request_id: format!("{run_id}:{}", state.requests),
                        provider,
                        queries: Vec::new(),
                        pages: Vec::new(),
                    },
                    allowed: watch::channel(None).0,
                }
            });
            match subject {
                Subject::Search(query) => waiting.request.queries.push(query.to_string()),
                Subject::Page(address) => waiting.request.pages.push(address.to_string()),
            }
            let allowed = waiting.allowed.subscribe();
            if !(self.show)(waiting.request.clone()) {
                state.mode = Mode::Denied;
                state.waiting = None;
                return false;
            }
            allowed
        };
        allowed.wait_for(Option::is_some).await.is_ok_and(|allowed| *allowed == Some(true))
    }

    /// False when the request the user saw is not the one waiting: it was answered, or a tool
    /// call joined it after it was shown. With an address in it, "always allow" only allows.
    pub(in crate::chat) fn answer(
        &self,
        request_id: &str,
        shown_items: usize,
        decision: Decision,
    ) -> bool {
        let mut state = self.state();
        let Some(waiting) = state
            .waiting
            .as_ref()
            .filter(|waiting| waiting.request.request_id == request_id && waiting.request.items() == shown_items)
        else {
            return false;
        };
        let decision = match decision {
            Decision::AlwaysAllow if !waiting.request.pages.is_empty() => Decision::Allow,
            decision => decision,
        };
        let asked_to_search = !waiting.request.queries.is_empty();
        match decision {
            // Refusing an address is not refusing to search.
            Decision::Deny if asked_to_search => state.mode = Mode::Denied,
            Decision::Deny => state.pages_denied = true,
            Decision::Allow => {},
            Decision::AlwaysAllow => state.mode = Mode::Allowed,
        }
        if let Some(waiting) = state.waiting.take() {
            waiting.allowed.send_replace(Some(decision != Decision::Deny));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use super::*;

    fn asking(asks: bool) -> (Permission, mpsc::UnboundedReceiver<PermissionRequest>) {
        let (shown, requests) = mpsc::unbounded_channel();
        (Permission::new(asks, Box::new(move |request| shown.send(request).is_ok())), requests)
    }

    fn answer(
        permission: &Permission,
        request: &PermissionRequest,
        decision: Decision,
    ) -> bool {
        permission.answer(&request.request_id, request.items(), decision)
    }

    #[tokio::test]
    async fn the_searches_of_one_turn_share_a_request_and_its_answer() {
        let (permission, mut requests) = asking(true);
        let user = async {
            let first = requests.recv().await.unwrap();
            let both = requests.recv().await.unwrap();
            assert_eq!(first.queries, ["first"]);
            assert_eq!(both.queries, ["first", "second"]);
            assert_eq!((both.provider, &both.request_id), ("DuckDuckGo", &first.request_id));
            assert!(!answer(&permission, &first, Decision::Allow), "the second search was not shown yet");
            assert!(answer(&permission, &both, Decision::Allow));
            assert!(!answer(&permission, &both, Decision::Deny), "answered once");
            both.request_id
        };
        let (first, second, request_id) = tokio::join!(
            permission.allows("run", "DuckDuckGo", Subject::Search("first")),
            permission.allows("run", "DuckDuckGo", Subject::Search("second")),
            user,
        );
        assert!(first && second);

        let user = async {
            let next = requests.recv().await.unwrap();
            assert_eq!(next.queries, ["third"]);
            assert_ne!(next.request_id, request_id, "the next turn asks again");
            assert!(answer(&permission, &next, Decision::Deny));
        };
        let (third, ()) = tokio::join!(permission.allows("run", "DuckDuckGo", Subject::Search("third")), user);
        assert!(!third);
        assert!(
            !permission.allows("run", "DuckDuckGo", Subject::Search("fourth")).await,
            "a refusal holds for the reply"
        );
        assert!(!permission.allows("run", "DuckDuckGo", Subject::Page("https://example.com/")).await);
        assert!(requests.try_recv().is_err());
    }

    #[tokio::test]
    async fn always_allow_and_a_chat_that_does_not_ask_stop_the_question_about_searches() {
        let (permission, mut requests) = asking(true);
        let user = async {
            let request = requests.recv().await.unwrap();
            assert!(answer(&permission, &request, Decision::AlwaysAllow));
        };
        let (first, ()) = tokio::join!(permission.allows("run", "Exa", Subject::Search("first")), user);
        assert!(first);
        assert!(permission.allows("run", "Exa", Subject::Search("second")).await);
        assert!(requests.try_recv().is_err());

        let (off, mut requests) = asking(false);
        assert!(off.allows("run", "Exa", Subject::Search("query")).await);
        assert!(requests.try_recv().is_err());
    }

    #[tokio::test]
    async fn an_address_the_user_wrote_is_asked_about_even_when_searches_are_not() {
        let (permission, mut requests) = asking(false);
        let address = "https://docs.example/d/1?key=secret";
        for _ in 0..2 {
            let user = async {
                let request = requests.recv().await.unwrap();
                assert_eq!((request.queries.len(), request.pages.as_slice()), (0, [address.to_string()].as_slice()));
                assert!(answer(&permission, &request, Decision::AlwaysAllow));
            };
            let (page, ()) = tokio::join!(permission.allows("run", "Exa", Subject::Page(address)), user);
            assert!(page);
        }
        assert!(permission.allows("run", "Exa", Subject::Search("still not asked")).await);
        assert!(requests.try_recv().is_err());

        let user = async {
            let request = requests.recv().await.unwrap();
            assert!(answer(&permission, &request, Decision::Deny));
        };
        let (page, ()) = tokio::join!(permission.allows("run", "Exa", Subject::Page(address)), user);
        assert!(!page);
        assert!(!permission.allows("run", "Exa", Subject::Page("https://docs.example/d/2")).await);
        assert!(permission.allows("run", "Exa", Subject::Search("searching goes on")).await);
        assert!(requests.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_request_that_cannot_be_shown_is_a_refusal() {
        let hidden = Permission::new(true, Box::new(|_| false));
        assert!(!hidden.allows("run", "Exa", Subject::Search("query")).await);
        assert!(!hidden.answer("run:1", 1, Decision::Allow));
    }
}
