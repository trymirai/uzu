use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uzu::{
    session::tool::{
        func_def::{ErrorFuture, ToolDescriptor},
        uzu_tool_closure,
    },
    types::basic::ToolCall,
};

use super::{SearchError, SearchResult, Subject, search};
use crate::chat::{
    known_urls::Origin,
    web_source::WebSource,
    web_tools::{WebTools, millis_since},
};

pub(in crate::chat) const WEB_SEARCH_TOOL_NAME: &str = "web_search";
const QUERY_MAX_CHARS: usize = 400;
const DENIED: &str = "The user did not allow this web search. Answer without it.";

pub(in crate::chat) fn web_search_tool(tools: Arc<WebTools>) -> ToolDescriptor {
    uzu_tool_closure! {
        /// Returns page titles, URLs and short snippets.
        web_search: async move |
            query: String
        | -> Result<SearchOutcome, ErrorFuture> {
            let query = normalize_query(&query)?;
            let (run_id, provider, permission) = tools
                .with_reply(|reply| (reply.run_id.clone(), reply.search.provider, reply.search.permission.clone()))
                .map_err(|error| error.to_string())?;
            if !permission.allows(&run_id, provider.label(), Subject::Search(&query)).await {
                return Ok(SearchOutcome::Failure {
                    error: DENIED.to_string(),
                    failure: Some(SearchFailure::Denied),
                });
            }
            let started = std::time::Instant::now();
            let outcome = search(provider, &query, &tools).await;
            crate::logger::info(
                "web-search:done",
                Some(serde_json::json!({
                    "runId": run_id,
                    "provider": provider,
                    "results": outcome.as_ref().ok().map(Vec::len),
                    "error": outcome.as_ref().err().map(SearchError::log_entry),
                    "durationMs": millis_since(started),
                })),
            );
            Ok(match outcome {
                Ok(results) => {
                    tools.with_known(|known| {
                        known.remember_addresses(results.iter().map(|result| result.url.as_str()), &Origin::Trusted);
                    });
                    let chars = serde_json::to_string(&results).map_or(0, |json| json.chars().count());
                    let _ = tools.with_reply(|reply| reply.search.result_chars += chars);
                    SearchOutcome::Results(results)
                },
                Err(error) => {
                    let provider = provider.label().to_string();
                    let (error, failure) = match error {
                        SearchError::RateLimited(_) => (
                            format!("{error}; another provider can be chosen in Settings"),
                            Some(SearchFailure::Limit {
                                provider,
                            }),
                        ),
                        SearchError::NoKey(_) | SearchError::KeyRejected(_) => (
                            error.to_string(),
                            Some(SearchFailure::Key {
                                provider,
                            }),
                        ),
                        SearchError::KeyUnreadable(_) => (error.to_string(), Some(SearchFailure::Keychain)),
                        _ => (error.to_string(), None),
                    };
                    SearchOutcome::Failure {
                        error,
                        failure,
                    }
                },
            })
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
enum SearchOutcome {
    Results(Vec<SearchResult>),
    Failure {
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure: Option<SearchFailure>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SearchFailure {
    Limit {
        provider: String,
    },
    Key {
        provider: String,
    },
    Denied,
    Keychain,
}

pub(in crate::chat) fn searched_query(call: &ToolCall) -> String {
    #[derive(Deserialize)]
    struct Arguments {
        query: String,
    }
    serde_json::from_str::<Arguments>(&call.arguments.json)
        .ok()
        .and_then(|arguments| normalize_query(&arguments.query).ok())
        .unwrap_or_default()
}

#[derive(Debug, Default, PartialEq)]
pub(in crate::chat) struct SearchView {
    pub sources: Vec<WebSource>,
    pub failure: Option<SearchFailure>,
}

pub(in crate::chat) fn search_view(result: &serde_json::Value) -> SearchView {
    match SearchOutcome::deserialize(result) {
        Ok(SearchOutcome::Results(results)) => SearchView {
            sources: results
                .into_iter()
                .map(|result| WebSource {
                    title: if result.title.trim().is_empty() {
                        result.url.clone()
                    } else {
                        result.title
                    },
                    url: result.url,
                })
                .collect(),
            ..SearchView::default()
        },
        Ok(SearchOutcome::Failure {
            failure,
            ..
        }) => SearchView {
            sources: Vec::new(),
            failure,
        },
        Err(_) => SearchView::default(),
    }
}

fn normalize_query(query: &str) -> Result<String, ErrorFuture> {
    let query = crate::chat::messages::collapse_whitespace(query);
    if query.is_empty() {
        return Err("The search query must not be empty".into());
    }
    if query.chars().count() > QUERY_MAX_CHARS {
        return Err(format!("The search query must be at most {QUERY_MAX_CHARS} characters").into());
    }
    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_are_collapsed_to_one_line_and_bounded() {
        assert_eq!(normalize_query("  rust \n stable\tversion ").unwrap(), "rust stable version");
        assert!(normalize_query(" \n ").is_err());
        assert!(normalize_query(&"a".repeat(QUERY_MAX_CHARS)).is_ok());
        assert!(normalize_query(&"a".repeat(QUERY_MAX_CHARS + 1)).is_err());
    }

    #[test]
    fn the_query_is_read_from_the_recorded_call() {
        let call = |arguments: serde_json::Value| ToolCall {
            identifier: None,
            name: WEB_SEARCH_TOOL_NAME.to_string(),
            arguments: arguments.into(),
        };
        assert_eq!(searched_query(&call(serde_json::json!({ "query": " rust\nstable " }))), "rust stable");
        assert_eq!(searched_query(&call(serde_json::json!({ "q": "wrong key" }))), "");
    }

    #[test]
    fn sources_are_the_pages_of_a_result_and_an_untitled_page_shows_its_address() {
        let result = serde_json::json!([
            { "title": "Rust Versions", "url": "https://releases.rs/", "snippet": "Stable", "publishedAt": "2026-10-05" },
            { "title": " ", "url": "http://example.com/untitled", "snippet": "" },
        ]);
        assert_eq!(
            search_view(&result).sources,
            [
                WebSource {
                    title: "Rust Versions".into(),
                    url: "https://releases.rs/".into(),
                },
                WebSource {
                    title: "http://example.com/untitled".into(),
                    url: "http://example.com/untitled".into(),
                },
            ]
        );
    }

    #[test]
    fn a_failed_search_is_serialized_with_its_reason_and_read_back_for_the_conversation() {
        let limit = SearchFailure::Limit {
            provider: "Exa".into(),
        };
        let limited = serde_json::to_value(SearchOutcome::Failure {
            error: "Exa search limit reached".into(),
            failure: Some(limit.clone()),
        })
        .unwrap();
        assert_eq!(
            limited,
            serde_json::json!({ "error": "Exa search limit reached", "failure": { "kind": "limit", "provider": "Exa" } })
        );
        let view = search_view(&limited);
        assert_eq!((view.failure, view.sources.len()), (Some(limit), 0));

        assert_eq!(
            search_view(&serde_json::json!({ "error": "The search query must not be empty" })),
            SearchView::default()
        );
        assert_eq!(search_view(&serde_json::json!([])), SearchView::default());

        let denied = serde_json::to_value(SearchOutcome::Failure {
            error: DENIED.into(),
            failure: Some(SearchFailure::Denied),
        })
        .unwrap();
        assert_eq!(denied, serde_json::json!({ "error": DENIED, "failure": { "kind": "denied" } }));
        assert_eq!(search_view(&denied).failure, Some(SearchFailure::Denied));

        let unreadable = serde_json::json!({ "error": "unreadable", "failure": { "kind": "keychain" } });
        assert_eq!(search_view(&unreadable).failure, Some(SearchFailure::Keychain));

        let no_key = serde_json::json!({
            "error": "Brave needs an API key, which is added in Settings",
            "failure": { "kind": "key", "provider": "Brave" },
        });
        assert_eq!(
            search_view(&no_key).failure,
            Some(SearchFailure::Key {
                provider: "Brave".into()
            })
        );
    }
}
