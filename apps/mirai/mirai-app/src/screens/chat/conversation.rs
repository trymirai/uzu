//! The chat conversation model: messages, their regenerate-versions, and the
//! projection sent to the engine for a reply.

use std::cell::RefCell;

use crate::components::markdown::ParsedMarkdown;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Role {
    User,
    Assistant,
}

/// One generated version of an assistant reply (regenerate keeps the prior
/// ones so they can be paged through). User messages have exactly one.
#[derive(Clone, Default)]
pub(super) struct Version {
    pub text: String,
    pub reasoning: Option<String>,
    pub tps: Option<f32>,
    pub tokens: Option<u32>,
    /// Seconds to the first generated token, and total reply duration (from
    /// uzu's `ChatReplyStats`). Shown in the per-message performance popover.
    pub ttft: Option<f32>,
    pub total_time: Option<f32>,
    pub error: bool,
    /// Display name of the model that produced this version.
    pub model_name: Option<String>,
    /// Memoized markdown parse of `text`, tagged with the text it was built for.
    /// The view rebuilds elements each frame but only re-parses when `text`
    /// changes — see [`Version::parsed_markdown`].
    pub parsed: RefCell<Option<(String, ParsedMarkdown)>>,
}

impl Version {
    /// Borrow the parsed markdown for `self.text`, parsing (and caching) only
    /// when the text differs from the cached one.
    pub(super) fn parsed_markdown(&self) -> std::cell::Ref<'_, ParsedMarkdown> {
        {
            let mut cache = self.parsed.borrow_mut();
            let stale = cache.as_ref().map(|(t, _)| t != &self.text).unwrap_or(true);
            if stale {
                *cache = Some((self.text.clone(), crate::components::markdown::parse(&self.text)));
            }
        }
        std::cell::Ref::map(self.parsed.borrow(), |c| &c.as_ref().unwrap().1)
    }
}

pub(super) struct ChatMsg {
    pub role: Role,
    pub versions: Vec<Version>,
    pub current: usize,
    /// Whether the reasoning panel is collapsed. Starts expanded (false) while
    /// streaming; auto-collapses once the reply body text starts arriving.
    pub reasoning_collapsed: bool,
}

impl ChatMsg {
    pub(super) fn user(text: String) -> Self {
        Self {
            role: Role::User,
            versions: vec![Version {
                text,
                ..Default::default()
            }],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    pub(super) fn assistant(version: Version) -> Self {
        Self {
            role: Role::Assistant,
            versions: vec![version],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    pub(super) fn cur(&self) -> &Version {
        &self.versions[self.current]
    }

    pub(super) fn cur_mut(&mut self) -> &mut Version {
        &mut self.versions[self.current]
    }
}

/// The conversation to send for a reply: prior turns only — the trailing
/// assistant placeholder being filled is dropped, as are any errored turns —
/// each as its current version's `(role, text)`.
pub(super) fn conversation_for_request(messages: &[ChatMsg]) -> Vec<(Role, String)> {
    let upto = messages.len().saturating_sub(1);
    messages[..upto].iter().filter(|m| !m.cur().error).map(|m| (m.role, m.cur().text.clone())).collect()
}
