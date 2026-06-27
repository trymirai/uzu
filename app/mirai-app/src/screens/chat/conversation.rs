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
            versions: vec![Version { text, ..Default::default() }],
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
    messages[..upto]
        .iter()
        .filter(|m| !m.cur().error)
        .map(|m| (m.role, m.cur().text.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, text: &str, error: bool) -> ChatMsg {
        ChatMsg {
            role,
            versions: vec![Version { text: text.into(), error, ..Default::default() }],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    #[test]
    fn user_message_has_single_version() {
        let m = ChatMsg::user("hi".into());
        assert_eq!(m.versions.len(), 1);
        assert_eq!(m.current, 0);
        assert_eq!(m.cur().text, "hi");
    }

    // Regenerate pushes a new version and points `current` at it, keeping priors
    // reachable via the pager.
    #[test]
    fn regenerate_keeps_prior_versions() {
        let mut m = ChatMsg::assistant(Version { text: "v0".into(), ..Default::default() });
        m.versions.push(Version { text: "v1".into(), ..Default::default() });
        m.current = m.versions.len() - 1;
        assert_eq!(m.versions.len(), 2);
        assert_eq!(m.cur().text, "v1");
        m.current = 0;
        assert_eq!(m.cur().text, "v0");
    }

    #[test]
    fn request_excludes_trailing_placeholder() {
        let messages = vec![
            msg(Role::User, "q1", false),
            msg(Role::Assistant, "a1", false),
            msg(Role::User, "q2", false),
            msg(Role::Assistant, "", false), // placeholder being filled
        ];
        let convo = conversation_for_request(&messages);
        assert_eq!(convo.len(), 3);
        assert_eq!(convo[0].1, "q1");
        assert_eq!(convo[2].1, "q2");
    }

    #[test]
    fn request_drops_errored_turns() {
        let messages = vec![
            msg(Role::User, "q1", false),
            msg(Role::Assistant, "boom", true), // errored — excluded from history
            msg(Role::User, "q2", false),
            msg(Role::Assistant, "", false),
        ];
        let convo = conversation_for_request(&messages);
        assert_eq!(convo.len(), 2);
        assert_eq!(convo[0].1, "q1");
        assert_eq!(convo[1].1, "q2");
    }

    #[test]
    fn request_empty_when_only_placeholder() {
        assert!(conversation_for_request(&[msg(Role::Assistant, "", false)]).is_empty());
    }
}
