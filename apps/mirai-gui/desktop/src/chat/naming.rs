use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;
use uzu::{
    session::tool::{
        func_def::{ErrorFuture, ToolDescriptor},
        uzu_tool_closure,
    },
    types::session::chat::{ChatMessage, ChatRole},
};

use super::payloads::RunEvent;

// Approximate prose budget for the sidebar's 186px label at 13px Inter.
// Glyph widths vary, so wider characters still rely on the UI's ellipsis.
pub(super) const TITLE_TARGET_LENGTH: usize = 25;
const TITLE_SOFT_LIMIT: usize = 28;
pub(super) const TITLE_STYLE_INSTRUCTION: &str = "Write a natural, human-readable phrase with normal capitalization and spaces between words. \
     Do not format it as a hashtag, an underscore-separated identifier, a hyphen-separated slug, or a CamelCase identifier. \
     If the user explicitly supplies an exact name, preserve it.";

fn naming_instruction() -> String {
    format!(
        "Use set_chat_name to give this chat a concise, descriptive name in your first turn. \
         Update the name when the conversation's main topic changes or the user asks you to rename it. \
         {TITLE_STYLE_INSTRUCTION} \
         Aim for about {TITLE_TARGET_LENGTH} characters so the name fits in the sidebar. \
         This is an approximate display budget, not a required length. \
         If the tool warns that the name is too long, call it again with a shorter name before finishing this reply. \
         Set hidden to true for autonomous naming housekeeping, and false when the user explicitly asks you to name or rename the chat. \
         Answer the user's request normally; do not mention autonomous housekeeping or ask for permission to name the chat."
    )
}

#[derive(Default)]
struct ActiveRun {
    channel: Option<Channel<RunEvent>>,
    name: Option<String>,
}

#[derive(Clone, Default)]
pub(super) struct ChatNaming(Arc<Mutex<ActiveRun>>);

pub(super) struct NamingRun(Arc<Mutex<ActiveRun>>);

impl ChatNaming {
    pub(super) fn begin(
        &self,
        channel: Channel<RunEvent>,
    ) -> NamingRun {
        *self.0.lock().expect("chat naming mutex poisoned") = ActiveRun {
            channel: Some(channel),
            name: None,
        };
        NamingRun(self.0.clone())
    }

    pub(super) fn tool(&self) -> ToolDescriptor {
        let state = self.0.clone();
        uzu_tool_closure! {
            /// Set or update the name of the current chat.
            set_chat_name: async move |
                /// A short, human-readable phrase with normal capitalization and spaces between words.
                /// Avoid hashtags, underscore-separated identifiers, hyphen-separated slugs, and CamelCase identifiers.
                /// Preserve an explicitly requested exact name; otherwise aim to fit the provided character budget.
                name: String,
                /// True for autonomous housekeeping; false when the user explicitly requests naming or renaming.
                hidden: bool
            | -> Result<String, ErrorFuture> {
                // The transcript reads visibility from the recorded call.
                let _ = hidden;
                let name = normalize_name(&name)?;
                let warning = title_length_warning(&name);
                let mut run = state.lock().expect("chat naming mutex poisoned");
                let channel = run.channel.as_ref().ok_or("No chat is currently running")?;
                channel.send(RunEvent::ChatName { name: name.clone() })?;
                run.name = Some(name);
                Ok(warning.unwrap_or_else(|| "Chat name accepted.".to_string()))
            }
        }
    }
}

impl NamingRun {
    pub(super) fn name(&self) -> Option<String> {
        self.0.lock().expect("chat naming mutex poisoned").name.clone()
    }
}

impl Drop for NamingRun {
    fn drop(&mut self) {
        self.0.lock().expect("chat naming mutex poisoned").channel = None;
    }
}

pub(super) fn normalize_name(title: &str) -> Result<String, ErrorFuture> {
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = title.trim_matches(['"', '\'', '`']).trim();
    if title.is_empty() || title.chars().any(char::is_control) {
        return Err("Chat names must not be empty or contain control characters".into());
    }
    Ok(title.to_string())
}

fn title_length_warning(title: &str) -> Option<String> {
    let length = title.chars().count();
    (length > TITLE_SOFT_LIMIT).then(|| {
        format!(
            "Name saved, but may be clipped: it has {length} characters, above the {TITLE_SOFT_LIMIT}-character display budget. \
             Aim for about {TITLE_TARGET_LENGTH} characters. \
             Call set_chat_name again with a shorter phrase before finishing your reply; shorten the wording, keeping normal spaces. \
             If you finish without replacing it, this name will be used and clipped in the interface as needed."
        )
    })
}

pub(super) fn add_naming_instruction(messages: &mut Vec<ChatMessage>) {
    let instruction = naming_instruction();
    if let Some(system) = messages.iter_mut().find(|message| message.role == (ChatRole::System {})) {
        *system = system.with_text(format!("\n\n{instruction}"));
    } else {
        messages.insert(0, ChatMessage::system().with_text(instruction));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(name: &str) -> serde_json::Value {
        serde_json::json!({ "name": name, "hidden": true })
    }

    fn capture(naming: &ChatNaming) -> (NamingRun, Arc<Mutex<Vec<serde_json::Value>>>) {
        let received = Arc::new(Mutex::new(Vec::new()));
        let events = received.clone();
        let run = naming.begin(Channel::new(move |body| {
            let tauri::ipc::InvokeResponseBody::Json(json) = body else {
                panic!("Expected JSON event")
            };
            events.lock().unwrap().push(serde_json::from_str(&json).unwrap());
            Ok(())
        }));
        (run, received)
    }

    #[test]
    fn normalization_does_not_treat_the_display_budget_as_a_hard_limit() {
        assert_eq!(normalize_name("  `Exploring\n  London` ").unwrap(), "Exploring London");
        assert!(normalize_name("\" \"").is_err());
        assert_eq!(normalize_name("X").unwrap(), "X");
        assert!(normalize_name(&"x".repeat(TITLE_SOFT_LIMIT + 1)).is_ok());
        assert!(normalize_name(&"x".repeat(51)).is_ok());
        assert!(normalize_name("bad\0title").is_err());
        assert!(title_length_warning(&"😀".repeat(TITLE_SOFT_LIMIT)).is_none());
        let warning = title_length_warning(&"x".repeat(TITLE_SOFT_LIMIT + 3)).unwrap();
        assert!(warning.contains(&format!("has {} characters", TITLE_SOFT_LIMIT + 3)));
        assert!(warning.contains("before finishing your reply"));
    }

    #[test]
    fn naming_instruction_keeps_the_existing_system_prompt() {
        let mut messages = vec![ChatMessage::system().with_text("Be helpful.".to_string())];
        add_naming_instruction(&mut messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text().unwrap(), format!("Be helpful.\n\n{}", naming_instruction()));
    }

    #[test]
    fn title_guidance_allows_some_room_above_the_target() {
        assert!(naming_instruction().contains("Aim for about 25 characters"));
        assert!(title_length_warning(&"x".repeat(25)).is_none());
        assert!(title_length_warning(&"😀".repeat(28)).is_none());
        let warning = title_length_warning(&"😀".repeat(29)).unwrap();
        assert!(warning.contains("has 29 characters"));
        assert!(warning.contains("above the 28-character display budget"));
        assert!(warning.contains("Aim for about 25 characters"));
    }

    #[tokio::test]
    async fn tool_only_names_the_active_run_and_requires_name_and_visibility() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        assert!(tool.execute(arguments("Before run").into()).await.is_err());
        let (run, received) = capture(&naming);
        assert!(tool.execute(arguments("").into()).await.is_err());
        assert!(run.name().is_none());
        for invalid in [
            serde_json::json!({ "hidden": true }),
            serde_json::json!({ "name": "Name" }),
            serde_json::json!({ "name": "Name", "hidden": "maybe" }),
        ] {
            assert!(tool.execute(invalid.into()).await.is_err());
        }
        tool.execute(arguments("  First\nname ").into()).await.unwrap();
        tool.execute(serde_json::json!({ "name": "Changed name", "hidden": false }).into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("Changed name"));
        assert_eq!(received.lock().unwrap().len(), 2);
        assert_eq!(received.lock().unwrap()[0], serde_json::json!({ "type": "chatName", "name": "First name" }));
        drop(run);
        assert!(tool.execute(arguments("After run").into()).await.is_err());
        let next = naming.begin(Channel::new(|_| Ok(())));
        assert!(next.name().is_none());
    }

    #[tokio::test]
    async fn over_budget_names_remain_replaceable_across_tool_continuations() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let (run, received) = capture(&naming);
        let too_long = "x".repeat(TITLE_SOFT_LIMIT + 1);
        let result = tool.execute(arguments(&too_long).into()).await.unwrap();
        let feedback = serde_json::Value::try_from(result).unwrap();
        assert!(feedback.as_str().unwrap().contains("Call set_chat_name again"));
        assert_eq!(received.lock().unwrap().len(), 1);
        assert_eq!(run.name().as_deref(), Some(too_long.as_str()));
        // A second automatic tool continuation is still inside the same run.
        tool.execute(arguments("Shorter name").into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("Shorter name"));
        drop(run);
        assert_eq!(received.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn keeps_the_latest_over_budget_name_without_truncating_it() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let (run, received) = capture(&naming);
        tool.execute(arguments("Initial name").into()).await.unwrap();
        tool.execute(arguments(&"x".repeat(TITLE_SOFT_LIMIT + 2)).into()).await.unwrap();
        let latest = "x".repeat(TITLE_SOFT_LIMIT + 1);
        tool.execute(arguments(&latest).into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some(latest.as_str()));
        drop(run);
        assert_eq!(received.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn accepted_names_are_already_published_when_a_run_is_interrupted() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let (run, received) = capture(&naming);
        let too_long = "x".repeat(TITLE_SOFT_LIMIT + 1);
        tool.execute(arguments(&too_long).into()).await.unwrap();
        // A client may stop accepting events immediately after cancellation.
        assert_eq!(received.lock().unwrap().len(), 1);
        drop(run);
        assert_eq!(received.lock().unwrap().len(), 1);
        assert_eq!(received.lock().unwrap()[0]["name"], too_long);
        assert!(tool.execute(arguments("After run").into()).await.is_err());
    }

    #[tokio::test]
    async fn naming_tool_requires_hidden_and_accepts_the_standard_boolean_coercion() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let schema: serde_json::Value = tool.parameters.clone().unwrap().try_into().unwrap();
        for field in ["name", "hidden"] {
            assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!(field)));
        }
        let run = naming.begin(Channel::new(|_| Ok(())));
        tool.execute(serde_json::json!({ "name": "Markup name", "hidden": " TRUE " }).into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("Markup name"));
    }
}
