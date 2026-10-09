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

const NAMING_INSTRUCTION: &str = "Use set_chat_name to give this chat a concise, descriptive name in your first turn. Update the name when the conversation's main topic changes or the user asks you to rename it. Choose a short name of 3 to 50 characters. Set hidden to true for autonomous naming housekeeping, and false when the user explicitly asks you to name or rename the chat. Answer the user's request normally; do not mention autonomous housekeeping or ask for permission to name the chat.";

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
                /// A concise, descriptive chat name of 3 to 50 characters.
                name: String,
                /// True for autonomous housekeeping; false when the user explicitly requests naming or renaming.
                hidden: bool
            | -> Result<String, ErrorFuture> {
                // The transcript reads visibility from the recorded call.
                let _ = hidden;
                let name = normalize_name(&name)?;
                let mut run = state.lock().expect("chat naming mutex poisoned");
                let channel = run.channel.as_ref().ok_or("No chat is currently running")?;
                channel.send(RunEvent::ChatName { name: name.clone() })?;
                run.name = Some(name);
                Ok("Chat name accepted.".to_string())
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

fn normalize_name(name: &str) -> Result<String, ErrorFuture> {
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    let name = name.trim_matches(['"', '\'', '`']).trim();
    // Match JavaScript's string length and the manual rename limit.
    if !(3..=50).contains(&name.encode_utf16().count()) || name.chars().any(char::is_control) {
        return Err("Chat names must contain 3 to 50 characters without control characters".into());
    }
    Ok(name.to_string())
}

pub(super) fn add_naming_instruction(messages: &mut Vec<ChatMessage>) {
    if let Some(system) = messages.iter_mut().find(|message| message.role == (ChatRole::System {})) {
        *system = system.with_text(format!("\n\n{NAMING_INSTRUCTION}"));
    } else {
        messages.insert(0, ChatMessage::system().with_text(NAMING_INSTRUCTION.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_names_and_matches_manual_rename_limits() {
        assert_eq!(normalize_name("  `Exploring\n  London` ").unwrap(), "Exploring London");
        assert!(normalize_name("\" \"").is_err());
        assert!(normalize_name("ab").is_err());
        assert!(normalize_name(&"x".repeat(51)).is_err());
        assert!(normalize_name(&"😀".repeat(25)).is_ok());
        assert!(normalize_name(&"😀".repeat(26)).is_err());
        assert!(normalize_name("bad\0name").is_err());
    }

    #[test]
    fn naming_instruction_keeps_the_existing_system_prompt() {
        let mut messages = vec![ChatMessage::system().with_text("Be helpful.".to_string())];
        add_naming_instruction(&mut messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text().unwrap(), format!("Be helpful.\n\n{NAMING_INSTRUCTION}"));
    }

    #[tokio::test]
    async fn tool_only_names_the_active_run_and_rejects_invalid_names() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let arguments = |name: &str| serde_json::json!({ "name": name, "hidden": true }).into();
        assert!(tool.execute(arguments("Before run")).await.is_err());
        let received = Arc::new(Mutex::new(Vec::new()));
        let events = received.clone();
        let run = naming.begin(Channel::new(move |body| {
            events.lock().unwrap().push(body);
            Ok(())
        }));
        assert!(tool.execute(arguments("x")).await.is_err());
        assert!(run.name().is_none());
        assert!(tool.execute(serde_json::json!({ "name": "Missing flag" }).into()).await.is_err());
        assert!(tool.execute(serde_json::json!({ "name": "Bad flag", "hidden": "maybe" }).into()).await.is_err());
        tool.execute(arguments("  First\nname ")).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("First name"));
        tool.execute(serde_json::json!({ "name": "Changed name", "hidden": false }).into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("Changed name"));
        assert_eq!(received.lock().unwrap().len(), 2);
        drop(run);
        assert!(tool.execute(arguments("After run")).await.is_err());
        let next = naming.begin(Channel::new(|_| Ok(())));
        assert!(next.name().is_none());
    }

    #[tokio::test]
    async fn naming_tool_requires_hidden_and_accepts_the_standard_boolean_coercion() {
        let naming = ChatNaming::default();
        let tool = naming.tool();
        let schema: serde_json::Value = tool.parameters.clone().unwrap().try_into().unwrap();
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("hidden")));
        let run = naming.begin(Channel::new(|_| Ok(())));
        tool.execute(serde_json::json!({ "name": "Markup name", "hidden": " TRUE " }).into()).await.unwrap();
        assert_eq!(run.name().as_deref(), Some("Markup name"));
    }
}
