use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::types::basic::{ReasoningEffort, ToolCall, ToolNamespace, TranslationPayload, Value};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChatContentBlockType {
    Identity,
    ReasoningEffort,
    ConversationStartDate,
    KnowledgeCutoff,
    BuiltinTools,
    Tools,
    Text,
    Image,
    Video,
    Audio,
    File,
    Reasoning,
    ToolCall,
    ToolCallCandidate,
    ToolCallResult,
    Translation,
    Custom,
}

impl FromStr for ChatContentBlockType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "identity" => Ok(ChatContentBlockType::Identity),
            "reasoning_effort" => Ok(ChatContentBlockType::ReasoningEffort),
            "conversation_start_date" => Ok(ChatContentBlockType::ConversationStartDate),
            "knowledge_cutoff" => Ok(ChatContentBlockType::KnowledgeCutoff),
            "builtin_tools" => Ok(ChatContentBlockType::BuiltinTools),
            "tools" => Ok(ChatContentBlockType::Tools),
            "text" => Ok(ChatContentBlockType::Text),
            "image" => Ok(ChatContentBlockType::Image),
            "video" => Ok(ChatContentBlockType::Video),
            "audio" => Ok(ChatContentBlockType::Audio),
            "file" => Ok(ChatContentBlockType::File),
            "reasoning" => Ok(ChatContentBlockType::Reasoning),
            "tool_call" => Ok(ChatContentBlockType::ToolCall),
            "tool_call_candidate" => Ok(ChatContentBlockType::ToolCallCandidate),
            "tool_call_result" => Ok(ChatContentBlockType::ToolCallResult),
            "translation" => Ok(ChatContentBlockType::Translation),
            "custom" => Ok(ChatContentBlockType::Custom),
            other => Err(format!("Unknown content block type: {other}")),
        }
    }
}

impl Display for ChatContentBlockType {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            ChatContentBlockType::Identity => "identity",
            ChatContentBlockType::ReasoningEffort => "reasoning_effort",
            ChatContentBlockType::ConversationStartDate => "conversation_start_date",
            ChatContentBlockType::KnowledgeCutoff => "knowledge_cutoff",
            ChatContentBlockType::BuiltinTools => "builtin_tools",
            ChatContentBlockType::Tools => "tools",
            ChatContentBlockType::Text => "text",
            ChatContentBlockType::Image => "image",
            ChatContentBlockType::Video => "video",
            ChatContentBlockType::Audio => "audio",
            ChatContentBlockType::File => "file",
            ChatContentBlockType::Reasoning => "reasoning",
            ChatContentBlockType::ToolCall => "tool_call",
            ChatContentBlockType::ToolCallCandidate => "tool_call_candidate",
            ChatContentBlockType::ToolCallResult => "tool_call_result",
            ChatContentBlockType::Translation => "translation",
            ChatContentBlockType::Custom => "custom",
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for ChatContentBlockType {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for ChatContentBlockType {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        ChatContentBlockType::from_str(&name).map_err(de::Error::custom)
    }
}

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentBlock {
    Identity {
        value: String,
    },
    ReasoningEffort {
        value: ReasoningEffort,
    },
    ConversationStartDate {
        value: String,
    },
    KnowledgeCutoff {
        value: String,
    },
    BuiltinTools {
        names: Vec<String>,
    },
    Tools {
        namespaces: Vec<ToolNamespace>,
    },
    Text {
        value: String,
    },
    Image {
        url: String,
    },
    Video {
        url: String,
    },
    Audio {
        url: String,
    },
    File {
        url: String,
    },
    Reasoning {
        value: String,
    },
    ToolCall {
        value: ToolCall,
    },
    ToolCallCandidate {
        value: Value,
    },
    ToolCallResult {
        #[serde(rename = "id")]
        identifier: Option<String>,
        name: Option<String>,
        value: Value,
    },
    Translation {
        #[serde(flatten)]
        payload: TranslationPayload,
        source_language_code: String,
        target_language_code: String,
    },
    Custom {
        value: Value,
    },
}

impl ChatContentBlock {
    pub fn get_type(&self) -> ChatContentBlockType {
        match self {
            ChatContentBlock::Identity {
                ..
            } => ChatContentBlockType::Identity,
            ChatContentBlock::ReasoningEffort {
                ..
            } => ChatContentBlockType::ReasoningEffort,
            ChatContentBlock::ConversationStartDate {
                ..
            } => ChatContentBlockType::ConversationStartDate,
            ChatContentBlock::KnowledgeCutoff {
                ..
            } => ChatContentBlockType::KnowledgeCutoff,
            ChatContentBlock::BuiltinTools {
                ..
            } => ChatContentBlockType::BuiltinTools,
            ChatContentBlock::Tools {
                ..
            } => ChatContentBlockType::Tools,
            ChatContentBlock::Text {
                ..
            } => ChatContentBlockType::Text,
            ChatContentBlock::Image {
                ..
            } => ChatContentBlockType::Image,
            ChatContentBlock::Video {
                ..
            } => ChatContentBlockType::Video,
            ChatContentBlock::Audio {
                ..
            } => ChatContentBlockType::Audio,
            ChatContentBlock::File {
                ..
            } => ChatContentBlockType::File,
            ChatContentBlock::Reasoning {
                ..
            } => ChatContentBlockType::Reasoning,
            ChatContentBlock::ToolCall {
                ..
            } => ChatContentBlockType::ToolCall,
            ChatContentBlock::ToolCallCandidate {
                ..
            } => ChatContentBlockType::ToolCallCandidate,
            ChatContentBlock::ToolCallResult {
                ..
            } => ChatContentBlockType::ToolCallResult,
            ChatContentBlock::Translation {
                ..
            } => ChatContentBlockType::Translation,
            ChatContentBlock::Custom {
                ..
            } => ChatContentBlockType::Custom,
        }
    }
}
