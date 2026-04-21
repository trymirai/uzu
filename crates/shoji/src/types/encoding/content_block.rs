use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::types::{
    basic::Value,
    encoding::{ReasoningEffort, ToolCall, ToolNamespace, TranslationInput},
};

#[bindings::export(Enum, name = "ContentBlockType")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContentBlockType {
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

impl FromStr for ContentBlockType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "identity" => Ok(ContentBlockType::Identity),
            "reasoning_effort" => Ok(ContentBlockType::ReasoningEffort),
            "conversation_start_date" => Ok(ContentBlockType::ConversationStartDate),
            "knowledge_cutoff" => Ok(ContentBlockType::KnowledgeCutoff),
            "builtin_tools" => Ok(ContentBlockType::BuiltinTools),
            "tools" => Ok(ContentBlockType::Tools),
            "text" => Ok(ContentBlockType::Text),
            "image" => Ok(ContentBlockType::Image),
            "video" => Ok(ContentBlockType::Video),
            "audio" => Ok(ContentBlockType::Audio),
            "file" => Ok(ContentBlockType::File),
            "reasoning" => Ok(ContentBlockType::Reasoning),
            "tool_call" => Ok(ContentBlockType::ToolCall),
            "tool_call_candidate" => Ok(ContentBlockType::ToolCallCandidate),
            "tool_call_result" => Ok(ContentBlockType::ToolCallResult),
            "translation" => Ok(ContentBlockType::Translation),
            "custom" => Ok(ContentBlockType::Custom),
            other => Err(format!("Unknown content block type: {other}")),
        }
    }
}

impl Display for ContentBlockType {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            ContentBlockType::Identity => "identity",
            ContentBlockType::ReasoningEffort => "reasoning_effort",
            ContentBlockType::ConversationStartDate => "conversation_start_date",
            ContentBlockType::KnowledgeCutoff => "knowledge_cutoff",
            ContentBlockType::BuiltinTools => "builtin_tools",
            ContentBlockType::Tools => "tools",
            ContentBlockType::Text => "text",
            ContentBlockType::Image => "image",
            ContentBlockType::Video => "video",
            ContentBlockType::Audio => "audio",
            ContentBlockType::File => "file",
            ContentBlockType::Reasoning => "reasoning",
            ContentBlockType::ToolCall => "tool_call",
            ContentBlockType::ToolCallCandidate => "tool_call_candidate",
            ContentBlockType::ToolCallResult => "tool_call_result",
            ContentBlockType::Translation => "translation",
            ContentBlockType::Custom => "custom",
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for ContentBlockType {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for ContentBlockType {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        ContentBlockType::from_str(&name).map_err(de::Error::custom)
    }
}

#[bindings::export(Enum, name = "ContentBlock")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
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
        input: TranslationInput,
        source_language_code: String,
        target_language_code: String,
    },
    Custom {
        value: Value,
    },
}

impl ContentBlock {
    pub fn get_type(&self) -> ContentBlockType {
        match self {
            ContentBlock::Identity {
                ..
            } => ContentBlockType::Identity,
            ContentBlock::ReasoningEffort {
                ..
            } => ContentBlockType::ReasoningEffort,
            ContentBlock::ConversationStartDate {
                ..
            } => ContentBlockType::ConversationStartDate,
            ContentBlock::KnowledgeCutoff {
                ..
            } => ContentBlockType::KnowledgeCutoff,
            ContentBlock::BuiltinTools {
                ..
            } => ContentBlockType::BuiltinTools,
            ContentBlock::Tools {
                ..
            } => ContentBlockType::Tools,
            ContentBlock::Text {
                ..
            } => ContentBlockType::Text,
            ContentBlock::Image {
                ..
            } => ContentBlockType::Image,
            ContentBlock::Video {
                ..
            } => ContentBlockType::Video,
            ContentBlock::Audio {
                ..
            } => ContentBlockType::Audio,
            ContentBlock::File {
                ..
            } => ContentBlockType::File,
            ContentBlock::Reasoning {
                ..
            } => ContentBlockType::Reasoning,
            ContentBlock::ToolCall {
                ..
            } => ContentBlockType::ToolCall,
            ContentBlock::ToolCallCandidate {
                ..
            } => ContentBlockType::ToolCallCandidate,
            ContentBlock::ToolCallResult {
                ..
            } => ContentBlockType::ToolCallResult,
            ContentBlock::Translation {
                ..
            } => ContentBlockType::Translation,
            ContentBlock::Custom {
                ..
            } => ContentBlockType::Custom,
        }
    }
}
