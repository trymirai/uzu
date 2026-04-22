use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChatReasoningEffort {
    Disabled,
    Default,
    Low,
    Medium,
    High,
}

impl FromStr for ChatReasoningEffort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "disabled" => Ok(ChatReasoningEffort::Disabled),
            "default" => Ok(ChatReasoningEffort::Default),
            "low" => Ok(ChatReasoningEffort::Low),
            "medium" => Ok(ChatReasoningEffort::Medium),
            "high" => Ok(ChatReasoningEffort::High),
            other => Err(format!("Unknown reasoning effort: {other}")),
        }
    }
}

impl Display for ChatReasoningEffort {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            ChatReasoningEffort::Disabled => "disabled",
            ChatReasoningEffort::Default => "default",
            ChatReasoningEffort::Low => "low",
            ChatReasoningEffort::Medium => "medium",
            ChatReasoningEffort::High => "high",
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for ChatReasoningEffort {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for ChatReasoningEffort {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(ChatReasoningEffort::from_str(&name).map_err(de::Error::custom)?)
    }
}
