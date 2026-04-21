use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[bindings::export(Enum, name = "ChatReasoningEffort")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReasoningEffort {
    Disabled,
    Default,
    Low,
    Medium,
    High,
}

impl FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "disabled" => Ok(ReasoningEffort::Disabled),
            "default" => Ok(ReasoningEffort::Default),
            "low" => Ok(ReasoningEffort::Low),
            "medium" => Ok(ReasoningEffort::Medium),
            "high" => Ok(ReasoningEffort::High),
            other => Err(format!("Unknown reasoning effort: {other}")),
        }
    }
}

impl Display for ReasoningEffort {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            ReasoningEffort::Disabled => "disabled",
            ReasoningEffort::Default => "default",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for ReasoningEffort {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for ReasoningEffort {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(ReasoningEffort::from_str(&name).map_err(de::Error::custom)?)
    }
}
