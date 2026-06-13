use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningEffort {
    Disabled,
    Default,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn from_openai(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(ReasoningEffort::Disabled),
            "minimal" => Ok(ReasoningEffort::Low),
            other => Self::from_str(other),
        }
    }
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
        ReasoningEffort::from_str(&name).map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_openai_maps_openai_specific_levels() {
        assert_eq!(ReasoningEffort::from_openai("none"), Ok(ReasoningEffort::Disabled));
        assert_eq!(ReasoningEffort::from_openai("minimal"), Ok(ReasoningEffort::Low));
    }

    #[test]
    fn from_openai_falls_back_to_from_str() {
        assert_eq!(ReasoningEffort::from_openai("low"), Ok(ReasoningEffort::Low));
        assert_eq!(ReasoningEffort::from_openai("medium"), Ok(ReasoningEffort::Medium));
        assert_eq!(ReasoningEffort::from_openai("high"), Ok(ReasoningEffort::High));
    }

    #[test]
    fn from_openai_rejects_unknown_values() {
        assert!(ReasoningEffort::from_openai("turbo").is_err());
    }

    #[test]
    fn from_openai_inverts_backend_remote_mapping() {
        // Mirrors crates/backend-remote/src/openai/bridging/reasoning_effort.rs:
        // Disabled <-> "none", Low <-> "minimal"/"low".
        assert_eq!(ReasoningEffort::from_openai("none"), Ok(ReasoningEffort::Disabled));
        assert_eq!(ReasoningEffort::from_openai("minimal"), Ok(ReasoningEffort::Low));
    }
}
