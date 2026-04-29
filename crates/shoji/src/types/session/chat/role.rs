use std::{
    convert::Infallible,
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChatRole {
    User {},
    Assistant {},
    System {},
    Developer {},
    Tool {},
    Custom {
        name: String,
    },
}

impl FromStr for ChatRole {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "user" => ChatRole::User {},
            "assistant" => ChatRole::Assistant {},
            "system" => ChatRole::System {},
            "developer" => ChatRole::Developer {},
            "tool" => ChatRole::Tool {},
            other => ChatRole::Custom {
                name: other.to_string(),
            },
        })
    }
}

impl Display for ChatRole {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            ChatRole::User {} => "user",
            ChatRole::Assistant {} => "assistant",
            ChatRole::System {} => "system",
            ChatRole::Developer {} => "developer",
            ChatRole::Tool {} => "tool",
            ChatRole::Custom {
                name,
            } => name,
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for ChatRole {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for ChatRole {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(ChatRole::from_str(&name).map_err(de::Error::custom)?)
    }
}
