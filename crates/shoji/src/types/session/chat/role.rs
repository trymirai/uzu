use std::{
    convert::Infallible,
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[bindings::export(Enum, name = "Role")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Role {
    User {},
    Assistant {},
    System {},
    Developer {},
    Tool {},
    Custom {
        name: String,
    },
}

impl FromStr for Role {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "user" => Role::User {},
            "assistant" => Role::Assistant {},
            "system" => Role::System {},
            "developer" => Role::Developer {},
            "tool" => Role::Tool {},
            other => Role::Custom {
                name: other.to_string(),
            },
        })
    }
}

impl Display for Role {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            Role::User {} => "user",
            Role::Assistant {} => "assistant",
            Role::System {} => "system",
            Role::Developer {} => "developer",
            Role::Tool {} => "tool",
            Role::Custom {
                name,
            } => name,
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for Role {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for Role {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Role::from_str(&name).map_err(de::Error::custom)?)
    }
}
