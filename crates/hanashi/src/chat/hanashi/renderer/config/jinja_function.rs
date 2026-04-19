use std::{
    fmt::{self, Display},
    str::FromStr,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JinjaFunction {
    StrftimeNow,
}

impl FromStr for JinjaFunction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "strftime_now" => Ok(JinjaFunction::StrftimeNow),
            other => Err(format!("Unknown jinja function: {other}")),
        }
    }
}

impl Display for JinjaFunction {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            JinjaFunction::StrftimeNow => "strftime_now",
        };
        write!(formatter, "{name}")
    }
}

impl Serialize for JinjaFunction {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'d> Deserialize<'d> for JinjaFunction {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(JinjaFunction::from_str(&name).map_err(de::Error::custom)?)
    }
}
