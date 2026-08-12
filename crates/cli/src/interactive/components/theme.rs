use iocraft::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::interactive::helpers::ColorRgb;

#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub accent_color: Color,
    pub subtitle_color: Color,
    pub symbol_heart: String,
}

impl Theme {
    pub fn all() -> Vec<Self> {
        vec![Self::blue(), Self::green(), Self::yellow(), Self::red(), Self::purple()]
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::all().into_iter().find(|theme| theme.name == name)
    }

    pub fn blue() -> Self {
        Self {
            name: "blue".to_string(),
            accent_color: Color::Blue,
            subtitle_color: Color::DarkGrey,
            symbol_heart: "💙".to_string(),
        }
    }

    pub fn green() -> Self {
        Self {
            name: "green".to_string(),
            accent_color: Color::Green,
            subtitle_color: Color::DarkGrey,
            symbol_heart: "💚".to_string(),
        }
    }

    pub fn yellow() -> Self {
        Self {
            name: "yellow".to_string(),
            accent_color: Color::Yellow,
            subtitle_color: Color::DarkGrey,
            symbol_heart: "💛".to_string(),
        }
    }

    pub fn red() -> Self {
        Self {
            name: "red".to_string(),
            accent_color: Color::Red,
            subtitle_color: Color::DarkGrey,
            symbol_heart: "❤️".to_string(),
        }
    }

    pub fn purple() -> Self {
        Self {
            name: "purple".to_string(),
            accent_color: Color::Magenta,
            subtitle_color: Color::DarkGrey,
            symbol_heart: "💜".to_string(),
        }
    }

    pub fn padding(&self) -> u16 {
        1
    }

    pub fn padding_wide(&self) -> u16 {
        self.padding() * 4
    }

    pub fn overlay_color(&self) -> Color {
        self.subtitle_color.darker(0.5)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::blue()
    }
}

impl Serialize for Theme {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.name)
    }
}

impl<'de> Deserialize<'de> for Theme {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name).ok_or_else(|| serde::de::Error::custom(format!("unknown theme: {name}")))
    }
}
