use iocraft::prelude::*;

use crate::{
    cli::helpers::ColorRgb,
    settings::{SettingKind, Settings, SettingsError},
};

const SETTINGS_THEME: &str = "theme";

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
}

impl Default for Theme {
    fn default() -> Self {
        Self::blue()
    }
}

impl Theme {
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

impl Theme {
    pub fn load(settings: &Settings) -> Result<Option<Self>, SettingsError> {
        let Some(name) = settings.load(SettingKind::Config, SETTINGS_THEME.to_string())? else {
            return Ok(None);
        };
        Ok(Self::all().into_iter().find(|theme| theme.name == name))
    }

    pub fn save(
        &self,
        settings: &Settings,
    ) -> Result<(), SettingsError> {
        settings.save(SettingKind::Config, SETTINGS_THEME.to_string(), Some(self.name.clone()))
    }
}
