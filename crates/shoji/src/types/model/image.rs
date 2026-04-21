use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ImageFormat")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Webp,
    Svg,
}

#[bindings::export(Enum, name = "ImageTheme")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTheme {
    Light,
    Dark,
}

#[bindings::export(Struct, name = "Image")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Image {
    pub id: String,
    pub url: String,
    pub theme: ImageTheme,
    pub format: ImageFormat,
    pub description: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}
