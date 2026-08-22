use std::{
    fmt,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

/// A non-empty, portable relative file path that cannot escape its destination root.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelativeFilePath(PathBuf);

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum RelativeFilePathError {
    #[error("relative file path cannot be empty")]
    Empty,
    #[error("file path must be a portable safe relative path: {0}")]
    Invalid(PathBuf),
}

impl RelativeFilePath {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, RelativeFilePathError> {
        let path = path.into();
        let Some(encoded_path) = path.to_str() else {
            return Err(RelativeFilePathError::Invalid(path));
        };
        if encoded_path.split('/').any(is_unsafe_portable_component) {
            return if encoded_path.is_empty() {
                Err(RelativeFilePathError::Empty)
            } else {
                Err(RelativeFilePathError::Invalid(path))
            };
        }

        let mut components = path.components();
        let Some(first) = components.next() else {
            return Err(RelativeFilePathError::Empty);
        };

        if !matches!(first, Component::Normal(_))
            || components.any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(RelativeFilePathError::Invalid(path));
        }

        Ok(Self(path))
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

fn is_unsafe_portable_component(component: &str) -> bool {
    component.is_empty()
        || matches!(component, "." | "..")
        || component.ends_with([' ', '.'])
        || component.chars().any(|character| {
            character <= '\u{1f}' || matches!(character, '<' | '>' | ':' | '"' | '\\' | '|' | '?' | '*')
        })
        || is_windows_device_name(component)
}

fn is_windows_device_name(component: &str) -> bool {
    let stem = component.split('.').next().unwrap_or(component).to_ascii_uppercase();
    matches!(
        stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "CLOCK$"
            | "CONIN$"
            | "CONOUT$"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "COM¹"
            | "COM²"
            | "COM³"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
            | "LPT¹"
            | "LPT²"
            | "LPT³"
    )
}

impl AsRef<Path> for RelativeFilePath {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

impl fmt::Display for RelativeFilePath {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.0.display().fmt(formatter)
    }
}

impl TryFrom<PathBuf> for RelativeFilePath {
    type Error = RelativeFilePathError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl TryFrom<&Path> for RelativeFilePath {
    type Error = RelativeFilePathError;

    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl TryFrom<&str> for RelativeFilePath {
    type Error = RelativeFilePathError;

    fn try_from(path: &str) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl Serialize for RelativeFilePath {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RelativeFilePath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let path = PathBuf::deserialize(deserializer)?;
        Self::new(path).map_err(D::Error::custom)
    }
}

#[cfg(test)]
#[path = "../tests/unit/relative_file_path_test.rs"]
mod tests;
