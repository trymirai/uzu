use std::{borrow::Borrow, fmt, ops::Deref};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelName(String);

impl KernelName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl From<String> for KernelName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for KernelName {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl Deref for KernelName {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for KernelName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelName {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArgumentName(String);

impl ArgumentName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ArgumentName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for ArgumentName {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl Deref for ArgumentName {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ArgumentName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ArgumentName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ArgumentName {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelPath(Box<[String]>);

impl Deref for KernelPath {
    type Target = [String];

    fn deref(&self) -> &[String] {
        &self.0
    }
}

impl AsRef<[String]> for KernelPath {
    fn as_ref(&self) -> &[String] {
        &self.0
    }
}

impl FromIterator<String> for KernelPath {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}
