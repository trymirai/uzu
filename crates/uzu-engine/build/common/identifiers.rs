use std::borrow::Borrow;

use derive_more::{AsRef, Deref, Display, From};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct KernelName(String);

impl Borrow<str> for KernelName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct ArgumentName(String);

impl Borrow<str> for ArgumentName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Deref)]
#[serde(transparent)]
#[as_ref([String])]
#[deref(forward)]
pub struct KernelPath(Box<[String]>);

impl FromIterator<String> for KernelPath {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}
