use std::borrow::Borrow;

use derive_more::{AsRef, Deref, Display, From};
use serde::{Deserialize, Serialize};

macro_rules! borrow_str {
    ($name:ident) => {
        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }
    };
}

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display,
)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct KernelName(String);
borrow_str!(KernelName);

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display,
)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct ArgumentName(String);
borrow_str!(ArgumentName);

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display,
)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct SpecializeConstantName(String);
borrow_str!(SpecializeConstantName);

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
