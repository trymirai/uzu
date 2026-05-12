use std::borrow::Borrow;

use derive_more::{AsRef, Deref, Display, From};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, From, AsRef, Deref, Display)]
#[serde(transparent)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct SpecializeConstantName(String);

impl Borrow<str> for SpecializeConstantName {
    fn borrow(&self) -> &str {
        &self.0
    }
}
