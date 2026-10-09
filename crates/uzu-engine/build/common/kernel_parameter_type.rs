use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum KernelParameterType {
    Type(Box<[Box<str>]>),
    Value(Box<str>),
}

impl KernelParameterType {
    /// A type parameter by the `DataType` names of its variants, compared as a set.
    pub fn types(data_types: impl IntoIterator<Item = Box<str>>) -> Self {
        Self::Type(data_types.into_iter().sorted().dedup().collect())
    }
}
