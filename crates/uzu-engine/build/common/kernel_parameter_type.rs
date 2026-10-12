use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum KernelParameterType {
    Type,
    Value(Box<str>),
}
