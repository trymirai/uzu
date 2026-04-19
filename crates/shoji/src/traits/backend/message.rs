use serde::{Deserialize, Serialize};

use crate::traits::backend::Backend as BackendTrait;
pub use crate::types::{Message, Value};

pub type Input = Vec<Message>;

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Output {
    Content(String),
    Reasoning(String),
    ToolCall(Vec<Value>),
    Done(),
}

pub trait Backend: BackendTrait<StreamInput = Input, StreamOutput = Output> {}

impl<T> Backend for T where T: BackendTrait<StreamInput = Input, StreamOutput = Output> {}
