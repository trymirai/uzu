use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::{
    traits::backend::{Error, Instance as InstanceTrait, chat::StreamConfig},
    types::{
        encoding::{Message, ToolCall},
        session::chat::{FinishReason, Stats},
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ToolCallState {
    Candidate(String),
    Finished(ToolCall),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Output {
    pub reasoning: Option<String>,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCallState>,
    pub finish_reason: Option<FinishReason>,
    pub stats: Stats,
}

pub type Config = ();
pub type StreamInput = Vec<Message>;
pub type StreamOutput = Output;

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance:
    InstanceTrait<StreamConfig = StreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}

impl<T> Instance for T where
    T: InstanceTrait<StreamConfig = StreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}
