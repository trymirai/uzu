pub mod chat;
pub mod chat_message;
pub mod chat_token;
pub mod classification;
pub mod text_to_speech;

use std::pin::Pin;

use futures::Stream;
use tokio_util::sync::CancellationToken;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub trait Backend: Send + Sync {
    fn identifier(&self) -> String;
    fn version(&self) -> String;

    fn as_chat_via_token_capable(&self) -> Option<&dyn chat_token::Backend> {
        None
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message::Backend> {
        None
    }

    fn as_classification_capable(&self) -> Option<&dyn classification::Backend> {
        None
    }

    fn as_text_to_speech_capable(&self) -> Option<&dyn text_to_speech::Backend> {
        None
    }
}

pub trait Instance: Send + Sync {
    type StreamConfig;
    type StreamInput;
    type StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>>;

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        config: Self::StreamConfig,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>>;
}

pub trait State: Send + Sync + 'static {
    fn clone_boxed(&self) -> Box<dyn State>;
}
