use std::pin::Pin;

use futures::{Stream, stream};
use shoji::traits::backend::chat_message::Output;

use crate::chat::ChatSessionError;

pub fn error_stream<'a>(
    err: ChatSessionError
) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
    Box::pin(stream::once(async move { Err(err) }))
}
