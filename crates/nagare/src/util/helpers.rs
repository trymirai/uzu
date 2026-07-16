use std::{io, pin::Pin};

use futures::{Stream, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Context, Encoding, TokenizerLocation},
};
use shoji::{
    traits::backend::chat_message::Output,
    types::model::{EncodingConfig, Model},
};

use crate::chat::ChatSessionError;

pub fn error_stream<'a>(
    err: ChatSessionError
) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
    Box::pin(stream::once(async move { Err(err) }))
}

pub fn build_encoding(
    reference: String,
    model: &Model,
) -> Result<Encoding, io::Error> {
    let config: Option<EncodingConfig> = if model.encodings.is_empty() {
        None
    } else if model.encodings.len() == 1 {
        model.encodings.first().cloned()
    } else {
        let hanashi_config = model.encodings.iter().find(|config| matches!(config, EncodingConfig::Hanashi { .. }));
        if hanashi_config.is_some() {
            hanashi_config.cloned()
        } else {
            model.encodings.first().cloned()
        }
    };
    let Some(config) = config else {
        return Err(io::Error::other("can not get encoding config"));
    };

    let encoding_context = Context {
        tokenizer_location: TokenizerLocation::Directory {
            path: reference,
            name: Some("tokenizer.json".to_string()),
        },
    };
    Encoding::new(config, encoding_context).map_err(io::Error::other)
}
