use std::{io, pin::Pin};

use futures::{Stream, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Context, Encoding, EncodingConfig, TokenizerLocation},
};
use shoji::{traits::backend::chat_message::Output, types::model::Model};

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
    let encoding_configs = model
        .encodings
        .iter()
        .map(|value| serde_json::from_str::<EncodingConfig>(value.json.as_str()))
        .collect::<Result<Vec<_>, _>>()?;

    let config: Option<EncodingConfig> = if encoding_configs.is_empty() {
        None
    } else if encoding_configs.len() == 1 {
        encoding_configs.first().cloned()
    } else {
        let hanashi_config = encoding_configs.iter().find(|config| matches!(config, EncodingConfig::Hanashi { .. }));
        if hanashi_config.is_some() {
            hanashi_config.cloned()
        } else {
            encoding_configs.first().cloned()
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
