use std::{io, pin::Pin};

use futures::{Stream, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Config, Context as HanashiContext, Encoding, TokenizerLocation, hanashi::Config as HanashiConfig},
};
use shoji::{
    traits::backend::chat_message::Output,
    types::model::{Model, ModelFamily},
};

use crate::chat::ChatSessionError;

pub fn error_stream<'a>(
    err: ChatSessionError
) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        return Err(err);
    }))
}

pub fn get_encoding(
    reference: String,
    model: &Model,
) -> Result<Encoding, io::Error> {
    let family: &ModelFamily =
        model.family.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing model family"))?;
    let model_name = family.metadata.name.to_lowercase();
    // TODO agolokoz: check different models
    let hanashi_config: HanashiConfig = serde_json::from_value(serde_json::json!({ "name": model_name }))
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let encoding_config = Config::Hanashi(hanashi_config);
    let encoding_context = HanashiContext {
        tokenizer_location: TokenizerLocation::Directory {
            path: reference,
            name: Some("tokenizer.json".to_string()),
        },
    };
    Encoding::new(encoding_config, encoding_context).map_err(|err| io::Error::new(io::ErrorKind::Other, err))
}
