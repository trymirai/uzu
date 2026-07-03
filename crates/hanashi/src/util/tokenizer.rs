use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::chat::TokenizerLocation;

const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";

pub fn load_tokenizer(location: &TokenizerLocation) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer_path = match location {
        TokenizerLocation::File {
            path,
        } => PathBuf::from(path),
        TokenizerLocation::Directory {
            path,
            name,
        } => match name {
            Some(name) => PathBuf::from(path).join(name),
            None => PathBuf::from(path).join(DEFAULT_TOKENIZER_FILENAME),
        },
    };
    Tokenizer::from_file(&tokenizer_path)
}
