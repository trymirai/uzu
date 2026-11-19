use std::io;

use shoji::types::session::chat::{ChatContentBlock, ChatMessage};

use crate::chat::ChatSessionError;

pub(crate) type ImageHash = [u8; 32];

pub(crate) fn get_image_hashes(messages: &[ChatMessage]) -> Result<Vec<ImageHash>, ChatSessionError> {
    let urls = get_image_urls(messages);
    let mut hashes = Vec::<ImageHash>::new();
    for url in urls {
        let hash = get_image_hash(url).map_err(ChatSessionError::backend)?;
        hashes.push(hash);
    }
    Ok(hashes)
}

pub(crate) fn get_image_hash(file_path: &str) -> Result<ImageHash, io::Error> {
    let bytes = std::fs::read(file_path)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

pub(crate) fn get_image_urls(messages: &[ChatMessage]) -> impl Iterator<Item = &str> {
    messages.iter().flat_map(|message| &message.content).filter_map(|block| match block {
        ChatContentBlock::Image {
            url,
        } => Some(url.as_str()),
        _ => None,
    })
}
