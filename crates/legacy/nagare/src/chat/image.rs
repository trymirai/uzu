use std::io;

use shoji::{
    traits::backend::chat_token::StreamImage,
    types::session::chat::{ChatContentBlock, ChatMessage},
};

use crate::chat::ChatSessionError;

pub(crate) type ImageHash = [u8; 32];

const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

pub(crate) fn get_image_hashes(messages: &[ChatMessage]) -> Result<Vec<ImageHash>, ChatSessionError> {
    let urls = get_image_urls(messages);
    let mut hashes = Vec::<ImageHash>::new();
    for url in urls {
        let hash = get_image_hash(url).map_err(ChatSessionError::backend)?;
        hashes.push(hash);
    }
    Ok(hashes)
}

pub(crate) fn get_image_hash(url: &str) -> Result<ImageHash, io::Error> {
    // A remote image is identified by its URL, so the images of a cached prompt are not downloaded again on every reply
    if is_remote(url) {
        return Ok(*blake3::hash(url.as_bytes()).as_bytes());
    }
    let bytes = std::fs::read(url)?;
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

pub(crate) async fn load_images(images: &mut [StreamImage]) -> Result<(), ChatSessionError> {
    for image in images {
        image.bytes = if is_remote(&image.url) {
            download_image(&image.url).await.map_err(ChatSessionError::backend)?
        } else {
            std::fs::read(&image.url).map_err(|error| ChatSessionError::backend(format!("{}: {error}", image.url)))?
        };
    }
    Ok(())
}

async fn download_image(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let client = reqwest::Client::builder().user_agent(USER_AGENT).build()?;
    let response = client.get(url).send().await?.error_for_status()?;
    Ok(response.bytes().await?.to_vec())
}

fn is_remote(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}
