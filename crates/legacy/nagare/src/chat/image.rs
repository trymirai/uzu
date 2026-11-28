use std::{io, time::Duration};

use futures::StreamExt;
use shoji::{
    traits::backend::chat_token::StreamImage,
    types::session::chat::{ChatContentBlock, ChatMessage},
};

use crate::chat::ChatSessionError;

const IMAGE_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_IMAGE_DOWNLOAD_BYTES_PER_TURN: usize = 64 * 1024 * 1024;
const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

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
    let mut remaining = MAX_IMAGE_DOWNLOAD_BYTES_PER_TURN;
    for image in images {
        image.bytes = if is_remote(&image.url) {
            let bytes = download_image(&image.url, remaining).await?;
            remaining -= bytes.len();
            bytes
        } else {
            std::fs::read(&image.url).map_err(|error| ChatSessionError::backend(format!("{}: {error}", image.url)))?
        };
    }
    Ok(())
}

async fn download_image(
    url: &str,
    remaining: usize,
) -> Result<Vec<u8>, ChatSessionError> {
    let over_max_download_size = || {
        ChatSessionError::backend(format!(
            "Image downloads exceed the {MAX_IMAGE_DOWNLOAD_BYTES_PER_TURN}-byte turn limit"
        ))
    };

    let client = reqwest::Client::builder().user_agent(USER_AGENT).build().map_err(ChatSessionError::backend)?;
    let response = client
        .get(url)
        .timeout(IMAGE_DOWNLOAD_TIMEOUT)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(ChatSessionError::backend)?;
    if response.content_length().is_some_and(|length| length > remaining as u64) {
        return Err(over_max_download_size());
    }

    let mut bytes = Vec::new();
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(ChatSessionError::backend)?;
        if chunk.len() > remaining - bytes.len() {
            return Err(over_max_download_size());
        }
        bytes.extend_from_slice(&chunk);
    }

    Ok(bytes)
}

fn is_remote(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}
