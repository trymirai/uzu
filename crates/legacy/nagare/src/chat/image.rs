use std::io;

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
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
    if is_remote(url) || is_data(url) {
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
        } else if is_data(&image.url) {
            let (bytes, header_length) = decode_data_url(&image.url).map_err(ChatSessionError::backend)?;
            image.url.truncate(header_length);
            bytes
        } else {
            std::fs::read(&image.url).map_err(|error| ChatSessionError::backend(format!("{}: {error}", image.url)))?
        };
    }
    Ok(())
}

fn decode_data_url(url: &str) -> Result<(Vec<u8>, usize), String> {
    let (header, payload) = url.split_once(',').ok_or("Image data URL has no payload")?;
    if !header.ends_with(";base64") {
        return Err("Image data URL must be base64 encoded".to_string());
    }

    let bytes = BASE64_STANDARD.decode(payload).map_err(|error| format!("Invalid image data URL: {error}"))?;
    Ok((bytes, header.len()))
}

async fn download_image(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let client = reqwest::Client::builder().user_agent(USER_AGENT).build()?;
    let response = client.get(url).send().await?.error_for_status()?;
    Ok(response.bytes().await?.to_vec())
}

fn is_data(url: &str) -> bool {
    url.starts_with("data:")
}

fn is_remote(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn data_url_image_loads_without_a_file() {
        let url = "data:image/png;base64,aGVsbG8=";
        assert_eq!(get_image_hash(url).unwrap(), *blake3::hash(url.as_bytes()).as_bytes());

        let mut images = [StreamImage {
            url: url.to_owned(),
            bytes: Vec::new(),
        }];
        load_images(&mut images).await.unwrap();
        assert_eq!(images[0].bytes, b"hello");
        assert_eq!(images[0].url, "data:image/png;base64");

        for url in ["data:image/png;base64", "data:image/png,hello", "data:image/png;base64,***"] {
            let mut images = [StreamImage {
                url: url.to_owned(),
                bytes: Vec::new(),
            }];
            assert!(load_images(&mut images).await.is_err(), "url: {url}");
        }
    }
}
