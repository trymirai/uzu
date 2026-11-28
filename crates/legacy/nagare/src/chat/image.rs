use std::{io, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
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
        let hash = get_image_hash(url)
            .map_err(|error| ChatSessionError::backend(format!("Reading image {url:?}: {error}")))?;
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
    let mut remaining = MAX_IMAGE_DOWNLOAD_BYTES_PER_TURN;
    for image in images {
        image.bytes = if is_remote(&image.url) {
            let bytes = download_image(&image.url, remaining).await?;
            remaining -= bytes.len();
            bytes
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

async fn download_image(
    url: &str,
    remaining: usize,
) -> Result<Vec<u8>, ChatSessionError> {
    let client = reqwest::Client::builder().user_agent(USER_AGENT).build().map_err(ChatSessionError::backend)?;
    let response = client
        .get(url)
        .timeout(IMAGE_DOWNLOAD_TIMEOUT)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(ChatSessionError::backend)?;
    read_image_response(response, remaining).await
}

async fn read_image_response(
    response: reqwest::Response,
    remaining: usize,
) -> Result<Vec<u8>, ChatSessionError> {
    let over_max_download_size = || {
        ChatSessionError::backend(format!(
            "Image downloads exceed the {MAX_IMAGE_DOWNLOAD_BYTES_PER_TURN}-byte turn limit"
        ))
    };

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

fn is_data(url: &str) -> bool {
    url.starts_with("data:")
}

fn is_remote(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn image_stream_enforces_limit_without_content_length() {
        for (limit, accepted) in [(4, true), (3, false)] {
            let chunks = futures::stream::iter([Ok::<_, io::Error>(b"ab".to_vec()), Ok(b"cd".to_vec())]);
            let response = reqwest::Response::from(http::Response::new(reqwest::Body::wrap_stream(chunks)));
            assert_eq!(response.content_length(), None);
            let result = read_image_response(response, limit).await;
            if accepted {
                assert_eq!(result.unwrap(), b"abcd");
            } else {
                assert!(result.unwrap_err().to_string().contains("turn limit"));
            }
        }
    }

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
