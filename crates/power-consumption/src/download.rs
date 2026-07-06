use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, bail};
use shoji::types::model::{Model, ModelAccessibility, ModelReference};
use uzu::storage::{Storage, types::DownloadPhase};

const MAX_HEADER_BYTES: u64 = 100_000_000;

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub header_path: PathBuf,
}

pub async fn fetch(
    storage: &Storage,
    model: &Model,
    temp_dir: &Path,
    timeout: Duration,
) -> Result<ModelFiles> {
    storage.refresh(vec![model.clone()]).await.map_err(|error| anyhow!("refresh: {error}"))?;

    let identifier = &model.identifier;
    let already_downloaded =
        matches!(storage.state(identifier).await.map(|state| state.phase), Some(DownloadPhase::Downloaded {}));
    if !already_downloaded {
        storage.download(identifier).await.map_err(|error| anyhow!("download: {error}"))?;
        wait_downloaded(storage, identifier, timeout).await?;
    }

    let model_dir = storage.config.cache_model_path(model).ok_or_else(|| anyhow!("no cache path for model"))?;
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        bail!("config.json missing after download");
    }

    let weights_url = weights_url(model).ok_or_else(|| anyhow!("no model.safetensors url in registry entry"))?;
    let header_path = temp_dir.join(format!("{}-header.safetensors", model.cache_identifier()));
    download_header(&weights_url, &header_path).await?;

    Ok(ModelFiles {
        config_path,
        header_path,
    })
}

async fn wait_downloaded(
    storage: &Storage,
    identifier: &String,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        match storage.state(identifier).await {
            Some(state) => match state.phase {
                DownloadPhase::Downloaded {} => return Ok(()),
                DownloadPhase::Error {
                    ..
                } => bail!("download entered error phase"),
                _ => {},
            },
            None => bail!("download state disappeared"),
        }
        if Instant::now() >= deadline {
            bail!("download timed out");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn weights_url(model: &Model) -> Option<String> {
    if let ModelAccessibility::Local {
        reference,
        ..
    } = &model.accessibility
        && let ModelReference::Mirai {
            files,
            ..
        } = reference
    {
        return files.iter().find(|file| file.name == "model.safetensors").map(|file| file.url.clone());
    }
    None
}

async fn download_header(
    url: &str,
    dest: &Path,
) -> Result<()> {
    let client = reqwest::Client::new();

    let length_response =
        client.get(url).header(reqwest::header::RANGE, "bytes=0-7").send().await?.error_for_status()?;
    if length_response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        bail!("server does not support range requests (status {})", length_response.status());
    }
    let length_bytes = length_response.bytes().await?;
    if length_bytes.len() < 8 {
        bail!("short read for safetensors header length");
    }
    let header_len = u64::from_le_bytes(length_bytes[..8].try_into().unwrap());
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!("implausible safetensors header length: {header_len}");
    }

    let end = 8 + header_len - 1;
    let header_response =
        client.get(url).header(reqwest::header::RANGE, format!("bytes=0-{end}")).send().await?.error_for_status()?;
    if header_response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        bail!("server does not support range requests for header body");
    }
    let header_bytes = header_response.bytes().await?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dest, &header_bytes)?;
    Ok(())
}
