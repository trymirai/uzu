use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use backend_uzu::summarize_header;
use shoji::types::model::{Model, ModelAccessibility, ModelReference};
use tokio::{runtime::Handle as TokioHandle, time::timeout};
use tokio_stream::StreamExt;
use uzu::{
    device::Device as StorageDevice,
    storage::{Config as StorageConfig, DownloadContents, Storage, types::DownloadPhase},
};

use super::artifacts::HEADER_FILE;

const MAX_HEADER_BYTES: u64 = 100_000_000;
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(180);

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub header_path: PathBuf,
}

pub struct Downloader {
    storage: Storage,
    client: reqwest::Client,
    timeout: Duration,
}

impl Downloader {
    pub async fn new(
        tokio: TokioHandle,
        storage_base: Option<PathBuf>,
    ) -> Result<Self> {
        let storage_device = StorageDevice::new().map_err(|error| anyhow!("device: {error}"))?;
        let storage_config = StorageConfig::new(storage_device, storage_base, super::artifacts::CACHE_NAME.to_string())
            .with_download_contents(DownloadContents::CONFIG);
        let storage = Storage::new(tokio, storage_config).await.map_err(|error| anyhow!("storage: {error}"))?;

        Ok(Self {
            storage,
            client: reqwest::Client::new(),
            timeout: DOWNLOAD_TIMEOUT,
        })
    }

    pub async fn fetch(
        &self,
        model: &Model,
    ) -> Result<ModelFiles> {
        self.storage.refresh(vec![model.clone()]).await.context("refresh")?;

        let identifier = &model.identifier;
        let already_downloaded =
            matches!(self.storage.state(identifier).await.map(|state| state.phase), Some(DownloadPhase::Downloaded {}));
        if !already_downloaded {
            let mut events = self.storage.subscribe();
            self.storage.download(identifier).await.context("download")?;
            self.wait_downloaded(identifier, &mut events).await?;
        }

        let model_dir = self.storage.config.cache_model_path(model).context("no cache path for model")?;
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            bail!("config.json missing after download");
        }

        let header_path = model_dir.join(HEADER_FILE);
        if !header_path.is_file() {
            let weights_url = weights_url(model).context("no model.safetensors url in registry entry")?;
            self.download_header(&weights_url, &header_path).await?;
        }

        summarize_header(&header_path).with_context(|| format!("summarize header {}", header_path.display()))?;

        Ok(ModelFiles {
            config_path,
            header_path,
        })
    }

    async fn wait_downloaded(
        &self,
        identifier: &String,
        events: &mut uzu::storage::types::StorageDownloadEventStream,
    ) -> Result<()> {
        if matches!(self.storage.state(identifier).await.map(|state| state.phase), Some(DownloadPhase::Downloaded {})) {
            return Ok(());
        }

        timeout(self.timeout, async {
            while let Some(event) = events.next().await {
                let Ok((id, state)) = event else {
                    if let Some(state) = self.storage.state(identifier).await {
                        match state.phase {
                            DownloadPhase::Downloaded {} => return Ok(()),
                            DownloadPhase::Error {
                                message,
                            } => bail!("download entered error phase: {message}"),
                            _ => {},
                        }
                    }
                    continue;
                };
                if &id != identifier {
                    continue;
                }
                match state.phase {
                    DownloadPhase::Downloaded {} => return Ok(()),
                    DownloadPhase::Error {
                        message,
                    } => bail!("download entered error phase: {message}"),
                    _ => {},
                }
            }
            bail!("download event stream ended")
        })
        .await
        .context("download timed out")?
    }

    async fn download_header(
        &self,
        url: &str,
        dest: &Path,
    ) -> Result<()> {
        let length_response = self
            .client
            .get(url)
            .header(reqwest::header::RANGE, "bytes=0-7")
            .send()
            .await
            .context("header length request")?
            .error_for_status()
            .context("header length status")?;
        if length_response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            bail!("server does not support range requests (status {})", length_response.status());
        }
        let length_bytes = length_response.bytes().await.context("header length body")?;
        if length_bytes.len() < 8 {
            bail!("short read for safetensors header length");
        }
        let header_len =
            u64::from_le_bytes(length_bytes[..8].try_into().map_err(|_| anyhow!("safetensors header length bytes"))?);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            bail!("implausible safetensors header length: {header_len}");
        }

        let end = 8 + header_len - 1;
        let header_response = self
            .client
            .get(url)
            .header(reqwest::header::RANGE, format!("bytes=0-{end}"))
            .send()
            .await
            .context("header body request")?
            .error_for_status()
            .context("header body status")?;
        if header_response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            bail!("server does not support range requests for header body");
        }
        let header_bytes = header_response.bytes().await.context("header body")?;

        let parent = dest.parent().context("header destination has no parent")?;
        tokio::fs::create_dir_all(parent).await.with_context(|| format!("create {}", parent.display()))?;
        let temp_path = dest.with_extension("safetensors.part");
        tokio::fs::write(&temp_path, &header_bytes).await.with_context(|| format!("write {}", temp_path.display()))?;
        tokio::fs::rename(&temp_path, dest)
            .await
            .with_context(|| format!("rename {} -> {}", temp_path.display(), dest.display()))?;
        Ok(())
    }
}

pub fn weights_size(model: &Model) -> Option<i64> {
    mirai_file(model, "model.safetensors").map(|file| file.size)
}

fn weights_url(model: &Model) -> Option<String> {
    mirai_file(model, "model.safetensors").map(|file| file.url.clone())
}

fn mirai_file<'a>(
    model: &'a Model,
    name: &str,
) -> Option<&'a shoji::types::basic::File> {
    let ModelAccessibility::Local {
        reference: ModelReference::Mirai {
            files,
            ..
        },
        ..
    } = &model.accessibility
    else {
        return None;
    };
    files.iter().find(|file| file.name == name)
}
