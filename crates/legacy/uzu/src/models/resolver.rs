use std::{collections::HashSet, io::ErrorKind, path::PathBuf, sync::Arc, time::Duration};

use download_manager::FileCheck;
use futures_util::{StreamExt, TryStreamExt, stream};
use reqwest::{Client, Url};
use serde_json::{from_slice, to_vec};
use shoji::types::{
    basic::{File, Repository},
    model::{Model, ModelAccessibility, ModelSource},
};
use tokio::fs::{
    create_dir_all as tokio_create_dir_all, read as tokio_read, rename as tokio_rename, write as tokio_write,
};
use uuid::Uuid;

use super::{ResolvedFile, ResolvedModel, ResolvedModels};
use crate::{
    api::{HuggingFaceFileResponse, HuggingFaceModelResponse},
    registry::RegistryError,
};

const HUGGING_FACE_URL: &str = "https://huggingface.co";
const MAX_CONCURRENT_REQUESTS: usize = 8;
const SIDECAR_EXTENSIONS: [&str; 4] = ["part", "resume_data", "lock", "integrity"];

#[derive(Clone)]
pub struct ModelsResolver {
    client: Client,
    endpoint: Url,
    api_key: Option<Arc<str>>,
    cache_path: PathBuf,
}

impl ModelsResolver {
    pub fn new(
        api_key: Option<Arc<str>>,
        cache_path: PathBuf,
    ) -> Result<Self, RegistryError> {
        let client =
            Client::builder().https_only(true).timeout(Duration::from_secs(30)).build().map_err(resolution_error)?;
        let endpoint = Url::parse(HUGGING_FACE_URL).map_err(resolution_error)?;
        Ok(Self {
            client,
            endpoint,
            api_key,
            cache_path,
        })
    }

    pub async fn load_cache(&self) -> Result<Option<ResolvedModels>, RegistryError> {
        let contents = match tokio_read(&self.cache_path).await {
            Ok(contents) => contents,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(resolution_error(error)),
        };
        let models = from_slice::<ResolvedModels>(&contents).map_err(resolution_error)?;
        Ok(models.validate_cache())
    }

    pub async fn save_cache(
        &self,
        models: &ResolvedModels,
    ) -> Result<(), RegistryError> {
        let parent = self.cache_path.parent().ok_or_else(|| resolution_error("invalid resolved models cache path"))?;
        tokio_create_dir_all(parent).await.map_err(resolution_error)?;
        let temporary = self.cache_path.with_added_extension(format!("{}.tmp", Uuid::new_v4()));
        tokio_write(&temporary, to_vec(models).map_err(resolution_error)?).await.map_err(resolution_error)?;
        tokio_rename(temporary, &self.cache_path).await.map_err(resolution_error)
    }

    pub async fn resolve(
        &self,
        models: Vec<Model>,
        previous: &ResolvedModels,
    ) -> Result<ResolvedModels, RegistryError> {
        let futures = models.into_iter().map(|model| self.resolve_model(model, previous));
        let resolved = stream::iter(futures).buffered(MAX_CONCURRENT_REQUESTS).try_collect::<Vec<_>>().await?;
        Ok(ResolvedModels::new(resolved))
    }

    async fn resolve_model(
        &self,
        model: Model,
        previous: &ResolvedModels,
    ) -> Result<ResolvedModel, RegistryError> {
        let ModelAccessibility::OnDevice {
            source:
                ModelSource::Managed {
                    repository,
                    files,
                    ..
                },
        } = &model.accessibility
        else {
            return Ok(ResolvedModel::passthrough(model));
        };
        let Some((repository, revision)) = repository
            .as_ref()
            .and_then(|repository| repository.commit_hash.as_deref().map(|revision| (repository, revision)))
        else {
            let files = files
                .iter()
                .cloned()
                .map(|file| {
                    let check = file.crc32c().map(FileCheck::CRC).ok_or_else(|| {
                        resolution_error(format!("missing CRC32C for {}/{}", model.identifier, file.name))
                    })?;
                    Ok(ResolvedFile {
                        file,
                        check,
                        requires_authentication: false,
                    })
                })
                .collect::<Result<Vec<_>, RegistryError>>()?;
            return Ok(ResolvedModel::downloadable(model, files));
        };
        if let Some(files) = previous.reusable_hugging_face_files(&model) {
            return Ok(ResolvedModel::downloadable(model, files));
        }
        let files = self.resolve_hugging_face(repository, revision).await?;
        Ok(ResolvedModel::downloadable(model, files))
    }

    async fn resolve_hugging_face(
        &self,
        repository: &Repository,
        revision: &str,
    ) -> Result<Vec<ResolvedFile>, RegistryError> {
        if !is_lower_hex(revision, 40) {
            return Err(resolution_error("invalid commit"));
        }
        let mut metadata_url = self.endpoint.clone();
        metadata_url
            .path_segments_mut()
            .map_err(|_| resolution_error("invalid Hugging Face endpoint"))?
            .clear()
            .extend(["api", "models"])
            .extend(repository.identifier.split('/'))
            .extend(["revision", revision]);
        metadata_url.query_pairs_mut().append_pair("blobs", "true");
        let mut request = self.client.get(metadata_url);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key.as_ref());
        }
        let response = request.send().await.map_err(resolution_error)?;
        if !response.status().is_success() {
            return Err(resolution_error(format!("Hugging Face returned HTTP {}", response.status().as_u16())));
        }
        let response: HuggingFaceModelResponse = response.json().await.map_err(resolution_error)?;
        if response.sha != revision {
            return Err(resolution_error(format!(
                "Hugging Face returned revision {} instead of {revision}",
                response.sha
            )));
        }
        let requires_authentication = response.requires_authentication();
        if requires_authentication && self.api_key.is_none() {
            return Err(resolution_error("Hugging Face authentication is required"));
        }
        self.resolve_files(repository, revision, response.siblings, requires_authentication)
    }

    fn resolve_files(
        &self,
        repository: &Repository,
        revision: &str,
        siblings: Vec<HuggingFaceFileResponse>,
        requires_authentication: bool,
    ) -> Result<Vec<ResolvedFile>, RegistryError> {
        let mut resolve_base = self.endpoint.clone();
        resolve_base
            .path_segments_mut()
            .map_err(|_| resolution_error("invalid Hugging Face endpoint"))?
            .clear()
            .extend(repository.identifier.split('/'))
            .extend(["resolve", revision]);
        let mut names = HashSet::with_capacity(siblings.len());
        let mut files = Vec::with_capacity(siblings.len());
        for sibling in siblings
            .iter()
            .filter(|sibling| repository.paths.as_ref().is_none_or(|paths| paths.contains(&sibling.rfilename)))
        {
            validate_path(&sibling.rfilename)?;
            let normalized_name = PathBuf::from(sibling.rfilename.to_lowercase());
            if !names.insert(normalized_name) {
                return Err(invalid_file(&sibling.rfilename));
            }
            let (size, check) = file_size_and_check(sibling)?;
            let mut url = resolve_base.clone();
            url.path_segments_mut()
                .map_err(|_| resolution_error("invalid Hugging Face endpoint"))?
                .extend(sibling.rfilename.split('/'));
            files.push(ResolvedFile {
                file: File {
                    url: url.into(),
                    name: sibling.rfilename.clone(),
                    size: i64::try_from(size).map_err(|_| invalid_file("file size overflow"))?,
                    hashes: Vec::new(),
                },
                check,
                requires_authentication,
            });
        }
        let sidecars = names
            .iter()
            .flat_map(|name| SIDECAR_EXTENSIONS.iter().map(|extension| name.with_added_extension(extension)))
            .collect::<HashSet<_>>();
        let collides = names.iter().any(|name| {
            name.ancestors().skip(1).any(|parent| names.contains(parent))
                || name.ancestors().any(|path| sidecars.contains(path))
        });
        if files.is_empty() || collides {
            return Err(invalid_file("empty or colliding manifest"));
        }
        Ok(files)
    }
}

fn file_size_and_check(file: &HuggingFaceFileResponse) -> Result<(u64, FileCheck), RegistryError> {
    if let Some(lfs) = &file.lfs {
        if let (Some(size), Some(lfs_size)) = (file.size, lfs.size)
            && size != lfs_size
        {
            return Err(resolution_error(format!("Hugging Face size mismatch for {}", file.rfilename)));
        }
        let size = lfs
            .size
            .or(file.size)
            .ok_or_else(|| resolution_error(format!("Hugging Face is missing size for {}", file.rfilename)))?;
        let digest = lfs
            .sha256
            .as_deref()
            .filter(|digest| is_lower_hex(digest, 64))
            .ok_or_else(|| resolution_error(format!("missing or invalid SHA-256 for {}", file.rfilename)))?;
        Ok((size, FileCheck::Sha256(digest.to_string())))
    } else {
        let size = file
            .size
            .ok_or_else(|| resolution_error(format!("Hugging Face is missing size for {}", file.rfilename)))?;
        let digest = file
            .blob_id
            .as_deref()
            .filter(|digest| is_lower_hex(digest, 40))
            .ok_or_else(|| resolution_error(format!("missing or invalid Git blob SHA-1 for {}", file.rfilename)))?;
        Ok((size, FileCheck::GitBlobSha1(digest.to_string())))
    }
}

fn validate_path(path: &str) -> Result<(), RegistryError> {
    if path.contains(['\\', ':']) || path.split('/').any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(invalid_file(path));
    }
    Ok(())
}

fn is_lower_hex(
    value: &str,
    length: usize,
) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn invalid_file(path: &str) -> RegistryError {
    resolution_error(format!("invalid file: {path}"))
}

fn resolution_error(message: impl ToString) -> RegistryError {
    RegistryError::UnableToGetModels {
        message: message.to_string(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/models/resolver_test.rs"]
mod tests;
