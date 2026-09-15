use std::time::Duration;

use bon::bon;
use download_manager::BearerToken;
use reqwest::{Client, Url, header::AUTHORIZATION};
use shoji::types::basic::{File, Hash, HashMethod, Repository};

use super::{api::HUGGING_FACE_URL, hugging_face_model::HuggingFaceModel};
use crate::registry::RegistryError;

pub struct HuggingFace {
    client: Client,
    endpoint: Url,
    token: Option<BearerToken>,
}

#[bon]
impl HuggingFace {
    #[builder]
    pub fn new(
        #[builder(default = HUGGING_FACE_URL.to_string(), into)] endpoint: String,
        token: Option<BearerToken>,
    ) -> Result<Self, RegistryError> {
        let client = Client::builder().timeout(Duration::from_secs(30)).build().map_err(unable_to_create)?;
        let endpoint = Url::parse(&endpoint).map_err(unable_to_create)?;
        Ok(Self {
            client,
            endpoint,
            token,
        })
    }

    pub async fn files(
        &self,
        repository: &Repository,
    ) -> Result<Vec<File>, RegistryError> {
        let revision = repository
            .commit_hash
            .as_deref()
            .filter(|hash| is_lower_hex(hash, 40))
            .ok_or_else(|| unable_to_get_models(format!("{} has no pinned commit", repository.identifier)))?;
        let mut metadata_url = self
            .url(["api", "models"].into_iter().chain(repository.identifier.split('/')).chain(["revision", revision]))?;
        metadata_url.set_query(Some("blobs=true"));
        let mut request = self.client.get(metadata_url);
        if let Some(token) = &self.token {
            request = request.header(AUTHORIZATION, token.header_value());
        }
        let response =
            request.send().await.and_then(|response| response.error_for_status()).map_err(unable_to_get_models)?;
        let model: HuggingFaceModel = response.json().await.map_err(unable_to_get_models)?;
        if model.sha != revision {
            return Err(unable_to_get_models(format!(
                "Hugging Face returned revision {} instead of {revision}",
                model.sha
            )));
        }
        let files = model
            .siblings
            .iter()
            .filter(|sibling| repository.paths.as_ref().is_none_or(|paths| paths.contains(&sibling.rfilename)))
            .map(|sibling| {
                let name = &sibling.rfilename;
                if name.contains(['\\', ':']) || name.split('/').any(|segment| matches!(segment, "" | "." | "..")) {
                    return Err(unable_to_get_models(format!("invalid file name: {name}")));
                }
                let (size, hash) = match &sibling.lfs {
                    Some(lfs) => (
                        lfs.size.or(sibling.size),
                        lfs.sha256.as_deref().filter(|digest| is_lower_hex(digest, 64)).map(|digest| Hash {
                            method: HashMethod::Sha256,
                            value: digest.to_string(),
                        }),
                    ),
                    None => (
                        sibling.size,
                        sibling.blob_id.as_deref().filter(|digest| is_lower_hex(digest, 40)).map(|digest| Hash {
                            method: HashMethod::GitBlobSha1,
                            value: digest.to_string(),
                        }),
                    ),
                };
                let (Some(size), Some(hash)) = (size, hash) else {
                    return Err(unable_to_get_models(format!("Hugging Face is missing size or digest for {name}")));
                };
                Ok(File {
                    url: self
                        .url(repository.identifier.split('/').chain(["resolve", revision]).chain(name.split('/')))?
                        .into(),
                    name: name.clone(),
                    size: i64::try_from(size)
                        .map_err(|_| unable_to_get_models(format!("file size overflow for {name}")))?,
                    hashes: vec![hash],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if files.is_empty() {
            return Err(unable_to_get_models(format!("{} has no files at {revision}", repository.identifier)));
        }
        Ok(files)
    }

    fn url<'a>(
        &self,
        segments: impl Iterator<Item = &'a str>,
    ) -> Result<Url, RegistryError> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut()
            .map_err(|_| unable_to_get_models("invalid Hugging Face endpoint"))?
            .clear()
            .extend(segments);
        Ok(url)
    }
}

fn is_lower_hex(
    value: &str,
    length: usize,
) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn unable_to_create(error: impl ToString) -> RegistryError {
    RegistryError::UnableToCreate {
        message: error.to_string(),
    }
}

fn unable_to_get_models(error: impl ToString) -> RegistryError {
    RegistryError::UnableToGetModels {
        message: error.to_string(),
    }
}
