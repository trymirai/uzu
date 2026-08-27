use hanashi::chat::EncodingConfig;
use serde::{Deserialize, Deserializer};
use shoji::types::{
    basic::{File, Metadata, Repository, Value},
    model::{
        Model, ModelAccessibility, ModelBackend, ModelFamily, ModelProperties, ModelQuantization, ModelRegistry,
        ModelSource, ModelSpecialization, ModelVendor,
    },
};

fn get_metadata(
    metadatas: &[Metadata],
    identifier: &str,
) -> Result<Metadata, String> {
    metadatas
        .iter()
        .find(|metadata| metadata.identifier == identifier)
        .cloned()
        .ok_or_else(|| format!("missing metadata {identifier}"))
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Registry {
    id: String,
    metadata_id: String,
}

impl Registry {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelRegistry, String> {
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelRegistry {
            identifier: self.id.clone(),
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Backend {
    id: String,
    version: String,
    metadata_id: String,
}

impl Backend {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelBackend, String> {
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelBackend {
            identifier: self.id.clone(),
            version: self.version.clone(),
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Vendor {
    id: String,
    metadata_id: String,
}

impl Vendor {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelVendor, String> {
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelVendor {
            identifier: self.id.clone(),
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Family {
    id: String,
    vendor: Vendor,
    metadata_id: String,
}

impl Family {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelFamily, String> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelFamily {
            identifier: self.id.clone(),
            vendor,
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Properties {
    id: String,
    size: i64,
    version: Option<String>,
    metadata_id: String,
}

impl Properties {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelProperties, String> {
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelProperties {
            identifier: self.id.clone(),
            size: self.size,
            version: self.version.clone(),
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Quantization {
    id: String,
    method: String,
    bits_per_weight: u32,
    vendor: Vendor,
    metadata_id: String,
}

impl Quantization {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<ModelQuantization, String> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, &self.metadata_id)?;
        Ok(ModelQuantization {
            identifier: self.id.clone(),
            method: self.method.clone(),
            bits_per_weight: self.bits_per_weight,
            vendor,
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FetchedAccessibility {
    Local {
        reference: FetchedReference,
    },
    Remote {
        repository: Option<Repository>,
    },
}

impl From<&FetchedAccessibility> for ModelAccessibility {
    fn from(accessibility: &FetchedAccessibility) -> Self {
        match accessibility {
            FetchedAccessibility::Local {
                reference,
            } => Self::OnDevice {
                source: reference.into(),
            },
            FetchedAccessibility::Remote {
                repository,
            } => Self::Remote {
                repository: repository.clone(),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FetchedReference {
    Mirai {
        toolchain_version: String,
        repository: Option<Repository>,
        source_repository: Option<Repository>,
        files: Vec<File>,
    },
    Local {
        path: String,
    },
}

impl From<&FetchedReference> for ModelSource {
    fn from(reference: &FetchedReference) -> Self {
        match reference {
            FetchedReference::Mirai {
                toolchain_version,
                repository,
                source_repository,
                files,
            } => Self::Managed {
                toolchain_version: toolchain_version.clone(),
                repository: repository.clone(),
                source_repository: source_repository.clone(),
                files: files.clone(),
            },
            FetchedReference::Local {
                path,
            } => Self::Filesystem {
                path: path.clone(),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct FetchedModel {
    id: String,
    registry: Registry,
    backends: Vec<Backend>,
    family: Option<Family>,
    properties: Option<Properties>,
    quantization: Option<Quantization>,
    specializations: Vec<ModelSpecialization>,
    accessibility: FetchedAccessibility,
    #[serde(rename = "encodings", default, deserialize_with = "deserialize_encoding")]
    encoding: Option<Value>,
}

fn deserialize_encoding<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    let encodings = Vec::<Value>::deserialize(deserializer)?;
    Ok(EncodingConfig::select(&encodings))
}

impl FetchedModel {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Result<Model, String> {
        let registry = self.registry.to(metadatas)?;
        let backends = self.backends.iter().map(|backend| backend.to(metadatas)).collect::<Result<Vec<_>, _>>()?;
        let family = self.family.as_ref().map(|family| family.to(metadatas)).transpose()?;
        let properties = self.properties.as_ref().map(|properties| properties.to(metadatas)).transpose()?;
        let quantization = self.quantization.as_ref().map(|quantization| quantization.to(metadatas)).transpose()?;
        Ok(Model {
            identifier: self.id.clone(),
            registry,
            backends,
            family,
            properties,
            quantization,
            specializations: self.specializations.clone(),
            accessibility: (&self.accessibility).into(),
            encoding: self.encoding.clone(),
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FetchedModels {
    models: Vec<FetchedModel>,
    metadatas: Vec<Metadata>,
}

impl FetchedModels {
    pub fn models(&self) -> Result<Vec<Model>, String> {
        self.models.iter().map(|model| model.to(&self.metadatas)).collect()
    }
}

#[derive(Deserialize)]
pub struct HuggingFaceModelResponse {
    pub sha: String,
    pub private: Option<bool>,
    pub gated: Option<HuggingFaceGatedResponse>,
    pub siblings: Vec<HuggingFaceFileResponse>,
}

impl HuggingFaceModelResponse {
    pub fn requires_authentication(&self) -> bool {
        if self.private != Some(false) {
            return true;
        }
        match &self.gated {
            Some(HuggingFaceGatedResponse::Boolean(false)) => false,
            Some(HuggingFaceGatedResponse::Approval(approval)) => {
                matches!(approval, HuggingFaceGatedApproval::Auto | HuggingFaceGatedApproval::Manual)
            },
            Some(HuggingFaceGatedResponse::Boolean(true)) | None => true,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum HuggingFaceGatedResponse {
    Boolean(bool),
    Approval(HuggingFaceGatedApproval),
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HuggingFaceGatedApproval {
    Auto,
    Manual,
}

#[derive(Deserialize)]
pub struct HuggingFaceFileResponse {
    pub rfilename: String,
    pub size: Option<u64>,
    #[serde(rename = "blobId")]
    pub blob_id: Option<String>,
    pub lfs: Option<HuggingFaceLfsResponse>,
}

#[derive(Deserialize)]
pub struct HuggingFaceLfsResponse {
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

#[derive(Deserialize)]
pub struct OpenAIModelsResponse {
    pub data: Vec<OpenAIModelResponse>,
}

#[derive(Deserialize)]
pub struct OpenAIModelResponse {
    pub id: String,
}

#[cfg(test)]
#[path = "../../tests/unit/api/response_test.rs"]
mod tests;
