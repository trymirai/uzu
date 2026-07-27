use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::{Metadata, Value},
    model::{
        Model, ModelAccessibility, ModelBackend, ModelFamily, ModelProperties, ModelQuantization, ModelRegistry,
        ModelSpecialization, ModelVendor,
    },
};

fn get_metadata(
    metadatas: &[Metadata],
    identifier: String,
) -> Option<Metadata> {
    metadatas.iter().find(|metadata| metadata.identifier == identifier).cloned()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Registry {
    pub id: String,
    pub metadata_id: String,
}

impl Registry {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelRegistry> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelRegistry {
            identifier: self.id.clone(),
            metadata,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Backend {
    pub id: String,
    pub version: String,
    pub metadata_id: String,
}

impl Backend {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelBackend> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelBackend {
            identifier: self.id.clone(),
            version: self.version.clone(),
            metadata,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Vendor {
    pub id: String,
    pub metadata_id: String,
}

impl Vendor {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelVendor> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelVendor {
            identifier: self.id.clone(),
            metadata,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Family {
    pub id: String,
    pub vendor: Vendor,
    pub metadata_id: String,
}

impl Family {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelFamily> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelFamily {
            identifier: self.id.clone(),
            vendor,
            metadata,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Properties {
    pub id: String,
    pub size: i64,
    pub version: Option<String>,
    pub metadata_id: String,
}

impl Properties {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelProperties> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelProperties {
            identifier: self.id.clone(),
            size: self.size,
            version: self.version.clone(),
            metadata,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Quantization {
    pub id: String,
    pub method: String,
    pub bits_per_weight: u32,
    pub vendor: Vendor,
    pub metadata_id: String,
}

impl Quantization {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<ModelQuantization> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        Some(ModelQuantization {
            identifier: self.id.clone(),
            method: self.method.clone(),
            bits_per_weight: self.bits_per_weight,
            vendor,
            metadata,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseModel {
    pub id: String,
    pub registry: Registry,
    pub backends: Vec<Backend>,
    pub family: Option<Family>,
    pub properties: Option<Properties>,
    pub quantization: Option<Quantization>,
    pub specializations: Vec<ModelSpecialization>,
    pub accessibility: ModelAccessibility,
    pub encodings: Vec<Value>,
}

impl ResponseModel {
    fn to(
        &self,
        metadatas: &[Metadata],
    ) -> Option<Model> {
        let registry = self.registry.to(metadatas)?;
        let backends = self.backends.iter().flat_map(|backend| backend.to(metadatas)).collect::<Vec<_>>();
        if backends.len() != self.backends.len() {
            return None;
        }
        let family = match &self.family {
            Some(family) => Some(family.to(metadatas)?),
            None => None,
        };
        let properties = match &self.properties {
            Some(properties) => Some(properties.to(metadatas)?),
            None => None,
        };
        let quantization = match &self.quantization {
            Some(quantization) => Some(quantization.to(metadatas)?),
            None => None,
        };
        Some(Model {
            identifier: self.id.clone(),
            registry,
            backends,
            family,
            properties,
            quantization,
            specializations: self.specializations.clone(),
            accessibility: self.accessibility.clone(),
            encodings: self.encodings.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Response {
    models: Vec<ResponseModel>,
    metadatas: Vec<Metadata>,
}

impl Response {
    pub fn models(&self) -> Option<Vec<Model>> {
        let models =
            self.models.iter().flat_map(|response_model| response_model.to(&self.metadatas)).collect::<Vec<_>>();
        if models.len() != self.models.len() {
            return None;
        }
        Some(models)
    }
}
