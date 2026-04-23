use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::Metadata,
    model::{
        Model, ModelAccessibility, ModelBackend, ModelFamily, ModelProperties, ModelQuantization, ModelRegistry,
        ModelSpecialization, ModelVendor,
    },
};

fn get_metadata(
    metadatas: &Vec<Metadata>,
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelRegistry> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelRegistry {
            identifier: self.id.clone(),
            metadata: metadata,
        });
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelBackend> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelBackend {
            identifier: self.id.clone(),
            version: self.version.clone(),
            metadata: metadata,
        });
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelVendor> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelVendor {
            identifier: self.id.clone(),
            metadata: metadata,
        });
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelFamily> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelFamily {
            identifier: self.id.clone(),
            vendor: vendor,
            metadata: metadata,
        });
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelProperties> {
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelProperties {
            identifier: self.id.clone(),
            size: self.size,
            version: self.version.clone(),
            metadata: metadata,
        });
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
        metadatas: &Vec<Metadata>,
    ) -> Option<ModelQuantization> {
        let vendor = self.vendor.to(metadatas)?;
        let metadata = get_metadata(metadatas, self.metadata_id.clone())?;
        return Some(ModelQuantization {
            identifier: self.id.clone(),
            method: self.method.clone(),
            bits_per_weight: self.bits_per_weight,
            vendor: vendor,
            metadata: metadata,
        });
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
}

impl ResponseModel {
    fn to(
        &self,
        metadatas: &Vec<Metadata>,
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
        let specializations = self.specializations.clone();
        let accessibility = self.accessibility.clone();
        return Some(Model {
            identifier: self.id.clone(),
            registry: registry,
            backends: backends,
            family: family,
            properties: properties,
            quantization: quantization,
            specializations: specializations,
            accessibility: accessibility,
        });
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
        return Some(models);
    }
}
