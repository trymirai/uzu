use download_manager::FileCheck;
use serde::{Deserialize, Serialize};
use shoji::types::{basic::File, model::Model};

const CACHE_VERSION: u8 = 2;

#[derive(Clone, Serialize, Deserialize)]
pub struct ResolvedModels {
    version: u8,
    models: Vec<ResolvedModel>,
    #[serde(skip)]
    reusable_hugging_face_files: bool,
}

impl Default for ResolvedModels {
    fn default() -> Self {
        Self {
            version: CACHE_VERSION,
            models: Vec::new(),
            reusable_hugging_face_files: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    model: Model,
    files: Option<Vec<ResolvedFile>>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedFile {
    pub file: File,
    pub check: FileCheck,
    pub requires_authentication: bool,
}

impl ResolvedModels {
    pub fn new(models: Vec<ResolvedModel>) -> Self {
        Self {
            version: CACHE_VERSION,
            models,
            reusable_hugging_face_files: true,
        }
    }

    pub fn models(&self) -> Vec<Model> {
        self.models.iter().map(|resolved| resolved.model.clone()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ResolvedModel> {
        self.models.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn validate_cache(mut self) -> Option<Self> {
        if self.version != CACHE_VERSION {
            return None;
        }
        self.reusable_hugging_face_files = false;
        Some(self)
    }

    pub fn reusable_hugging_face_files(
        &self,
        model: &Model,
    ) -> Option<Vec<ResolvedFile>> {
        if !self.reusable_hugging_face_files {
            return None;
        }
        self.models
            .iter()
            .find(|resolved| {
                resolved.model.identifier == model.identifier && resolved.model.accessibility == model.accessibility
            })
            .and_then(|resolved| resolved.files.clone())
    }
}

impl ResolvedModel {
    pub fn downloadable(
        model: Model,
        files: Vec<ResolvedFile>,
    ) -> Self {
        Self {
            model,
            files: Some(files),
        }
    }

    pub fn passthrough(model: Model) -> Self {
        Self {
            model,
            files: None,
        }
    }

    pub fn parts(&self) -> (&Model, Option<&[ResolvedFile]>) {
        (&self.model, self.files.as_deref())
    }
}
