use shoji::types::model::Model;

use super::{downloader::weights_size, local_artifact::LocalArtifact, resolved_target::ResolvedTarget};
use crate::types::SourceMode;

pub enum BenchmarkTarget {
    Registry(Model),
    Local(LocalArtifact),
}

impl BenchmarkTarget {
    pub fn id(&self) -> &str {
        match self {
            Self::Registry(model) => &model.identifier,
            Self::Local(artifact) => &artifact.id,
        }
    }

    /// Best available estimate of the resident footprint, used to skip models that cannot fit.
    pub fn estimated_memory_bytes(&self) -> Option<u64> {
        match self {
            Self::Registry(model) => weights_size(model)
                .map(|size| size as u64)
                .or_else(|| model.properties.as_ref().map(|properties| properties.size as u64)),
            Self::Local(artifact) => Some(artifact.header_summary.logical_payload_bytes),
        }
    }

    /// Local targets are already on disk, so they resolve without a downloader.
    pub fn resolved_locally(&self) -> Option<ResolvedTarget> {
        let Self::Local(artifact) = self else {
            return None;
        };
        Some(ResolvedTarget {
            source: SourceMode::Local,
            id: artifact.id.clone(),
            model_dir: artifact.model_dir.clone(),
            header_path: artifact.header_path.clone(),
        })
    }
}
