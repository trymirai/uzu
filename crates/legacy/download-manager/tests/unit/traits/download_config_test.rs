use std::path::Path;

use crate::{compute_download_id, traits::DownloadConfig};

impl DownloadConfig {
    pub(crate) fn resume_artifact_path_for(
        destination: &Path,
        download_id: crate::DownloadId,
        extension: &str,
    ) -> std::path::PathBuf {
        Self::default_artifact_root(destination, download_id).join(format!("download.{extension}"))
    }

    pub(crate) fn integrity_receipt_path_for(
        destination: &Path,
        download_id: crate::DownloadId,
    ) -> std::path::PathBuf {
        Self::default_artifact_root(destination, download_id).join("integrity.json")
    }
}

#[test]
fn resume_artifact_paths_do_not_collide_for_different_destination_extensions() {
    let binary = Path::new("/models/weights.bin");
    let safetensors = Path::new("/models/weights.safetensors");

    let binary_artifact = DownloadConfig::resume_artifact_path_for(binary, compute_download_id(binary), "part");
    let safetensors_artifact =
        DownloadConfig::resume_artifact_path_for(safetensors, compute_download_id(safetensors), "part");
    let binary_receipt = DownloadConfig::integrity_receipt_path_for(binary, compute_download_id(binary));

    assert_ne!(binary_artifact, safetensors_artifact);
    assert_ne!(binary_artifact.parent(), binary.parent());
    assert_ne!(safetensors_artifact.parent(), safetensors.parent());
    assert_eq!(
        binary_artifact,
        binary
            .parent()
            .unwrap()
            .join(".uzu-download-manager")
            .join(compute_download_id(binary).to_string())
            .join("download.part")
    );
    assert_eq!(
        safetensors_artifact,
        safetensors
            .parent()
            .unwrap()
            .join(".uzu-download-manager")
            .join(compute_download_id(safetensors).to_string())
            .join("download.part")
    );
    assert_eq!(binary_receipt, binary_artifact.parent().unwrap().join("integrity.json"));
}
