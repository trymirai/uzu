use std::path::PathBuf;

use uuid::Uuid;

use super::persisted_recovery_metadata;
use crate::{FileCheck, HttpDownloadRequest, RequestHeaders, compute_download_id, traits::DownloadConfig};

impl super::AppleBackendContext {
    #[allow(dead_code)]
    pub(crate) fn event_sink_count_for_download(
        &self,
        download_id: crate::DownloadId,
    ) -> usize {
        self.event_registry
            .lock()
            .map(|registry| registry.keys().filter(|key| key.download_id() == download_id).count())
            .unwrap_or(0)
    }

    #[allow(dead_code)]
    pub(crate) fn event_sink_task_identifiers_for_download(
        &self,
        download_id: crate::DownloadId,
    ) -> Vec<u64> {
        self.event_registry
            .lock()
            .map(|registry| {
                registry
                    .keys()
                    .filter_map(|key| (key.download_id() == download_id).then_some(key.task_identifier()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[test]
fn persisted_recovery_metadata_excludes_request_headers() {
    let destination = PathBuf::from("/tmp/model.bin");
    let config = DownloadConfig {
        download_id: compute_download_id(&destination),
        request: HttpDownloadRequest::with_headers(
            "https://example.test/model.bin",
            RequestHeaders::bearer("secret-token").expect("valid bearer header"),
        ),
        destination,
        artifact_root: PathBuf::from("/tmp/uzu-download-manager-test"),
        file_check: FileCheck::None,
        expected_bytes: None,
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::nil(),
    };

    let json = persisted_recovery_metadata(&config).to_json().expect("RecoveryMetadata must serialize");
    assert!(json.contains("\"schema_version\":1"));
    assert!(json.contains("source_fingerprint"));
    assert!(!json.contains("https://example.test/model.bin"));
    assert!(!json.contains("authorization"));
    assert!(!json.contains("secret-token"));

    let debug = format!("{:?}", persisted_recovery_metadata(&config));
    assert!(!debug.contains("https://example.test/model.bin"));
    assert!(!debug.contains("secret-token"));
}
