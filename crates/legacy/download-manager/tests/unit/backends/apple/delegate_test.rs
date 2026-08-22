use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use super::{
    AppleSinkKey, DestinationInstallBarrier, download_location_path, install_downloaded_file, is_https_to_http,
    is_success_http_status,
};

impl AppleSinkKey {
    pub(crate) fn download_id(self) -> crate::DownloadId {
        self.download_id
    }

    pub(crate) fn task_identifier(self) -> u64 {
        self.task_identifier
    }
}

#[test]
fn task_identifiers_from_different_sessions_do_not_share_a_sink_key() {
    let download_id = uuid::Uuid::nil();
    let background = AppleSinkKey {
        session_identity: 1,
        download_id,
        task_identifier: 1,
    };
    let authenticated = AppleSinkKey {
        session_identity: 2,
        download_id,
        task_identifier: 1,
    };

    assert_ne!(background, authenticated);
}

#[test]
fn only_success_statuses_may_be_installed() {
    assert!(is_success_http_status(200));
    assert!(is_success_http_status(206));
    assert!(is_success_http_status(299));
    for status in [199, 300, 401, 403, 404, 410, 429, 500] {
        assert!(!is_success_http_status(status));
    }
}

#[test]
fn rejects_only_https_to_http_redirects() {
    assert!(is_https_to_http(Some("https"), Some("http")));
    assert!(!is_https_to_http(Some("https"), Some("https")));
    assert!(!is_https_to_http(Some("http"), Some("http")));
    assert!(!is_https_to_http(Some("http"), Some("https")));
    assert!(!is_https_to_http(None, Some("http")));
}

#[test]
fn missing_download_location_is_a_protocol_error() {
    assert_eq!(
        download_location_path(None),
        Err(crate::DownloadError::Protocol("download completed without a temporary file path".to_string()))
    );
}

#[test]
fn cancelled_destination_install_barrier_skips_installation() {
    let barrier = DestinationInstallBarrier::default();
    barrier.prevent_installation();

    assert_eq!(barrier.install(|| "installed"), None);
}

#[test]
fn cancellation_waits_for_an_in_flight_installation() {
    let barrier = DestinationInstallBarrier::default();
    let cancel_barrier = barrier.clone();
    let installed = Arc::new(AtomicBool::new(false));
    let install_result = Arc::clone(&installed);
    let (cancel_started_sender, cancel_started_receiver) = mpsc::channel();
    let (cancelled_sender, cancelled_receiver) = mpsc::channel();

    let result = barrier.install(|| {
        let cancel_thread = thread::spawn(move || {
            cancel_started_sender.send(()).unwrap();
            cancel_barrier.prevent_installation();
            cancelled_sender.send(()).unwrap();
        });
        cancel_started_receiver.recv().unwrap();
        assert!(cancelled_receiver.try_recv().is_err());
        install_result.store(true, Ordering::SeqCst);
        cancel_thread
    });

    result.unwrap().join().unwrap();
    assert!(installed.load(Ordering::SeqCst));
    assert_eq!(barrier.install(|| "late install"), None);
}

#[test]
fn installation_renames_an_owned_staging_file_into_place() {
    let directory = tempfile::tempdir().unwrap();
    let temporary_path = directory.path().join("url-session.tmp");
    let installation_artifact = directory.path().join(".uzu-download-id.installing");
    let destination = directory.path().join("model.bin");
    std::fs::write(&temporary_path, b"model").unwrap();

    install_downloaded_file(&temporary_path, &installation_artifact, &destination).unwrap();

    assert_eq!(std::fs::read(destination).unwrap(), b"model");
    assert!(!temporary_path.exists());
    assert!(!installation_artifact.exists());
}

#[test]
fn installation_rejects_a_symlinked_destination_ancestor() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let temporary_path = directory.path().join("url-session.tmp");
    let installation_artifact = directory.path().join("artifacts/installing");
    let destination_parent = directory.path().join("models");
    symlink(outside.path(), &destination_parent).unwrap();
    std::fs::create_dir_all(installation_artifact.parent().unwrap()).unwrap();
    std::fs::write(&temporary_path, b"model").unwrap();

    let error = install_downloaded_file(&temporary_path, &installation_artifact, &destination_parent.join("model.bin"))
        .unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(!outside.path().join("model.bin").exists());
}

#[test]
fn installation_rejects_a_symlinked_artifact_ancestor() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let temporary_path = directory.path().join("url-session.tmp");
    let artifact_parent = directory.path().join("artifacts");
    let destination = directory.path().join("models/model.bin");
    symlink(outside.path(), &artifact_parent).unwrap();
    std::fs::write(&temporary_path, b"model").unwrap();

    let error =
        install_downloaded_file(&temporary_path, &artifact_parent.join("installing"), &destination).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(!outside.path().join("installing").exists());
    assert!(!destination.exists());
}
