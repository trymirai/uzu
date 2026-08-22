use std::time::UNIX_EPOCH;

use shoji::types::{
    basic::{Hash, HashMethod, Repository},
    model::{ModelAccessibility, ModelReference},
};

use super::*;

#[cfg(unix)]
#[tokio::test]
async fn storage_new_rejects_a_symlinked_cache_ancestor_without_writing_through_it() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let outside = temporary.path().join("outside");
    let linked_base = temporary.path().join("linked-base");
    create_dir_all(&outside).unwrap();
    symlink(&outside, &linked_base).unwrap();
    let config = Config::new(
        crate::device::Device {
            os_name: None,
            cpu_name: None,
            memory_total: 0,
            home_path: temporary.path().to_string_lossy().into_owned(),
        },
        Some(linked_base),
        "unsafe-cache-test".to_owned(),
    )
    .with_download_manager_type(download_manager::FileDownloadManagerType::Universal);

    let result = Storage::new(RuntimeHandle::current(), config).await;

    assert!(matches!(result, Err(StorageError::UnableToCreateDirectory { .. })));
    assert!(!outside.join(".cache/unsafe-cache-test").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn legacy_migration_rejects_a_symlinked_models_root() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let config = Config::new(
        crate::device::Device {
            os_name: None,
            cpu_name: None,
            memory_total: 0,
            home_path: temporary.path().to_string_lossy().into_owned(),
        },
        Some(temporary.path().to_path_buf()),
        "legacy-migration-test".to_owned(),
    )
    .with_download_manager_type(download_manager::FileDownloadManagerType::Universal);
    let storage = Storage::new(RuntimeHandle::current(), config.clone()).await.unwrap();
    let outside = temporary.path().join("outside-models");
    create_dir_all(&outside).unwrap();
    let models_root = config.cache_models_path();
    symlink(&outside, &models_root).unwrap();
    let file = File {
        url: "https://example.invalid/model.bin".to_owned(),
        name: "model.bin".to_owned(),
        size: 5,
        hashes: vec![Hash {
            method: HashMethod::CRC32C,
            value: "AAAAAA==".to_owned(),
        }],
    };
    let model = test_model(
        "legacy-model",
        ModelReference::Mirai {
            toolchain_version: "1".to_owned(),
            repository: None,
            source_repository: None,
            files: vec![file.clone()],
        },
    );
    let legacy_path = config.legacy_cache_model_path(&model).unwrap();
    let relative_legacy_path = legacy_path.strip_prefix(&models_root).unwrap();
    create_dir_all(outside.join(relative_legacy_path)).unwrap();
    let safe_path = models_root.join("safe-model");

    let error = storage.migrate_legacy_cache(&model, safe_path.clone(), &[file]).await.unwrap_err();

    assert!(matches!(error, StorageError::DownloadManager { ref message } if message.contains("symlink")));
    assert!(!outside.join("safe-model").exists());
}

#[tokio::test]
async fn verified_legacy_install_copies_only_declared_model_files() {
    let temporary = tempfile::tempdir().unwrap();
    let legacy_root = temporary.path().join("legacy");
    let safe_path = temporary.path().join("safe/model");
    create_dir_all(legacy_root.join("nested")).unwrap();
    let model_path = legacy_root.join("nested/model.bin");
    std::fs::write(&model_path, b"model").unwrap();
    let metadata = std::fs::metadata(&model_path).unwrap();
    let modified = metadata.modified().unwrap().duration_since(UNIX_EPOCH).unwrap();
    let receipt = serde_json::json!({
        "version": 1,
        "crc": "AAAAAA==",
        "file_size": metadata.len(),
        "modified_unix_seconds": modified.as_secs(),
        "modified_nanos": modified.subsec_nanos(),
    });
    std::fs::write(legacy_root.join("nested/model.bin.crc"), serde_json::to_vec(&receipt).unwrap()).unwrap();
    std::fs::write(legacy_root.join("unrelated.txt"), b"keep me").unwrap();
    let files = vec![File {
        url: "https://example.invalid/model.bin".to_owned(),
        name: "nested/model.bin".to_owned(),
        size: 5,
        hashes: vec![Hash {
            method: HashMethod::CRC32C,
            value: "AAAAAA==".to_owned(),
        }],
    }];
    let verified = verified_legacy_files(&legacy_root, &files).await.unwrap();

    install_verified_legacy_files(&legacy_root, &safe_path, &verified).unwrap();

    assert_eq!(std::fs::read(safe_path.join("nested/model.bin")).unwrap(), b"model");
    assert!(!safe_path.join("nested/model.bin.crc").exists());
    assert!(!safe_path.join("unrelated.txt").exists());
    assert!(legacy_root.join("nested/model.bin.crc").exists());
    assert!(legacy_root.join("unrelated.txt").exists());
}

#[test]
fn mirai_and_hugging_face_use_the_same_file_group_contract() {
    let temporary = tempfile::tempdir().unwrap();
    let config = Config::new(
        crate::device::Device {
            os_name: None,
            cpu_name: None,
            memory_total: 0,
            home_path: temporary.path().to_string_lossy().into_owned(),
        },
        Some(temporary.path().to_path_buf()),
        "group-contract-test".to_owned(),
    );
    let mirai_file = File {
        url: "https://example.invalid/mirai/config.json".to_owned(),
        name: "nested/config.json".to_owned(),
        size: 12,
        hashes: vec![Hash {
            method: HashMethod::CRC32C,
            value: "AAAAAA==".to_owned(),
        }],
    };
    let mirai_model = test_model(
        "mirai-fixture",
        ModelReference::Mirai {
            toolchain_version: "1".to_owned(),
            repository: None,
            source_repository: None,
            files: vec![mirai_file.clone()],
        },
    );
    let hugging_face_model = test_model(
        "hf-fixture",
        ModelReference::HuggingFace {
            repository: Repository {
                identifier: "acme/model".to_owned(),
                commit_hash: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
                paths: None,
            },
        },
    );

    let mirai = build_mirai_download(&config, &mirai_model, &[mirai_file]).unwrap();
    let hugging_face = build_hugging_face_download(
        &config,
        &hugging_face_model,
        hugging_face::ResolvedHuggingFaceRepository {
            commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            files: vec![hugging_face::ResolvedHuggingFaceFile {
                relative_path: PathBuf::from("nested/config.json"),
                source_url: "https://example.invalid/hf/config.json".to_owned(),
                size: 12,
                digest: HuggingFaceDigest::GitBlobSha1("3b18e512dba79e4c8300dd08aeb37f8e728b8dad".to_owned()),
            }],
            authorization: None,
        },
    )
    .unwrap();

    for download in [&mirai, &hugging_face] {
        assert_eq!(download.group_spec.files().len(), 1);
        assert_eq!(download.group_spec.files()[0].relative_path.as_path(), Path::new("nested/config.json"));
        assert_eq!(download.group_spec.files()[0].expected_bytes, Some(12));
        assert_ne!(download.group_spec.files()[0].check, FileCheck::None);
        assert_eq!(download.group_spec.destination_root(), download.cache_path);
    }
    assert!(matches!(mirai.group_spec.files()[0].check, FileCheck::CRC(_)));
    assert!(matches!(hugging_face.group_spec.files()[0].check, FileCheck::GitBlobSha1(_)));
}

fn test_model(
    identifier: &str,
    reference: ModelReference,
) -> Model {
    Model::external(
        identifier.to_owned(),
        "registry".to_owned(),
        "Registry".to_owned(),
        "backend".to_owned(),
        "Backend".to_owned(),
        "1".to_owned(),
        Vec::new(),
        ModelAccessibility::Local {
            reference,
        },
        Vec::new(),
    )
}

#[test]
fn legacy_tree_rejects_live_download_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    std::fs::write(temporary.path().join("weights.part"), b"partial").unwrap();

    assert!(!legacy_tree_is_safe(temporary.path()));
}

#[cfg(unix)]
#[test]
fn legacy_tree_rejects_symlinks_without_following_them() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let legacy_root = temporary.path().join("legacy");
    let outside = temporary.path().join("outside");
    create_dir_all(&legacy_root).unwrap();
    create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("weights.bin"), b"outside").unwrap();
    symlink(&outside, legacy_root.join("nested")).unwrap();

    assert!(!legacy_tree_is_safe(&legacy_root));
}

#[cfg(unix)]
#[test]
fn legacy_install_rejects_a_symlinked_destination_ancestor() {
    use std::os::unix::fs::symlink;

    let legacy = tempfile::tempdir().unwrap();
    std::fs::write(legacy.path().join("model.bin"), b"model").unwrap();
    let destination_parent = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let linked_cache = destination_parent.path().join("cache");
    symlink(outside.path(), &linked_cache).unwrap();
    let safe_path = linked_cache.join("model");
    let files = [RelativeFilePath::try_from("model.bin").unwrap()];

    let error = install_verified_legacy_files(legacy.path(), &safe_path, &files).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(!outside.path().join("model").exists());
}

#[cfg(unix)]
#[test]
fn legacy_tree_rejects_unreadable_files() {
    use std::os::unix::fs::PermissionsExt;

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("weights.bin");
    std::fs::write(&path, b"model").unwrap();
    let original_permissions = std::fs::metadata(&path).unwrap().permissions();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let open_is_denied = FsFile::open(&path).is_err();
    let tree_is_safe = legacy_tree_is_safe(temporary.path());
    std::fs::set_permissions(&path, original_permissions).unwrap();

    if open_is_denied {
        assert!(!tree_is_safe);
    }
}
