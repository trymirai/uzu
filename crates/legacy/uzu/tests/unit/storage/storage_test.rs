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
