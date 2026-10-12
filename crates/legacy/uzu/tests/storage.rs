#![cfg(not(target_family = "wasm"))]

mod common;

use std::{sync::Arc, time::Duration};

use download_manager::{DestinationLock, LockOwner};
use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry, artifact_path};
use rstest::rstest;
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uuid::Uuid;
use uzu::{
    engine::Downloader,
    registry::mirai::HUGGING_FACE_URL,
    storage::{DownloadManagerType, DownloadPhase, DownloadState, Storage},
    types::{
        basic::{File, Repository},
        model::{Model, ModelAccessibility, ModelIdentifier, ModelSource},
    },
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, header_exists, method, path},
};

use crate::common::TestStorage;

const HELLO: &[u8] = b"hello\n";
const REVISION: &str = "f5dec40ceb1d8c4b15f049bf5e185dd7be2cc150";

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn model_lifecycle(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, HUGGING_FACE_URL, None).await?;
    let storage = &test_storage.storage;
    let identifier = model.identifier.clone();
    let cache_path = storage.cache_model_path(model).ok_or("model must have a cache path")?;
    let mut events = storage.subscribe();
    let total_bytes = storage.state(&identifier).await?.total_bytes;

    storage.download(&identifier).await?;
    let downloading = wait_for(storage, &identifier, &mut events, |state| {
        state.total_bytes > 0 && state.downloaded_bytes >= state.total_bytes / 4
    })
    .await;
    assert!(matches!(downloading.phase, DownloadPhase::Downloading {}), "got {:?}", downloading.phase);

    storage.pause(&identifier).await?;
    let paused =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(paused.downloaded_bytes >= total_bytes / 4);

    storage.download(&identifier).await?;
    let downloaded =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Downloaded {})).await;
    assert_eq!(downloaded.downloaded_bytes, total_bytes);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served.bytes.to_vec());
        assert!(!artifact_path(&destination, "checksum").exists());
    }

    storage.delete(&identifier).await?;
    let deleted = storage.state(&identifier).await?;
    assert!(matches!(deleted.phase, DownloadPhase::NotDownloaded {}));
    assert_eq!(deleted.downloaded_bytes, 0);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert!(!destination.exists());
        assert!(!artifact_path(&destination, "checksum").exists());
    }

    let served = registry.files.first().ok_or("mock registry must include files")?;
    let destination = cache_path.join(&served.file.name);
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(&destination, served.bytes.as_ref()).await?;
    let _lock = DestinationLock::acquire(
        &destination,
        &LockOwner {
            manager_id: "foreign-manager".to_string(),
            instance_id: Uuid::new_v4(),
        },
    )
    .await?;
    let refused = storage.delete(&identifier).await.expect_err("delete must be refused while locked");
    assert!(refused.to_string().contains("foreign-manager"), "unexpected error: {refused}");
    assert!(destination.exists());

    let mut changed = model.clone();
    if let ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            files,
            ..
        },
    } = &mut changed.accessibility
    {
        files.truncate(1);
    }
    storage.refresh(&[changed.clone()], false).await?;
    assert_eq!(storage.ready_state(&identifier).await?.total_bytes, served.file.size);
    storage.refresh(&[model.clone(), changed.clone()], false).await?;
    assert_eq!(storage.ready_state(&identifier).await?.total_bytes, total_bytes);
    storage.refresh(&[changed, model.clone()], false).await?;
    assert_eq!(storage.ready_state(&identifier).await?.total_bytes, served.file.size);
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn downloader_streams(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, HUGGING_FACE_URL, None).await?;
    let storage = Arc::clone(&test_storage.storage);
    let identifier = model.identifier.clone();
    let downloader = Downloader::new(identifier.clone(), Arc::clone(&storage));
    let mut events = storage.subscribe();

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    wait_for(&storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Downloading {})).await;
    downloader.pause().await?;
    timeout(Duration::from_secs(10), async { while progress.next().await.is_some() {} }).await?;
    let paused =
        wait_for(&storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(!paused.is_in_progress());
    assert!(!paused.can_pause());

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    let last = timeout(Duration::from_secs(30), async {
        let mut last = None;
        while let Some(state) = progress.next().await {
            last = Some(state);
        }
        last
    })
    .await?;
    assert!(matches!(last.map(|state| state.phase), Some(DownloadPhase::Downloaded {})));
    assert!(downloader.progress().await?.next().await.is_none());
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn hugging_face_model(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let hugging_face = MockServer::start().await;
    let route = format!("/trymirai/model/resolve/{REVISION}/model.safetensors");
    Mock::given(method("GET"))
        .and(path(route.clone()))
        .and(header("authorization", "Bearer hf_test"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(HELLO))
        .mount(&hugging_face)
        .await;
    // Like the Mirai CDN, this origin rejects any bearer token, so its file must download anonymously.
    let cdn = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/config.json"))
        .and(header_exists("authorization"))
        .respond_with(ResponseTemplate::new(401))
        .with_priority(1)
        .mount(&cdn)
        .await;
    Mock::given(method("GET"))
        .and(path("/config.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(HELLO))
        .mount(&cdn)
        .await;
    let files = vec![
        File {
            url: format!("{}/config.json", cdn.uri()),
            name: "config.json".to_string(),
            size: HELLO.len() as i64,
            hashes: vec![],
        },
        File {
            url: format!("{}{route}", hugging_face.uri()),
            name: "model.safetensors".to_string(),
            size: HELLO.len() as i64,
            hashes: vec![],
        },
    ];
    let model = Model::external(
        "pinned".to_string(),
        "mirai".to_string(),
        "Mirai".to_string(),
        "uzu".to_string(),
        "Uzu".to_string(),
        "1".to_string(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".to_string(),
                repository: Some(Repository {
                    identifier: "trymirai/model".to_string(),
                    commit_hash: Some(REVISION.to_string()),
                    paths: None,
                }),
                source_repository: None,
                files,
            },
        },
        None,
    );
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, &hugging_face.uri(), Some("hf_test"))
            .await?;
    let storage = &test_storage.storage;
    let cache_path = storage.cache_model_path(&model).ok_or("model must have a cache path")?;
    assert_eq!(cache_path.file_name().and_then(|name| name.to_str()), Some(REVISION));
    let mut events = storage.subscribe();

    storage.download(&model.identifier).await?;
    let state = wait_for(storage, &model.identifier, &mut events, |state| {
        matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. })
    })
    .await;
    assert!(matches!(state.phase, DownloadPhase::Downloaded {}), "got {:?}", state.phase);
    for name in ["config.json", "model.safetensors"] {
        let destination = cache_path.join(name);
        assert_eq!(tokio::fs::read(&destination).await?, HELLO);
        assert!(!artifact_path(&destination, "checksum").exists());
    }
    Ok(())
}

async fn wait_for(
    storage: &Storage,
    identifier: &ModelIdentifier,
    events: &mut BroadcastStream<(ModelIdentifier, DownloadState)>,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        let state = storage.state(identifier).await.expect("model state must exist");
        if is_expected(&state) {
            return state;
        }
        while let Some(result) = events.next().await {
            let (event_identifier, state) = result.expect("storage event stream must not lag");
            if &event_identifier == identifier && is_expected(&state) {
                return state;
            }
        }
        panic!("storage event stream ended before the expected state");
    })
    .await
    .expect("timed out waiting for storage state")
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_model_does_not_hide_a_cached_model() -> Result<(), Box<dyn std::error::Error>> {
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let storage = &test_storage.storage;
    let healthy = catalog_model("healthy", &[("model.bin", HELLO.len() as i64)]);
    let cache_path = storage.cache_model_path(&healthy).unwrap();
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(cache_path.join("model.bin"), HELLO).await?;

    // Two incompatible requests for the same file make only this model uninitializable.
    let malformed = catalog_model("broken", &[("model.bin", 1), ("model.bin", 2)]);
    timeout(Duration::from_secs(2), storage.refresh(&[malformed.clone(), healthy.clone()], true)).await??;
    let healthy_state = timeout(Duration::from_secs(2), storage.ready_state(&healthy.identifier)).await??;
    assert_eq!(healthy_state.phase, DownloadPhase::Downloaded {});
    assert_eq!(healthy_state.downloaded_bytes, HELLO.len() as i64);
    let error = timeout(Duration::from_secs(2), storage.ready_state(&malformed.identifier))
        .await?
        .expect_err("conflicting files must fail initialization");
    assert!(error.to_string().contains("conflicting download config"), "unexpected error: {error}");
    assert!(matches!(storage.state(&malformed.identifier).await?.phase, DownloadPhase::Error { .. }));
    assert_eq!(storage.states().await.len(), 2);

    let repaired = catalog_model("broken", &[("model.bin", 1)]);
    storage.refresh(std::slice::from_ref(&repaired), false).await?;
    let repaired_state = timeout(Duration::from_secs(2), storage.ready_state(&repaired.identifier)).await??;
    assert_eq!(repaired_state.phase, DownloadPhase::NotDownloaded {});
    assert_eq!(storage.state(&healthy.identifier).await?, healthy_state);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn partial_catalog_keeps_existing_models_and_complete_catalog_removes_obsolete_files()
-> Result<(), Box<dyn std::error::Error>> {
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let storage = &test_storage.storage;
    let cached = catalog_model("cached", &[("model.bin", HELLO.len() as i64)]);
    let cache_path = storage.cache_model_path(&cached).unwrap();
    let destination = cache_path.join("model.bin");
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(&destination, HELLO).await?;
    storage.refresh(std::slice::from_ref(&cached), true).await?;
    let cached_state = timeout(Duration::from_secs(2), storage.ready_state(&cached.identifier)).await??;
    assert_eq!(cached_state.phase, DownloadPhase::Downloaded {});

    let added = catalog_model("added", &[("model.bin", 3)]);
    storage.refresh(std::slice::from_ref(&added), false).await?;
    assert_eq!(storage.state(&cached.identifier).await?, cached_state);
    assert_eq!(storage.states().await.len(), 2);
    let added_state = timeout(Duration::from_secs(2), storage.ready_state(&added.identifier)).await??;
    assert_eq!(added_state.phase, DownloadPhase::NotDownloaded {});
    storage.refresh(&[], false).await?;
    assert_eq!(storage.state(&cached.identifier).await?, cached_state);
    assert_eq!(storage.state(&added.identifier).await?, added_state);

    storage.refresh(&[added], true).await?;
    assert!(storage.state(&cached.identifier).await.is_err());
    assert_eq!(storage.states().await.len(), 1);
    assert!(!destination.exists(), "a complete catalog must clean obsolete model files");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn checkpoint_cleanup_waits_for_complete_catalog_and_skips_locked_downloads()
-> Result<(), Box<dyn std::error::Error>> {
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let storage = &test_storage.storage;
    let old = catalog_model("revised", &[("model.bin", HELLO.len() as i64)]);
    let old_path = storage.cache_model_path(&old).unwrap();
    tokio::fs::create_dir_all(&old_path).await?;
    tokio::fs::write(old_path.join("model.bin"), HELLO).await?;
    storage.refresh(std::slice::from_ref(&old), true).await?;
    assert_eq!(storage.ready_state(&old.identifier).await?.phase, DownloadPhase::Downloaded {});

    let mut current = old.clone();
    if let ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            toolchain_version,
            ..
        },
    } = &mut current.accessibility
    {
        *toolchain_version = "2".into();
    }
    let current_path = storage.cache_model_path(&current).unwrap();
    tokio::fs::create_dir_all(&current_path).await?;
    tokio::fs::write(current_path.join("model.bin"), HELLO).await?;
    let lock = DestinationLock::acquire(
        &old_path.join("model.bin"),
        &LockOwner {
            manager_id: "active-downloader".into(),
            instance_id: Uuid::new_v4(),
        },
    )
    .await?;

    storage.refresh(std::slice::from_ref(&current), true).await?;
    assert_eq!(tokio::fs::read(old_path.join("model.bin")).await?, HELLO);
    assert_eq!(storage.ready_state(&current.identifier).await?.phase, DownloadPhase::Downloaded {});
    drop(lock);
    storage.refresh(std::slice::from_ref(&current), false).await?;
    assert!(old_path.exists(), "partial discovery must not remove checkpoints");
    storage.refresh(&[], true).await?;
    assert!(old_path.exists(), "an empty catalog must not erase the cache");
    #[cfg(unix)]
    {
        let link = old_path.join("linked-checkpoint");
        std::os::unix::fs::symlink(&old_path, &link)?;
        timeout(Duration::from_secs(2), storage.refresh(std::slice::from_ref(&current), true)).await??;
        assert!(old_path.exists(), "cleanup must not traverse linked checkpoint contents");
        tokio::fs::remove_file(link).await?;
    }
    storage.refresh(std::slice::from_ref(&current), true).await?;
    assert!(!old_path.exists());
    assert_eq!(tokio::fs::read(current_path.join("model.bin")).await?, HELLO);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn changed_file_list_replaces_live_task_without_conflicting_with_it() -> Result<(), Box<dyn std::error::Error>> {
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let storage = &test_storage.storage;
    let full = catalog_model("changing", &[("model.bin", HELLO.len() as i64), ("config.json", 2)]);
    let reduced = catalog_model("changing", &[("model.bin", HELLO.len() as i64)]);
    let cache_path = storage.cache_model_path(&full).unwrap();
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(cache_path.join("model.bin"), HELLO).await?;
    tokio::fs::write(cache_path.join("config.json"), b"{}").await?;

    for index in 0..20 {
        let (model, expected_bytes) = if index % 2 == 0 {
            (&full, HELLO.len() + 2)
        } else {
            (&reduced, HELLO.len())
        };
        storage.refresh(std::slice::from_ref(model), false).await?;
        let state = timeout(Duration::from_secs(2), storage.ready_state(&model.identifier)).await??;
        assert_eq!(state.phase, DownloadPhase::Downloaded {}, "replacement {index}");
        assert_eq!(state.total_bytes, expected_bytes as i64);
        assert_eq!(state.downloaded_bytes, expected_bytes as i64);
    }
    assert_eq!(tokio::fs::read(cache_path.join("config.json")).await?, b"{}");
    Ok(())
}

fn catalog_model(
    identifier: &str,
    files: &[(&str, i64)],
) -> Model {
    Model::external(
        identifier.to_string(),
        "test".to_string(),
        "Test".to_string(),
        "uzu".to_string(),
        "Uzu".to_string(),
        "1".to_string(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".to_string(),
                repository: None,
                source_repository: None,
                files: files
                    .iter()
                    .map(|(name, size)| File {
                        url: format!("https://example.invalid/{identifier}/{name}"),
                        name: (*name).to_string(),
                        size: *size,
                        hashes: vec![],
                    })
                    .collect(),
            },
        },
        None,
    )
}
