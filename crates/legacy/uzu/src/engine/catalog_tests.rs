use std::{future::Future, pin::Pin, sync::Mutex, time::Duration};

use shoji::types::{basic::Repository, model::ModelAccessibility};
use tempfile::TempDir;
use tokio::sync::Notify;

use super::*;
use crate::storage::DownloadManagerType;

type RegistryResult = Result<Vec<Model>, RegistryError>;

struct ControlledRegistry {
    identifier: &'static str,
    cached: Option<Vec<Model>>,
    started: Arc<Notify>,
    result: Mutex<Option<oneshot::Receiver<RegistryResult>>>,
}

impl Registry for ControlledRegistry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        self.identifier.to_string()
    }

    fn cached_listing(&self) -> Option<(Vec<Model>, bool)> {
        self.cached.clone().map(|models| (models, true))
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = RegistryResult> + Send + '_>> {
        Box::pin(async {
            let result = self.result.lock().unwrap().take().expect("registry fetched more than once");
            self.started.notify_one();
            result.await.expect("test dropped registry result")
        })
    }
}

fn registry(
    identifier: &'static str,
    cached: Option<Vec<Model>>,
) -> (Box<dyn Registry<Error = RegistryError>>, oneshot::Sender<RegistryResult>, Arc<Notify>) {
    let (sender, result) = oneshot::channel();
    let started = Arc::new(Notify::new());
    (
        Box::new(ControlledRegistry {
            identifier,
            cached,
            started: started.clone(),
            result: Mutex::new(Some(result)),
        }),
        sender,
        started,
    )
}

fn model(identifier: &str) -> Model {
    Model::external(
        identifier.to_string(),
        "test".to_string(),
        "Test".to_string(),
        "test".to_string(),
        "Test".to_string(),
        "1".to_string(),
        vec![],
        ModelAccessibility::Remote {
            repository: None,
        },
        None,
    )
}

fn model_with_repo(
    identifier: &str,
    repo_id: &str,
) -> Model {
    Model {
        accessibility: ModelAccessibility::Remote {
            repository: Some(Repository {
                identifier: repo_id.to_string(),
                commit_hash: None,
                paths: None,
            }),
        },
        ..model(identifier)
    }
}

async fn engine() -> (Engine, TempDir) {
    let directory = tempfile::tempdir().unwrap();
    let config = StorageConfig::new(
        Device::new().unwrap(),
        Some(directory.path().to_path_buf()),
        "catalog-test".to_string(),
        DownloadManagerType::Universal,
        "https://example.invalid".to_string(),
        None,
    );
    let storage = Arc::new(Storage::new(RuntimeHandle::current(), config).await.unwrap());
    (
        Engine {
            registry: SharedAccess::new(MergedRegistry::new(vec![])),
            storage,
            backends: SharedAccess::new(HashMap::new()),
            callback: SharedAccess::new(None),
            catalog_events: broadcast::channel(64).0,
            catalog_refreshes: Arc::new(AtomicUsize::new(0)),
            catalog_publish: Arc::new(tokio::sync::Mutex::new(())),
            published_catalog: Arc::new(tokio::sync::Mutex::new((Vec::new(), true))),
        },
        directory,
    )
}

#[tokio::test]
async fn refresh_models_waits_for_in_flight_catalog_publication() {
    let (engine, _directory) = engine().await;
    let (registry, complete, started) = registry("refresh", Some(vec![model("cached")]));
    engine.start_registry(registry).await.unwrap();
    started.notified().await;
    let refresh = tokio::spawn({
        let engine = engine.clone();
        async move { engine.refresh_models().await }
    });
    tokio::task::yield_now().await;
    assert!(!refresh.is_finished());
    complete.send(Ok(vec![model("updated")])).unwrap();
    let models = tokio::time::timeout(Duration::from_secs(1), refresh).await.unwrap().unwrap().unwrap();
    assert_eq!(models, vec![model("updated")]);
    assert!(!engine.catalog_is_refreshing());
}

#[tokio::test]
async fn chat_model_info_derives_capabilities_from_the_catalog_encoding() {
    use shoji::types::{
        basic::Value,
        model::{ModelSource, ModelSpecialization},
    };

    let (engine, _directory) = engine().await;
    let model = Model::external(
        "metadata-test".into(),
        "local".into(),
        "Local".into(),
        "uzu".into(),
        "Uzu".into(),
        "1".into(),
        vec![ModelSpecialization::Chat {}],
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".into(),
                repository: None,
                source_repository: None,
                files: vec![],
            },
        },
        Some(Value::from(serde_json::json!({ "type": "hanashi", "name": "qwen3.5" }))),
    );
    let info = engine.chat_model_info(&model).await.unwrap().unwrap();
    assert_eq!(info.context_length, None);
    assert!(info.capabilities.supports_tools);
    assert!(info.capabilities.supports_reasoning);
    assert!(info.capabilities.supports_disable_reasoning);
}

#[tokio::test]
async fn chat_model_info_returns_none_for_models_without_a_local_uzu_chat_config() {
    use shoji::types::{
        basic::Value,
        model::{ModelSource, ModelSpecialization},
    };

    let (engine, _directory) = engine().await;
    let local = |specializations: Vec<ModelSpecialization>, backend: &str| {
        Model::external(
            "metadata-test".into(),
            "local".into(),
            "Local".into(),
            backend.into(),
            "Uzu".into(),
            "1".into(),
            specializations,
            ModelAccessibility::OnDevice {
                source: ModelSource::Registry {
                    toolchain_version: "1".into(),
                    repository: None,
                    source_repository: None,
                    files: vec![],
                },
            },
            Some(Value::from(serde_json::json!({ "type": "hanashi", "name": "qwen3.5" }))),
        )
    };
    let remote = Model {
        accessibility: ModelAccessibility::Remote {
            repository: None,
        },
        ..local(vec![ModelSpecialization::Chat {}], "uzu")
    };
    assert!(engine.chat_model_info(&remote).await.unwrap().is_none());
    assert!(
        engine.chat_model_info(&local(vec![ModelSpecialization::Classification {}], "uzu")).await.unwrap().is_none()
    );
    assert!(engine.chat_model_info(&local(vec![ModelSpecialization::Chat {}], "other")).await.unwrap().is_none());
    let mut unencoded = local(vec![ModelSpecialization::Chat {}], "uzu");
    unencoded.encoding = None;
    assert!(engine.chat_model_info(&unencoded).await.is_err());
}

#[tokio::test]
async fn a_stalled_registry_does_not_delay_a_healthy_registry() {
    let (engine, _directory) = engine().await;
    let (slow, slow_result, started) = registry("slow", None);
    let slow_ready = engine.start_registry(slow).await.unwrap();
    started.notified().await;

    let (healthy, healthy_result, _) = registry("healthy", None);
    healthy_result.send(Ok(vec![model("available")])).unwrap();
    tokio::time::timeout(Duration::from_secs(1), engine.add_registry(healthy)).await.unwrap().unwrap();
    assert_eq!(engine.models().await.unwrap(), vec![model("available")]);
    assert_eq!(engine.catalog_refreshes.load(Ordering::SeqCst), 1);

    slow_result
        .send(Err(RegistryError::UnableToGetModels {
            message: "offline".to_string(),
        }))
        .unwrap();
    // It may have signaled readiness from the healthy aggregate before its own error.
    let _ = tokio::time::timeout(Duration::from_secs(1), slow_ready).await.unwrap().unwrap();
    let mut events = engine.catalog_subscribe();
    while engine.catalog_refreshes.load(Ordering::SeqCst) != 0 {
        tokio::time::timeout(Duration::from_secs(1), events.next()).await.unwrap();
    }
    assert_eq!(engine.models().await.unwrap(), vec![model("available")]);
}

#[tokio::test]
async fn cached_models_and_lookups_remain_available_during_refresh() {
    let (engine, _directory) = engine().await;
    let (cached, result, started) = registry("cached", Some(vec![model("existing")]));
    let ready = engine.start_registry(cached).await.unwrap();
    started.notified().await;

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), engine.models()).await.unwrap().unwrap(),
        vec![model("existing")]
    );
    let found =
        tokio::time::timeout(Duration::from_secs(1), engine.model("existing".to_string())).await.unwrap().unwrap();
    assert_eq!(found, Some(model("existing")));

    result.send(Ok(vec![model("existing"), model("new")])).unwrap();
    ready.await.unwrap().unwrap();
    let mut events = engine.catalog_subscribe();
    while engine.catalog_refreshes.load(Ordering::SeqCst) != 0 {
        tokio::time::timeout(Duration::from_secs(1), events.next()).await.unwrap();
    }
    assert_eq!(engine.models().await.unwrap(), vec![model("existing"), model("new")]);
}

#[tokio::test]
async fn missing_model_lookup_waits_for_its_registry_and_finishes_when_refresh_ends() {
    for result in [
        Ok(vec![model("later")]),
        Ok(vec![]),
        Err(RegistryError::UnableToGetModels {
            message: "offline".to_string(),
        }),
    ] {
        let (engine, _directory) = engine().await;
        let (pending, sender, started) = registry("pending", None);
        let ready = engine.start_registry(pending).await.unwrap();
        started.notified().await;
        let lookup = tokio::spawn({
            let engine = engine.clone();
            async move { engine.model("later".to_string()).await }
        });
        tokio::task::yield_now().await;
        assert!(!lookup.is_finished());

        let expected = result.as_ref().ok().and_then(|models| models.first()).cloned();
        sender.send(result).unwrap();
        let found = tokio::time::timeout(Duration::from_secs(1), lookup).await.unwrap().unwrap().unwrap();
        assert_eq!(found, expected);
        let _ = ready.await.unwrap();
        assert_eq!(engine.catalog_refreshes.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn exact_lookups_wait_for_discovery_and_finish_after_empty_or_failed_refreshes() {
    for by_repo_id in [false, true] {
        for result in [
            Ok(vec![model_with_repo("later", "org/later")]),
            Ok(vec![]),
            Err(RegistryError::UnableToGetModels {
                message: "offline".to_string(),
            }),
        ] {
            let (engine, _directory) = engine().await;
            let (pending, sender, started) = registry("pending", None);
            let ready = engine.start_registry(pending).await.unwrap();
            started.notified().await;
            let lookup = tokio::spawn({
                let engine = engine.clone();
                async move {
                    if by_repo_id {
                        engine.model_by_repo_id("org/later".into()).await
                    } else {
                        engine.model_by_identifier("later".into()).await
                    }
                }
            });
            tokio::task::yield_now().await;
            assert!(!lookup.is_finished(), "exact lookup must wait for the pending registry");

            let expected = result.as_ref().ok().and_then(|models| models.first()).cloned();
            sender.send(result).unwrap();
            let found = tokio::time::timeout(Duration::from_secs(1), lookup).await.unwrap().unwrap().unwrap();
            assert_eq!(found, expected);
            let _ = ready.await.unwrap();
        }
    }
}

#[tokio::test]
async fn exact_lookups_return_cached_hits_without_confusing_identifiers_and_repo_ids() {
    let (engine, _directory) = engine().await;
    let models = vec![model_with_repo("first", "second"), model_with_repo("second", "third")];
    let (cached, result, started) = registry("cached", Some(models.clone()));
    let ready = engine.start_registry(cached).await.unwrap();
    started.notified().await;

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), engine.model_by_identifier("second".into()))
            .await
            .unwrap()
            .unwrap(),
        Some(models[1].clone()),
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), engine.model_by_repo_id("second".into())).await.unwrap().unwrap(),
        Some(models[0].clone()),
    );
    result.send(Ok(models)).unwrap();
    ready.await.unwrap().unwrap();
}

#[tokio::test]
async fn discovered_downloadable_models_have_storage_entries_when_returned() {
    let (engine, _directory) = engine().await;
    let (pending, sender, started) = registry("pending", Some(vec![model("existing")]));
    let ready = engine.start_registry(pending).await.unwrap();
    started.notified().await;
    ready.await.unwrap().unwrap();
    // A previous publication can hold this lock while the registry finishes.
    let publication = engine.catalog_publish.lock().await;
    let expected = Model {
        accessibility: ModelAccessibility::OnDevice {
            source: shoji::types::model::ModelSource::Registry {
                toolchain_version: "1".into(),
                repository: None,
                source_repository: None,
                files: vec![],
            },
        },
        ..model("downloadable")
    };
    sender.send(Ok(vec![model("existing"), expected.clone()])).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while engine.registry.lock().await.cached_listing().unwrap_or_default().0.len() != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), engine.model_by_identifier("existing".into()))
            .await
            .unwrap()
            .unwrap(),
        Some(model("existing")),
    );
    let lookup = tokio::spawn({
        let engine = engine.clone();
        async move { engine.model_by_identifier("downloadable".into()).await }
    });
    tokio::task::yield_now().await;
    assert!(!lookup.is_finished(), "lookup must wait for the model's storage registration");
    assert!(engine.catalog_is_refreshing(), "refresh remains active until publication completes");
    drop(publication);
    let found = tokio::time::timeout(Duration::from_secs(1), lookup).await.unwrap().unwrap().unwrap().unwrap();
    assert_eq!(found, expected);
    assert!(engine.download_state(&found).await.is_some(), "lookup returned a model before its storage entry existed");
    assert!(!engine.catalog_is_refreshing());
}

#[tokio::test]
async fn incompatible_downloaded_models_are_rejected_by_every_session_loader() {
    use shoji::types::{basic::File, model::ModelSource};

    let (engine, _directory) = engine().await;
    engine.add_backend(Arc::new(UzuLlmBackend::new())).await;
    let current = Engine::toolchain_version();
    let mut compatible = model("downloaded");
    compatible.backends[0].identifier = "uzu".into();
    compatible.backends[0].version = current.clone();
    compatible.accessibility = ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            toolchain_version: current.clone(),
            repository: Some(Repository {
                identifier: "test/downloaded".into(),
                commit_hash: Some("a".repeat(40)),
                paths: None,
            }),
            source_repository: None,
            files: vec![File {
                url: "https://example.invalid/model.bin".into(),
                name: "model.bin".into(),
                size: 6,
                hashes: vec![],
            }],
        },
    };
    let path = engine.storage.cache_model_path(&compatible).unwrap();
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("model.bin"), b"hello\n").unwrap();
    engine.storage.refresh(std::slice::from_ref(&compatible), false).await.unwrap();
    assert_eq!(engine.ready_download_state(&compatible).await.unwrap().phase, DownloadPhase::Downloaded {});
    assert_eq!(engine.model_path(&compatible).await, Some(path.to_string_lossy().into_owned()));

    // The checkpoint exists and is downloaded. Reject stale descriptors before
    // any backend or model files are opened, including descriptors held by callers.
    for (toolchain, backend) in [
        ("0.0.0", "0.0.0"),
        ("999.0.0", "999.0.0"),
        ("0.0.0", current.as_str()),
        ("999.0.0", current.as_str()),
        (current.as_str(), "0.0.0"),
        (current.as_str(), "999.0.0"),
    ] {
        let mut incompatible = compatible.clone();
        incompatible.backends[0].version = backend.into();
        if let ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version,
                ..
            },
        } = &mut incompatible.accessibility
        {
            *toolchain_version = toolchain.into();
        }
        assert!(engine.model_path(&incompatible).await.is_none());
        let expected = EngineError::ModelConfig {
            message: format!("Model downloaded is incompatible with uzu toolchain {current}"),
        };
        assert_eq!(engine.chat(incompatible.clone(), ChatConfig::default()).await.err(), Some(expected.clone()));
        assert_eq!(
            engine.chat_instance(incompatible.clone(), ChatConfig::default()).await.err(),
            Some(expected.clone())
        );
        assert_eq!(engine.decision(incompatible.clone()).await.err(), Some(expected.clone()));
        #[cfg(not(target_family = "wasm"))]
        {
            assert_eq!(engine.classification(incompatible.clone()).await.err(), Some(expected.clone()));
            assert_eq!(engine.text_to_speech(incompatible).await.err(), Some(expected));
        }
    }
}
