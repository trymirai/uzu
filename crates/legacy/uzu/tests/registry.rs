#![cfg(not(target_family = "wasm"))]

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use serde_json::{Value, json};
use uzu::{
    device::Device,
    registry::{
        CachedRegistry, MergedRegistry, RegistryError,
        mirai::{Backend, HuggingFace, Registry},
    },
    traits::Registry as RegistryTrait,
    types::{
        basic::{File, Hash, HashMethod, Repository},
        model::{Model, ModelAccessibility, ModelSource},
    },
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path, query_param},
};

const REVISION: &str = "f5dec40ceb1d8c4b15f049bf5e185dd7be2cc150";
const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
const HELLO_GIT_BLOB_SHA1: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

fn repository(
    revision: &str,
    paths: Option<Vec<String>>,
) -> Repository {
    Repository {
        identifier: "trymirai/model".to_string(),
        commit_hash: Some(revision.to_string()),
        paths,
    }
}

fn siblings() -> Value {
    json!([
        { "rfilename": "config.json", "size": 6, "blobId": HELLO_GIT_BLOB_SHA1, "lfs": null },
        {
            "rfilename": "model.safetensors",
            "size": 6,
            "blobId": "0000000000000000000000000000000000000000",
            "lfs": { "sha256": HELLO_SHA256, "size": 6, "pointerSize": 134 }
        }
    ])
}

async fn mount_hugging_face(
    server: &MockServer,
    revision: &str,
    expected_calls: u64,
) {
    Mock::given(method("GET"))
        .and(path(format!("/api/models/trymirai/model/revision/{revision}")))
        .and(query_param("blobs", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "sha": REVISION,
            "private": false,
            "gated": false,
            "siblings": siblings(),
        })))
        .expect(expected_calls)
        .mount(server)
        .await;
}

fn listed_model(
    identifier: &str,
    reference: Value,
) -> Value {
    json!({
        "id": identifier,
        "registry": { "id": "mirai", "metadata_id": "registry-meta" },
        "backends": [{ "id": "uzu", "version": "1", "metadata_id": "backend-meta" }],
        "family": null,
        "properties": null,
        "quantization": null,
        "specializations": [],
        "accessibility": { "type": "local", "reference": reference },
        "encodings": []
    })
}

fn registry_response(models: Vec<Value>) -> Value {
    json!({
        "metadatas": [
            { "id": "registry-meta", "name": "Mirai", "description": null, "icons": [] },
            { "id": "backend-meta", "name": "Uzu", "description": null, "icons": [] }
        ],
        "models": models,
    })
}

fn cached_model(identifier: &str) -> Model {
    Model::external(
        identifier.to_string(),
        "mirai".to_string(),
        "Mirai".to_string(),
        "uzu".to_string(),
        "Uzu".to_string(),
        "1".to_string(),
        vec![],
        ModelAccessibility::Remote {
            repository: None,
        },
        None,
    )
}

struct BlockingRegistry {
    started: Arc<tokio::sync::Notify>,
    finish: Arc<tokio::sync::Notify>,
    snapshot: Option<Vec<Model>>,
    result: Result<Vec<Model>, RegistryError>,
}

impl RegistryTrait for BlockingRegistry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        "blocking".to_string()
    }

    fn cached_listing(&self) -> Option<(Vec<Model>, bool)> {
        self.snapshot.clone().map(|models| (models, true))
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            self.started.notify_one();
            self.finish.notified().await;
            self.result.clone()
        })
    }
}

#[tokio::test]
async fn snapshots_stay_readable_during_refresh_and_survive_failure() {
    let started = Arc::new(tokio::sync::Notify::new());
    let finish = Arc::new(tokio::sync::Notify::new());
    let registry = Arc::new(CachedRegistry::new(Box::new(BlockingRegistry {
        started: started.clone(),
        finish: finish.clone(),
        snapshot: Some(vec![cached_model("cached")]),
        result: Err(RegistryError::UnableToGetModels {
            message: "offline".to_string(),
        }),
    })));
    let mut merged = MergedRegistry::new(vec![]);
    merged.add(registry.clone()).unwrap();
    let refresh = tokio::spawn({
        let registry = registry.clone();
        async move { registry.refresh_listing(Arc::new(|| {})).await }
    });
    started.notified().await;

    let (models, complete) = tokio::time::timeout(Duration::from_secs(1), merged.listing()).await.unwrap().unwrap();
    assert_eq!(models[0].identifier, "cached");
    assert!(!complete);
    finish.notify_one();
    assert!(refresh.await.unwrap().is_err());
    assert_eq!(registry.cached_listing().unwrap().0[0].identifier, "cached");
}

#[tokio::test]
async fn an_empty_completed_snapshot_is_distinct_from_an_unloaded_registry() {
    let finish = Arc::new(tokio::sync::Notify::new());
    let registry = CachedRegistry::new(Box::new(BlockingRegistry {
        started: Arc::new(tokio::sync::Notify::new()),
        finish: finish.clone(),
        snapshot: None,
        result: Ok(vec![]),
    }));
    assert!(registry.cached_listing().is_none());

    finish.notify_one();
    registry.refresh_listing(Arc::new(|| {})).await.unwrap();

    assert_eq!(registry.cached_listing(), Some((vec![], true)));
}

#[tokio::test]
async fn mirai_publishes_healthy_models_before_a_stalled_resolution() -> Result<(), Box<dyn std::error::Error>> {
    let hugging_face = MockServer::start().await;
    for (name, delay) in [("slow", Duration::from_secs(1)), ("fast", Duration::ZERO)] {
        Mock::given(method("GET"))
            .and(path(format!("/api/models/trymirai/{name}/revision/{REVISION}")))
            .respond_with(ResponseTemplate::new(200).set_delay(delay).set_body_json(json!({
                "sha": REVISION, "siblings": siblings(),
            })))
            .mount(&hugging_face)
            .await;
    }
    let mirai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/fetch/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(registry_response(vec![
            listed_model("slow", pinned("trymirai/slow", REVISION)),
            listed_model("fast", pinned("trymirai/fast", REVISION)),
        ])))
        .mount(&mirai)
        .await;
    let directory = tempfile::tempdir()?;
    let registry = Arc::new(CachedRegistry::new(Box::new(
        Registry::builder()
            .device(Device::new()?)
            .backends(vec![])
            .cache_path(directory.path().to_path_buf())
            .registry_url(mirai.uri())
            .hugging_face_url(hugging_face.uri())
            .build()?,
    )));
    assert!(registry.cached_listing().is_none());
    let (updates, mut received) = tokio::sync::mpsc::unbounded_channel();
    let refresh = tokio::spawn({
        let registry = registry.clone();
        let snapshot = registry.clone();
        async move {
            registry
                .refresh_listing(Arc::new(move || {
                    let _ = updates.send(snapshot.cached_listing().unwrap());
                }))
                .await
        }
    });
    let first_nonempty = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let listing = received.recv().await.expect("refresh ended without a snapshot");
            if !listing.0.is_empty() {
                break listing;
            }
        }
    })
    .await?;
    assert_eq!(first_nonempty.0.iter().map(|model| model.identifier.as_str()).collect::<Vec<_>>(), ["fast"]);
    assert!(!first_nonempty.1);
    let (models, complete) = refresh.await??;
    assert!(complete);
    assert_eq!(models.iter().map(|model| model.identifier.as_str()).collect::<Vec<_>>(), ["slow", "fast"]);
    Ok(())
}

#[tokio::test]
async fn mirai_serves_disk_cache_and_keeps_it_when_new_metadata_fails() -> Result<(), Box<dyn std::error::Error>> {
    let mirai = MockServer::start().await;
    let hugging_face = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/fetch/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(registry_response(vec![listed_model("cached", pinned("trymirai/missing", REVISION))])),
        )
        .expect(1)
        .mount(&mirai)
        .await;
    let directory = tempfile::tempdir()?;
    let mut previous = cached_model("cached");
    previous.accessibility = ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            toolchain_version: "1".to_string(),
            repository: Some(repository(&"e".repeat(40), None)),
            source_repository: None,
            files: vec![File {
                url: format!("{}/trymirai/model/resolve/{}/config.json", hugging_face.uri(), "e".repeat(40)),
                name: "config.json".to_string(),
                size: 6,
                hashes: vec![],
            }],
        },
    };
    std::fs::write(directory.path().join("registry.json"), serde_json::to_vec(&vec![previous.clone()])?)?;
    let registry = Registry::builder()
        .device(Device::new()?)
        .backends(vec![])
        .cache_path(directory.path().to_path_buf())
        .registry_url(mirai.uri())
        .hugging_face_url(hugging_face.uri())
        .build()?;

    assert_eq!(registry.models().await?, vec![previous.clone()]);
    assert!(mirai.received_requests().await.unwrap().is_empty());
    let (models, complete) = registry.refresh_listing(Arc::new(|| {})).await?;
    assert_eq!(models, vec![previous]);
    assert!(!complete);
    Ok(())
}

#[tokio::test]
async fn hugging_face_files() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    mount_hugging_face(&server, REVISION, 2).await;
    mount_hugging_face(&server, "main", 1).await;
    let hugging_face = HuggingFace::builder().endpoint(server.uri()).build()?;

    let files = hugging_face.files(&repository(REVISION, None)).await?;
    assert_eq!(files.iter().map(|file| file.name.as_str()).collect::<Vec<_>>(), ["config.json", "model.safetensors"]);
    assert_eq!(files[0].url, format!("{}/trymirai/model/resolve/{REVISION}/config.json", server.uri()));
    assert_eq!(files[0].hashes[0].method, HashMethod::GitBlobSha1);
    assert_eq!(files[0].hashes[0].value, HELLO_GIT_BLOB_SHA1);
    assert_eq!(files[1].hashes[0].method, HashMethod::Sha256);
    assert_eq!(files[1].hashes[0].value, HELLO_SHA256);
    assert!(files.iter().all(|file| file.size == 6));

    let filtered = hugging_face.files(&repository(REVISION, Some(vec!["config.json".to_string()]))).await?;
    assert_eq!(filtered.len(), 1);

    let tagged = hugging_face.files(&repository("main", None)).await?;
    assert_eq!(tagged, files);

    let unpinned = hugging_face
        .files(&Repository {
            commit_hash: None,
            ..repository(REVISION, None)
        })
        .await;
    assert!(unpinned.is_err());
    Ok(())
}

#[tokio::test]
async fn hugging_face_rejects_bad_metadata() -> Result<(), Box<dyn std::error::Error>> {
    for body in [
        json!({ "sha": "main", "siblings": siblings() }),
        json!({ "sha": REVISION, "siblings": [{ "rfilename": "../config.json", "size": 6, "blobId": HELLO_GIT_BLOB_SHA1 }] }),
        json!({ "sha": REVISION, "siblings": [{ "rfilename": "config.json", "blobId": HELLO_GIT_BLOB_SHA1 }] }),
        json!({ "sha": REVISION, "siblings": [] }),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_json(body)).mount(&server).await;
        let hugging_face = HuggingFace::builder().endpoint(server.uri()).build()?;
        assert!(hugging_face.files(&repository(REVISION, None)).await.is_err());
    }
    Ok(())
}

#[tokio::test]
async fn hugging_face_accepts_files_without_digests() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "sha": REVISION,
            "siblings": [
                { "rfilename": "config.json", "size": 6, "blobId": null },
                { "rfilename": "model.safetensors", "lfs": { "size": 20 } }
            ],
        })))
        .mount(&server)
        .await;
    let hugging_face = HuggingFace::builder().endpoint(server.uri()).build()?;

    let files = hugging_face.files(&repository(REVISION, None)).await?;

    assert_eq!(files.len(), 2);
    assert_eq!(files.iter().map(|file| file.size).collect::<Vec<_>>(), [6, 20]);
    assert!(files.iter().all(|file| file.hashes.is_empty()));
    Ok(())
}

fn pinned(
    identifier: &str,
    revision: &str,
) -> Value {
    json!({
        "type": "mirai",
        "toolchain_version": "1",
        "repository": { "identifier": identifier, "commit_hash": revision, "paths": null },
        "source_repository": null,
        "files": []
    })
}

#[tokio::test]
async fn mirai_registry_resolves_pinned_models_once() -> Result<(), Box<dyn std::error::Error>> {
    let hugging_face = MockServer::start().await;
    mount_hugging_face(&hugging_face, REVISION, 1).await;
    mount_hugging_face(&hugging_face, "main", 2).await;
    let mirai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/fetch/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "metadatas": [
                { "id": "registry-meta", "name": "Mirai", "description": null, "icons": [] },
                { "id": "backend-meta", "name": "Uzu", "description": null, "icons": [] }
            ],
            "models": [
                listed_model("pinned", pinned("trymirai/model", REVISION)),
                listed_model("tagged", pinned("trymirai/model", "main")),
                listed_model("unpinned", json!({
                    "type": "mirai",
                    "toolchain_version": "1",
                    "repository": null,
                    "source_repository": null,
                    "files": [{
                        "url": "https://assets.example/config.json",
                        "name": "config.json",
                        "size": 6,
                        "hashes": [{ "method": "crc32c", "value": "AAAAAA==" }]
                    }]
                })),
                listed_model("unresolvable", pinned("trymirai/missing", REVISION)),
            ]
        })))
        .mount(&mirai)
        .await;
    let directory = tempfile::tempdir()?;
    // A cache left by an older engine points the pinned files at the Mirai CDN; it must be re-resolved, not reused.
    let stale = Model::external(
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
                repository: Some(repository(REVISION, None)),
                source_repository: None,
                files: vec![File {
                    url: "https://assets.example/model.safetensors".to_string(),
                    name: "model.safetensors".to_string(),
                    size: 6,
                    hashes: vec![Hash {
                        method: HashMethod::CRC32C,
                        value: "AAAAAA==".to_string(),
                    }],
                }],
            },
        },
        None,
    );
    std::fs::write(directory.path().join("registry.json"), serde_json::to_vec(&vec![stale])?)?;

    for _ in 0..2 {
        let registry = Registry::builder()
            .device(Device::new()?)
            .backends(vec![Backend {
                identifier: "uzu".to_string(),
                version: "1".to_string(),
            }])
            .cache_path(directory.path().to_path_buf())
            .registry_url(mirai.uri())
            .hugging_face_url(hugging_face.uri())
            .build()?;
        let (models, complete) = registry.refresh_listing(Arc::new(|| {})).await?;
        assert!(!complete);
        assert_eq!(
            models.iter().map(|model| model.identifier.as_str()).collect::<Vec<_>>(),
            ["pinned", "tagged", "unpinned"]
        );
        for model in &models[..2] {
            let ModelAccessibility::OnDevice {
                source:
                    ModelSource::Registry {
                        repository: Some(repository),
                        files,
                        ..
                    },
            } = &model.accessibility
            else {
                panic!("{} must stay on device", model.identifier);
            };
            assert_eq!(files.len(), 2);
            assert_eq!(files[1].hashes[0].method, HashMethod::Sha256);
            assert!(files.iter().all(|file| file.url.contains(REVISION)));
            assert_eq!(model.checkpoint_version(), repository.commit_hash);
        }
    }
    Ok(())
}
