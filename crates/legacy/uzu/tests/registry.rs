#![cfg(not(target_family = "wasm"))]

use serde_json::{Value, json};
use uzu::{
    device::Device,
    registry::mirai::{Backend, HuggingFace, Registry},
    traits::Registry as RegistryTrait,
    types::{
        basic::{HashMethod, Repository},
        model::{ModelAccessibility, ModelSource},
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
        json!({ "sha": REVISION, "siblings": [{ "rfilename": "config.json", "size": 6, "blobId": null }] }),
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
async fn mirai_registry_resolves_pinned_models_once() -> Result<(), Box<dyn std::error::Error>> {
    let hugging_face = MockServer::start().await;
    mount_hugging_face(&hugging_face, REVISION, 1).await;
    mount_hugging_face(&hugging_face, "main", 2).await;
    let mirai = MockServer::start().await;
    let pinned = |identifier: &str, revision: &str| {
        json!({
            "type": "mirai",
            "toolchain_version": "1",
            "repository": { "identifier": identifier, "commit_hash": revision, "paths": null },
            "source_repository": null,
            "files": []
        })
    };
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
        let models = registry.models().await?;
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
