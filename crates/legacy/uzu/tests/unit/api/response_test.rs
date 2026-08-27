use serde_json::{from_value, json};

use super::*;

#[test]
fn fetched_models_converts_one_complete_response() {
    let response: FetchedModels = from_value(json!({
        "metadatas": [
            { "id": "registry-meta", "name": "Mirai", "description": null, "icons": [] },
            { "id": "backend-meta", "name": "Uzu", "description": null, "icons": [] }
        ],
        "models": [{
            "id": "model",
            "registry": { "id": "mirai", "metadata_id": "registry-meta" },
            "backends": [{ "id": "uzu", "version": "1", "metadata_id": "backend-meta" }],
            "family": null,
            "properties": null,
            "quantization": null,
            "specializations": [],
            "accessibility": {
                "type": "local",
                "reference": {
                    "type": "mirai",
                    "toolchain_version": "1",
                    "repository": null,
                    "source_repository": null,
                    "files": []
                }
            },
            "encodings": []
        }]
    }))
    .expect("response should deserialize");

    let models = response.models().expect("response should resolve metadata references");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].identifier, "model");
    assert!(matches!(
        &models[0].accessibility,
        ModelAccessibility::OnDevice {
            source: ModelSource::Managed {
                toolchain_version,
                repository: None,
                source_repository: None,
                files,
            },
        } if toolchain_version == "1" && files.is_empty()
    ));

    let filesystem: FetchedAccessibility = from_value(json!({
        "type": "local",
        "reference": { "type": "local", "path": "/models/local" }
    }))
    .expect("filesystem response should deserialize");
    assert_eq!(
        ModelAccessibility::from(&filesystem),
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path: "/models/local".to_string(),
            },
        }
    );
}

#[test]
fn hugging_face_access_is_conservative() {
    for (gated, requires_authentication) in [(json!(false), false), (json!("manual"), true), (json!(null), true)] {
        let response: HuggingFaceModelResponse = from_value(json!({
            "sha": "",
            "private": false,
            "gated": gated,
            "siblings": []
        }))
        .expect("response should deserialize");

        assert_eq!(response.requires_authentication(), requires_authentication);
    }
}

#[test]
fn fetched_models_rejects_missing_metadata() {
    let response: FetchedModels = from_value(json!({
        "metadatas": [],
        "models": [{
            "id": "model",
            "registry": { "id": "mirai", "metadata_id": "missing" },
            "backends": [],
            "family": null,
            "properties": null,
            "quantization": null,
            "specializations": [],
            "accessibility": { "type": "remote", "repository": null },
            "encodings": []
        }]
    }))
    .expect("response should deserialize");

    assert!(response.models().is_err());
}
