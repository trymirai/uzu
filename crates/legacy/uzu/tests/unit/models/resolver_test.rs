use std::{error::Error, path::PathBuf};

use serde_json::{Value, from_value, json, to_value};
use shoji::types::{
    basic::{File, Hash, HashMethod, Repository},
    model::{Model, ModelAccessibility, ModelSource},
};

use super::*;
use crate::api::HuggingFaceFileResponse;

const REVISION: &str = "5ed3b433df2e390ace215873283927856573d9e6";

fn repository() -> Repository {
    Repository {
        identifier: "trymirai/model".to_string(),
        commit_hash: Some(REVISION.to_string()),
        paths: None,
    }
}

fn pinned_model(revision: &str) -> Model {
    Model::external(
        "model".to_string(),
        "registry".to_string(),
        "Registry".to_string(),
        "backend".to_string(),
        "Backend".to_string(),
        "1".to_string(),
        Vec::new(),
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".to_string(),
                repository: Some(Repository {
                    identifier: "trymirai/model".to_string(),
                    commit_hash: Some(revision.to_string()),
                    paths: None,
                }),
                source_repository: None,
                files: Vec::new(),
            },
        },
        None,
    )
}

fn git_file(
    name: &str,
    blob_id: Option<&str>,
) -> HuggingFaceFileResponse {
    hugging_face_file(json!({ "rfilename": name, "size": 6, "blobId": blob_id }))
}

fn hugging_face_file(value: Value) -> HuggingFaceFileResponse {
    from_value(value).expect("Hugging Face file should deserialize")
}

#[test]
fn hugging_face_files_use_git_and_lfs_digests() -> Result<(), Box<dyn Error>> {
    let resolver = ModelsResolver::new(None, PathBuf::new())?;
    let files = resolver.resolve_files(
        &repository(),
        REVISION,
        vec![
            git_file("config.json", Some("ce013625030ba8dba906f756967f9e9ca394464a")),
            hugging_face_file(json!({
                "rfilename": "model.safetensors",
                "size": null,
                "blobId": null,
                "lfs": {
                    "sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
                    "size": 6
                }
            })),
        ],
        false,
    )?;

    assert!(matches!(files[0].check, FileCheck::GitBlobSha1(_)));
    assert!(matches!(files[1].check, FileCheck::Sha256(_)));
    assert!(files.iter().all(|file| file.file.url.contains(REVISION)));
    Ok(())
}

#[test]
fn invalid_hugging_face_file_metadata_is_rejected() -> Result<(), Box<dyn Error>> {
    let resolver = ModelsResolver::new(None, PathBuf::new())?;

    let missing_digest = resolver.resolve_files(&repository(), REVISION, vec![git_file("config.json", None)], false);
    let unsafe_path = resolver.resolve_files(
        &repository(),
        REVISION,
        vec![git_file("../config.json", Some("ce013625030ba8dba906f756967f9e9ca394464a"))],
        false,
    );

    assert!(missing_digest.is_err());
    assert!(unsafe_path.is_err());
    Ok(())
}

#[test]
fn persisted_files_are_reused_only_for_the_exact_registry_source() -> Result<(), Box<dyn Error>> {
    let model = pinned_model(REVISION);
    let file = ResolvedFile {
        file: File {
            url: format!("https://huggingface.co/trymirai/model/resolve/{REVISION}/config.json"),
            name: "config.json".to_string(),
            size: 6,
            hashes: Vec::new(),
        },
        check: FileCheck::GitBlobSha1("ce013625030ba8dba906f756967f9e9ca394464a".to_string()),
        requires_authentication: false,
    };
    let saved = to_value(ResolvedModels::new(vec![ResolvedModel::downloadable(model.clone(), vec![file.clone()])]))?;
    let cached = from_value::<ResolvedModels>(saved)?.validate_cache().expect("cache should be valid");

    assert!(cached.reusable_hugging_face_files(&model) == Some(vec![file]));
    assert!(cached.reusable_hugging_face_files(&pinned_model("6ed3b433df2e390ace215873283927856573d9e6")).is_none());
    Ok(())
}

#[tokio::test]
async fn unpinned_mirai_files_keep_crc_verification() -> Result<(), Box<dyn Error>> {
    let resolver = ModelsResolver::new(None, PathBuf::new())?;
    let model = Model::external(
        "model".to_string(),
        "registry".to_string(),
        "Registry".to_string(),
        "backend".to_string(),
        "Backend".to_string(),
        "1".to_string(),
        Vec::new(),
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".to_string(),
                repository: None,
                source_repository: None,
                files: vec![File {
                    url: String::new(),
                    name: "model".to_string(),
                    size: 1,
                    hashes: vec![Hash {
                        method: HashMethod::CRC32C,
                        value: "AAAAAA==".to_string(),
                    }],
                }],
            },
        },
        None,
    );

    let resolved = resolver.resolve(vec![model], &ResolvedModels::default()).await?;
    let (_, files) = resolved.iter().next().expect("model should be resolved").parts();

    assert!(matches!(files.expect("model should be downloadable")[0].check, FileCheck::CRC(_)));
    Ok(())
}
