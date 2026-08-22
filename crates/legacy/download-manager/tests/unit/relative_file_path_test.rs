use std::path::PathBuf;

use super::{RelativeFilePath, RelativeFilePathError};

#[test]
fn accepts_nested_relative_file_path() {
    let path = RelativeFilePath::try_from("weights/model.safetensors").unwrap();

    assert_eq!(path.as_path(), PathBuf::from("weights/model.safetensors"));
}

#[test]
fn accepts_download_manager_named_repository_directory() {
    let path = RelativeFilePath::try_from(".uzu-download-manager/metadata.json").unwrap();

    assert_eq!(path.as_path(), PathBuf::from(".uzu-download-manager/metadata.json"));
}

#[test]
fn rejects_empty_path() {
    assert_eq!(RelativeFilePath::try_from(""), Err(RelativeFilePathError::Empty));
}

#[test]
fn rejects_absolute_path() {
    assert!(matches!(RelativeFilePath::try_from("/tmp/model.bin"), Err(RelativeFilePathError::Invalid(_))));
}

#[test]
fn rejects_parent_and_current_directory_components() {
    for path in ["../model.bin", "weights/../model.bin", "./model.bin", "weights/./model.bin"] {
        assert!(matches!(RelativeFilePath::try_from(path), Err(RelativeFilePathError::Invalid(_))));
    }
}

#[test]
fn rejects_cross_platform_separators_colons_and_control_characters() {
    for path in
        ["C:\\model.bin", "C:/model.bin", "weights\\model.bin", "weights.bin:stream", "bad\0name", "bad\u{1f}name"]
    {
        assert!(matches!(RelativeFilePath::try_from(path), Err(RelativeFilePathError::Invalid(_))));
    }
}

#[test]
fn rejects_windows_unsafe_characters_and_normalized_endings() {
    for path in [
        "bad<name",
        "bad>name",
        "bad\"name",
        "bad|name",
        "bad?name",
        "bad*name",
        "trailing-dot.",
        "trailing-space ",
        "nested/trailing. ",
    ] {
        assert!(matches!(RelativeFilePath::try_from(path), Err(RelativeFilePathError::Invalid(_))));
    }
}

#[test]
fn rejects_windows_device_names_with_extensions() {
    for path in ["CON", "nul.bin", "nested/COM1.safetensors", "Lpt9", "COM¹.bin", "CONOUT$"] {
        assert!(matches!(RelativeFilePath::try_from(path), Err(RelativeFilePathError::Invalid(_))));
    }

    for path in ["console.bin", "com10.bin", "lpt-model.bin"] {
        assert!(RelativeFilePath::try_from(path).is_ok());
    }
}

#[test]
fn deserialize_revalidates_path() {
    let error = serde_json::from_str::<RelativeFilePath>(r#""../model.bin""#).unwrap_err();

    assert!(error.to_string().contains("portable safe relative path"));
}
