use std::path::Path;

use super::compute_download_id;

#[test]
fn lexical_aliases_have_one_download_id() {
    assert_eq!(
        compute_download_id(Path::new("models/a/../model.bin")),
        compute_download_id(Path::new("models/model.bin"))
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn case_and_unicode_aliases_have_one_download_id() {
    let directory = tempfile::tempdir().unwrap();

    assert_eq!(
        compute_download_id(&directory.path().join("Weights.bin")),
        compute_download_id(&directory.path().join("weights.bin")),
    );
    assert_eq!(
        compute_download_id(&directory.path().join("é.bin")),
        compute_download_id(&directory.path().join("e\u{301}.bin")),
    );
}
