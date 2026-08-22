use std::path::Path;

use crate::{FileCheck, FileDownloadGroupSpec, FileDownloadGroupSpecError, FileDownloadRequest, RelativeFilePath};

fn request(path: &str) -> FileDownloadRequest {
    FileDownloadRequest::new(
        "https://example.com/file",
        RelativeFilePath::try_from(path).unwrap(),
        FileCheck::None,
        None,
    )
}

#[test]
fn rejects_empty_group() {
    assert_eq!(FileDownloadGroupSpec::new("/cache/model", []), Err(FileDownloadGroupSpecError::Empty));
}

#[test]
fn rejects_empty_destination_root() {
    assert_eq!(
        FileDownloadGroupSpec::new("", [request("weights.bin")]),
        Err(FileDownloadGroupSpecError::EmptyDestinationRoot)
    );
}

#[test]
fn rejects_duplicate_destinations() {
    let duplicate = RelativeFilePath::try_from("weights.bin").unwrap();

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [request("weights.bin"), request("weights.bin")]),
        Err(FileDownloadGroupSpecError::DuplicateDestination(duplicate))
    );
}

#[test]
fn rejects_destinations_that_only_differ_by_case() {
    let duplicate = RelativeFilePath::try_from("weights.bin").unwrap();

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [request("Weights.BIN"), request("weights.bin")]),
        Err(FileDownloadGroupSpecError::DuplicateDestination(duplicate))
    );
}

#[test]
fn rejects_canonically_equivalent_unicode_destinations() {
    let duplicate = RelativeFilePath::try_from("e\u{301}.bin").unwrap();

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [request("é.bin"), request("e\u{301}.bin")]),
        Err(FileDownloadGroupSpecError::DuplicateDestination(duplicate))
    );
}

#[test]
fn joins_validated_path_under_destination_root() {
    let spec = FileDownloadGroupSpec::new("/cache/model", [request("nested/weights.bin")]).unwrap();

    assert_eq!(spec.destination_for(&spec.files()[0]), Path::new("/cache/model/nested/weights.bin"));
}

#[test]
fn rejects_file_and_directory_path_conflict() {
    let ancestor = RelativeFilePath::try_from("weights").unwrap();
    let descendant = RelativeFilePath::try_from("weights/model.bin").unwrap();

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [request("weights/model.bin"), request("weights")]),
        Err(FileDownloadGroupSpecError::ConflictingDestinations {
            ancestor,
            descendant,
        })
    );
}

#[test]
fn rejects_file_and_directory_conflict_that_only_differs_by_case() {
    let ancestor = RelativeFilePath::try_from("Weights").unwrap();
    let descendant = RelativeFilePath::try_from("weights/model.bin").unwrap();

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [request("weights/model.bin"), request("Weights")]),
        Err(FileDownloadGroupSpecError::ConflictingDestinations {
            ancestor,
            descendant,
        })
    );
}

#[test]
fn rejects_expected_total_overflow() {
    let mut first = request("first.bin");
    first.expected_bytes = Some(u64::MAX);
    let mut second = request("second.bin");
    second.expected_bytes = Some(1);

    assert_eq!(
        FileDownloadGroupSpec::new("/cache/model", [first, second]),
        Err(FileDownloadGroupSpecError::TotalBytesOverflow)
    );
}

#[test]
fn normalizes_file_order() {
    let first = FileDownloadGroupSpec::new(
        "/cache/model",
        [request("nested/z.bin"), request("a.bin"), request("nested/b.bin")],
    )
    .unwrap();
    let second = FileDownloadGroupSpec::new(
        "/cache/model",
        [request("nested/b.bin"), request("nested/z.bin"), request("a.bin")],
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.files().iter().map(|file| file.relative_path.to_string()).collect::<Vec<_>>(),
        ["a.bin", "nested/b.bin", "nested/z.bin"]
    );
}
