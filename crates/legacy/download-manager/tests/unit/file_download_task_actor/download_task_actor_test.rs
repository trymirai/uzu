use super::{remove_directory_if_empty, remove_owned_file};

#[tokio::test]
async fn unrelated_artifact_directory_contents_do_not_make_cleanup_fail() {
    let directory = tempfile::tempdir().unwrap();
    let artifact_root = directory.path().join("owned-artifacts");
    tokio::fs::create_dir(&artifact_root).await.unwrap();
    let unrelated = artifact_root.join("unrelated.txt");
    tokio::fs::write(&unrelated, b"keep").await.unwrap();

    remove_directory_if_empty(&artifact_root).await.unwrap();

    assert_eq!(tokio::fs::read(unrelated).await.unwrap(), b"keep");
}

#[cfg(unix)]
#[tokio::test]
async fn owned_file_cleanup_rejects_a_late_root_symlink() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let owned_root = directory.path().join("owned");
    let moved_root = directory.path().join("owned-before-symlink");
    tokio::fs::create_dir(&owned_root).await.unwrap();
    tokio::fs::rename(&owned_root, moved_root).await.unwrap();
    symlink(outside.path(), &owned_root).unwrap();
    let outside_file = outside.path().join("download.part");
    tokio::fs::write(&outside_file, b"keep").await.unwrap();

    remove_owned_file(&owned_root.join("download.part"), Some(&owned_root), "test artifact").await;

    assert_eq!(tokio::fs::read(outside_file).await.unwrap(), b"keep");
}
