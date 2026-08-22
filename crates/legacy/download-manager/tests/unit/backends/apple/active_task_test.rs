use super::write_resume_data;

#[cfg(unix)]
#[tokio::test]
async fn authenticated_resume_reset_rejects_a_symlinked_file() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let resume_path = directory.path().join("download.resume_data");
    let outside_file = outside.path().join("keep");
    tokio::fs::write(&outside_file, b"keep").await.unwrap();
    symlink(&outside_file, &resume_path).unwrap();

    write_resume_data(&resume_path, &[]).await.expect_err("resume-data symlinks must be rejected");

    assert_eq!(tokio::fs::read(outside_file).await.unwrap(), b"keep");
}
