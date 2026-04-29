use download_manager::{FileDownloadManagerType, create_download_manager};

#[tokio::test]
async fn test_create_download_manager_universal_returns_manager() {
    let manager =
        create_download_manager(FileDownloadManagerType::Universal, tokio::runtime::Handle::current()).await.unwrap();

    assert_eq!(manager.manager_id(), "download-manager-universal");
}

#[cfg(target_vendor = "apple")]
#[tokio::test]
async fn test_create_download_manager_apple_returns_manager_on_apple() {
    let manager =
        create_download_manager(FileDownloadManagerType::Apple, tokio::runtime::Handle::current()).await.unwrap();

    assert_eq!(manager.manager_id(), "download-manager-apple");
}
