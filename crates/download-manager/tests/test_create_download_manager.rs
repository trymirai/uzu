use download_manager::{FileDownloadManager, FileDownloadManagerType};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_file_download_manager_new_returns_manager(#[case] download_manager_type: FileDownloadManagerType) {
    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let expected_manager_id = match download_manager_type {
        FileDownloadManagerType::Universal => "mirai.universal",
        FileDownloadManagerType::Apple => "mirai.apple",
    };
    assert!(manager.manager_id().ends_with(expected_manager_id));
}
