use download_manager::FileDownloadManagerType;
use rstest::rstest;

use crate::common::scenarios::{
    run_cancel_redownload_scenario, run_fresh_download_scenario, run_pause_resume_scenario,
};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_fresh_completes(#[case] download_manager_type: FileDownloadManagerType) {
    run_fresh_download_scenario(download_manager_type).await;
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_pause_resume_preserves_progress(#[case] download_manager_type: FileDownloadManagerType) {
    run_pause_resume_scenario(download_manager_type).await;
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_cancel_redownload_cleans_partial_state(#[case] download_manager_type: FileDownloadManagerType) {
    run_cancel_redownload_scenario(download_manager_type).await;
}
