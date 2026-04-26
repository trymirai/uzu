use crate::common::scenarios::{ManagerKind, run_fresh_download_scenario};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_fresh_download_completes() {
    run_fresh_download_scenario(ManagerKind::Apple).await;
}
