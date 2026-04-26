use crate::common::scenarios::{ManagerKind, run_cancel_redownload_scenario};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_cancel_removes_resume_data() {
    run_cancel_redownload_scenario(ManagerKind::Apple).await;
}
