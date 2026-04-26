use crate::common::scenarios::{ManagerKind, run_cancel_redownload_scenario};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_cancel_removes_partial_state() {
    run_cancel_redownload_scenario(ManagerKind::Universal).await;
}
