use crate::common::scenarios::{ManagerKind, run_pause_resume_scenario};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_pause_resume_preserves_progress() {
    run_pause_resume_scenario(ManagerKind::Apple).await;
}
