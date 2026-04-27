use download_manager::{FileDownloadManagerType, create_download_manager, managers::apple::URLSessionDownloadTaskResumeData};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_pause_writes_parseable_resume_data() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::StallAt {
            byte_offset: 128 * 1024,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-resume-data".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("failed to start download");
    context.wait_for_bytes(128 * 1024).await;
    task.pause().await.expect("failed to pause download");
    let paused_state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Paused).await;
    assert!(paused_state.downloaded_bytes > 0, "pause should report downloaded bytes");

    let resume_data = URLSessionDownloadTaskResumeData::from_file(context.resume_data_path())
        .expect("resume data should parse through URLSessionDownloadTaskResumeData");
    let bytes_received = resume_data.bytes_received.expect("resume data should include bytes_received");
    assert!(bytes_received > 0, "resume data should preserve positive progress");
    assert!(
        resume_data.original_request.is_some() || resume_data.current_request.is_some(),
        "resume data should retain request information"
    );
}
