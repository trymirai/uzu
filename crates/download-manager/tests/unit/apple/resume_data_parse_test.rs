use download_manager::managers::apple::URLSessionDownloadTaskResumeData;

use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_pause_writes_parseable_resume_data() {
    let scenario = DownloadScenario::new(
        ManagerKind::Apple,
        MockFile::large_tokenizer("download-manager/apple-resume-data"),
        RouteBehavior::StallAt {
            byte_offset: 128 * 1024,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_bytes(128 * 1024).await;
    let paused_state = scenario.pause_download().await;
    assert!(paused_state.downloaded_bytes > 0, "pause should report downloaded bytes");

    let resume_data = URLSessionDownloadTaskResumeData::from_file(scenario.resume_data_path())
        .expect("resume data should parse through URLSessionDownloadTaskResumeData");
    let bytes_received = resume_data.bytes_received.expect("resume data should include bytes_received");
    assert!(bytes_received > 0, "resume data should preserve positive progress");
    assert!(
        resume_data.original_request.is_some() || resume_data.current_request.is_some(),
        "resume data should retain request information"
    );
}
