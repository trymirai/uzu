#![allow(dead_code)]

use std::path::PathBuf;

use download_manager::{FileDownloadManager, FileDownloadManagerType, create_download_manager};

pub struct TestFile {
    pub url: String,
    pub size: u64,
    pub crc: String,
}

impl Default for TestFile {
    fn default() -> Self {
        Self {
            url: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json".to_string(),
            size: 12807982,
            crc: "C/tYqQ==".to_string(),
        }
    }
}

pub struct TestDownloadManager {
    pub manager: Box<dyn FileDownloadManager>,
    pub temp_dir: PathBuf,
    pub _temp_dir_guard: tempfile::TempDir,
    pub test_file: TestFile,
}

impl TestDownloadManager {
    /// Create a test manager (platform-specific implementation)
    pub async fn new(
        test_name: &str,
        r#type: FileDownloadManagerType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir_guard = tempfile::tempdir()?;
        let temp_dir = temp_dir_guard.path().to_path_buf();

        tracing::info!("Test download manager directory: {}", temp_dir.display());

        tracing::info!("Using {:?} manager for test: {}", r#type, test_name);
        let tokio_handle = tokio::runtime::Handle::current();
        let manager = create_download_manager(r#type, Some(test_name.to_string()), tokio_handle).await?;

        Ok(Self {
            manager,
            temp_dir,
            _temp_dir_guard: temp_dir_guard,
            test_file: TestFile::default(),
        })
    }

    pub fn dest_path(
        &self,
        filename: &str,
    ) -> PathBuf {
        self.temp_dir.join(filename)
    }

    #[allow(dead_code)]
    pub fn cleanup_artifacts(
        &self,
        filename: &str,
    ) {
        let file_path = self.dest_path(filename);
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_file(format!("{}.crc", file_path.display()));
        let _ = std::fs::remove_file(format!("{}.resume_data", file_path.display()));
    }

    #[allow(dead_code)]
    pub fn create_test_file(
        &self,
        filename: &str,
        content: &[u8],
    ) -> PathBuf {
        let file_path = self.dest_path(filename);
        std::fs::write(&file_path, content).unwrap();
        file_path
    }

    #[allow(dead_code)]
    pub fn create_crc_file(
        &self,
        filename: &str,
        crc_value: &str,
    ) {
        let file_path = self.dest_path(filename);
        let crc_path = format!("{}.crc", file_path.display());
        std::fs::write(crc_path, crc_value).unwrap();
    }

    #[allow(dead_code)]
    pub async fn cleanup(self) {
        drop(self.manager);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
