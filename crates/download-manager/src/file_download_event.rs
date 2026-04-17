use crate::PathBuf;

#[derive(Debug, Clone)]
pub enum FileDownloadEvent {
    DownloadCompleted {
        tmp_path: PathBuf,
        final_destination: PathBuf,
    },
    ProgressUpdate {
        bytes_written: u64,
        total_bytes_written: u64,
        total_bytes_expected: u64,
    },
    Error {
        message: String,
    },
}
