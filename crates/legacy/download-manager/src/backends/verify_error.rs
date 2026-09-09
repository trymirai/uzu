#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("downloaded file is {actual} bytes but registry declared {expected}")]
    Size {
        expected: u64,
        actual: u64,
    },
    #[error("CRC verification failed")]
    Crc,
}
