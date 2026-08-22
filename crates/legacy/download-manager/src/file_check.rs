use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FileCheck {
    CRC(String),
    Sha256(String),
    GitBlobSha1(String),
    None,
}

impl FileCheck {
    pub fn expected_crc(&self) -> Option<String> {
        match self {
            Self::CRC(crc) => Some(crc.clone()),
            Self::Sha256(_) | Self::GitBlobSha1(_) | Self::None => None,
        }
    }

    pub(crate) fn verification_failure_message(&self) -> &'static str {
        match self {
            Self::CRC(_) => "CRC verification failed",
            Self::Sha256(_) => "SHA-256 verification failed",
            Self::GitBlobSha1(_) => "Git blob SHA-1 verification failed",
            Self::None => "integrity verification failed",
        }
    }
}
