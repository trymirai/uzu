use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FileCheck {
    CRC(String),
    #[default]
    None,
}

impl FileCheck {
    pub fn expected_crc(&self) -> Option<String> {
        match self {
            Self::CRC(crc) => Some(crc.clone()),
            Self::None => None,
        }
    }
}
