use std::{fs::remove_file, path::Path};

use crate::storage::StorageError;

pub struct CrcSnapshot;

impl CrcSnapshot {
    fn crc_file_path(
        model_dir: &Path,
        filename: &str,
    ) -> std::path::PathBuf {
        model_dir.join(format!("{}.crc", filename))
    }

    pub fn remove_crc(
        model_dir: &Path,
        filename: &str,
    ) -> Result<(), StorageError> {
        let crc_path = Self::crc_file_path(model_dir, filename);
        if crc_path.exists() {
            remove_file(&crc_path).map_err(|error| StorageError::IO {
                message: error.to_string(),
            })?;
        }
        Ok(())
    }
}
