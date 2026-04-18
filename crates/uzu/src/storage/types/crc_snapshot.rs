use std::{fs::remove_file, path::Path};

use crate::storage::Error;

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
    ) -> Result<(), Error> {
        let crc_path = Self::crc_file_path(model_dir, filename);
        if crc_path.exists() {
            remove_file(&crc_path).map_err(|error| Error::IO {
                message: error.to_string(),
            })?;
        }
        Ok(())
    }
}
