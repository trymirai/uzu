use std::{io, path::Path};

#[cfg(target_family = "wasm")]
mod opfs_part_file;
#[cfg(not(target_family = "wasm"))]
mod tokio_part_file;

#[cfg(target_family = "wasm")]
use opfs_part_file::OpfsPartFile;
#[cfg(not(target_family = "wasm"))]
use tokio_part_file::TokioPartFile;

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait PartFile {
    async fn write_all(
        &mut self,
        buf: &[u8],
    ) -> Result<(), io::Error>;

    async fn flush(&mut self) -> Result<(), io::Error>;
}

impl dyn PartFile {
    pub async fn new(
        path: impl AsRef<Path>,
        resume_from_bytes: u64,
    ) -> Result<impl PartFile, io::Error> {
        #[cfg(target_family = "wasm")]
        return OpfsPartFile::new(path.as_ref().to_str().unwrap(), resume_from_bytes).await;

        #[cfg(not(target_family = "wasm"))]
        TokioPartFile::new(path, resume_from_bytes).await
    }
}
