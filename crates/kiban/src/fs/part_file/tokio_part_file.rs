use std::{io, path::Path};

use tokio::{
    fs::{File, OpenOptions},
    io::AsyncWriteExt,
};

use super::PartFile;

pub struct TokioPartFile {
    file: File,
}

impl TokioPartFile {
    pub async fn new(
        path: impl AsRef<Path>,
        resume_from_bytes: u64,
    ) -> io::Result<Self> {
        let file = if resume_from_bytes > 0 {
            OpenOptions::new().create(true).append(true).open(path.as_ref()).await?
        } else {
            OpenOptions::new().create(true).write(true).truncate(true).open(path.as_ref()).await?
        };
        Ok(Self {
            file,
        })
    }
}

#[async_trait::async_trait]
impl PartFile for TokioPartFile {
    async fn write_all(
        &mut self,
        buf: &[u8],
    ) -> io::Result<()> {
        self.file.write_all(buf).await
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.file.flush().await
    }
}
