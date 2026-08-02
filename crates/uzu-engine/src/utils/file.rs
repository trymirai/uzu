use std::{io, path::Path};

pub use async_fs::read;
use futures::{
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
    lock::Mutex,
};

pub struct File(Mutex<async_fs::File>);

impl File {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self(Mutex::new(async_fs::File::open(path).await?)))
    }

    pub async fn read_exact_at(
        &self,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        let mut file = self.0.lock().await;
        file.seek(SeekFrom::Start(offset)).await?;
        file.read_exact(buffer).await
    }
}

pub async fn exists(path: impl AsRef<Path>) -> bool {
    async_fs::metadata(path).await.is_ok()
}
