use std::ops::RangeBounds;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemCreateWritableOptions, FileSystemFileHandle, FileSystemHandle, FileSystemHandleKind,
    FileSystemWritableFileStream,
};

use super::{File, Result, WritableFileStream};

/// Mirrors [`FileSystemFileHandle`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle).
#[derive(Debug, Clone)]
pub struct FileHandle(FileSystemFileHandle);

impl From<FileSystemFileHandle> for FileHandle {
    fn from(handle: FileSystemFileHandle) -> Self {
        Self(handle)
    }
}

impl FileHandle {
    pub fn name(&self) -> String {
        self.0.name()
    }

    pub fn kind(&self) -> FileSystemHandleKind {
        self.0.kind()
    }

    pub async fn is_same_entry(
        &self,
        other: &FileSystemHandle,
    ) -> Result<bool> {
        let same = JsFuture::from(self.0.is_same_entry(other)).await?;
        Ok(same.as_bool().unwrap_or(false))
    }

    pub async fn get_file(&self) -> Result<File> {
        let file = JsFuture::from(self.0.get_file()).await?;
        Ok(File::from(web_sys::File::from(file)))
    }

    pub async fn create_writable(
        &self,
        keep_existing_data: bool,
    ) -> Result<WritableFileStream> {
        let web_options = FileSystemCreateWritableOptions::new();
        web_options.set_keep_existing_data(keep_existing_data);
        let stream = JsFuture::from(self.0.create_writable_with_options(&web_options)).await?;
        Ok(WritableFileStream::from(stream.unchecked_into::<FileSystemWritableFileStream>()))
    }

    pub async fn read(&self) -> Result<Vec<u8>> {
        self.get_file().await?.read().await
    }

    pub async fn read_range<R: RangeBounds<u64>>(
        &self,
        range: R,
    ) -> Result<Vec<u8>> {
        self.get_file().await?.read_range(range).await
    }

    pub async fn size(&self) -> Result<u64> {
        Ok(self.get_file().await?.size())
    }
}
