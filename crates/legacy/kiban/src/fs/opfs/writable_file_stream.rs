use wasm_bindgen::JsValue;
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    Blob, FileSystemWritableFileStream,
    js_sys::{Array, Uint8Array},
};

use super::Result;

/// Mirrors the `type` field accepted by
/// [`FileSystemWritableFileStream.write()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemWritableFileStream/write#type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteCommandType {
    /// Write [`data`](WriteParams::data) at [`position`](WriteParams::position).
    Write,
    /// Move the cursor to [`position`](WriteParams::position).
    Seek,
    /// Resize the file to [`size`](WriteParams::size) bytes.
    Truncate,
}

/// Mirrors the [`WriteParams`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemWritableFileStream/write#parameters)
/// dictionary.
#[derive(Debug, Clone)]
pub struct WriteParams {
    pub command_type: WriteCommandType,
    /// Used by [`WriteCommandType::Write`].
    pub data: Option<Vec<u8>>,
    /// Used by [`WriteCommandType::Write`] and [`WriteCommandType::Seek`].
    pub position: Option<u64>,
    /// Used by [`WriteCommandType::Truncate`].
    pub size: Option<u64>,
}

/// Mirrors [`FileSystemWritableFileStream`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemWritableFileStream).
#[derive(Debug, Clone)]
pub struct WritableFileStream(FileSystemWritableFileStream);

impl From<FileSystemWritableFileStream> for WritableFileStream {
    fn from(stream: FileSystemWritableFileStream) -> Self {
        Self(stream)
    }
}

impl WritableFileStream {
    pub async fn write(
        &self,
        data: &[u8],
    ) -> Result<()> {
        JsFuture::from(self.0.write_with_blob(&blob_from_bytes(data)?)?).await?;
        Ok(())
    }

    pub async fn write_with_params(
        &self,
        params: &WriteParams,
    ) -> Result<()> {
        let web_params = web_sys::WriteParams::new(match params.command_type {
            WriteCommandType::Write => web_sys::WriteCommandType::Write,
            WriteCommandType::Seek => web_sys::WriteCommandType::Seek,
            WriteCommandType::Truncate => web_sys::WriteCommandType::Truncate,
        });

        if let Some(data) = &params.data {
            web_params.set_data(&JsValue::from(blob_from_bytes(data)?));
        }
        if let Some(position) = params.position {
            web_params.set_position(Some(position as f64));
        }
        if let Some(size) = params.size {
            web_params.set_size(Some(size as f64));
        }

        JsFuture::from(self.0.write_with_write_params(&web_params)?).await?;
        Ok(())
    }

    pub async fn seek(
        &self,
        offset: u64,
    ) -> Result<()> {
        JsFuture::from(self.0.seek_with_f64(offset as f64)?).await?;
        Ok(())
    }

    pub async fn truncate(
        &self,
        size: u64,
    ) -> Result<()> {
        JsFuture::from(self.0.truncate_with_f64(size as f64)?).await?;
        Ok(())
    }

    pub async fn close(&self) -> Result<()> {
        JsFuture::from(self.0.close()).await?;
        Ok(())
    }

    pub async fn abort(&self) -> Result<()> {
        JsFuture::from(self.0.abort()).await?;
        Ok(())
    }

    pub fn close_in_background(self) {
        let close = self.0.close();
        spawn_local(async move {
            let _ = JsFuture::from(close).await;
        });
    }
}

fn blob_from_bytes(data: &[u8]) -> Result<Blob> {
    let bytes = Uint8Array::from(data);
    let parts = Array::new();
    parts.push(&bytes);
    Blob::new_with_u8_array_sequence(&parts)
}
