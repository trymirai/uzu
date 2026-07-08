use std::{
    ops::{Bound, RangeBounds},
    time::Duration,
};

use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Blob, js_sys::Uint8Array};

use super::Result;
use crate::time::SystemTime;

/// Mirrors [`File`](https://developer.mozilla.org/en-US/docs/Web/API/File), which
/// inherits the read methods of [`Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob).
#[derive(Debug, Clone)]
pub struct File(web_sys::File);

impl From<web_sys::File> for File {
    fn from(file: web_sys::File) -> Self {
        Self(file)
    }
}

impl File {
    pub fn last_modified(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(self.0.last_modified() as u64)
    }

    pub async fn read(&self) -> Result<Vec<u8>> {
        let buffer = JsFuture::from(self.0.array_buffer()).await?;
        Ok(bytes_from_array_buffer(
            &buffer,
            usize::try_from(self.size()).map_err(|_| JsValue::from_str("file is too large to read into memory"))?,
        ))
    }

    pub async fn read_range<R: RangeBounds<u64>>(
        &self,
        range: R,
    ) -> Result<Vec<u8>> {
        let size = self.size();
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => size,
        }
        .min(size);

        if start >= end {
            return Ok(Vec::new());
        }

        let blob: Blob = self.0.slice_with_f64_and_f64(start as f64, end as f64)?;
        let buffer = JsFuture::from(blob.array_buffer()).await?;
        Ok(bytes_from_array_buffer(
            &buffer,
            usize::try_from(end - start)
                .map_err(|_| JsValue::from_str("file range is too large to read into memory"))?,
        ))
    }

    pub fn size(&self) -> u64 {
        self.0.size() as u64
    }

    pub async fn text(&self) -> Result<String> {
        JsFuture::from(self.0.text()).await?.as_string().ok_or_else(|| JsValue::from_str("file is not valid UTF-8"))
    }
}

fn bytes_from_array_buffer(
    buffer: &JsValue,
    len: usize,
) -> Vec<u8> {
    let array = Uint8Array::new(buffer);
    let mut bytes = vec![0u8; len];
    array.copy_to(&mut bytes);
    bytes
}
