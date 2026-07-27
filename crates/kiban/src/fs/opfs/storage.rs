use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::FileSystemDirectoryHandle;

use super::{DirectoryHandle, Result};

/// Mirrors [`StorageManager.getDirectory()`](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/getDirectory).
pub async fn get_storage_root() -> Result<DirectoryHandle> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window object available"))?;
    let directory = JsFuture::from(window.navigator().storage().get_directory()).await?;
    Ok(DirectoryHandle::from(FileSystemDirectoryHandle::from(directory)))
}
