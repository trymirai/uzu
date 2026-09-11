use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetDirectoryOptions, FileSystemGetFileOptions,
    FileSystemHandle, FileSystemHandleKind, FileSystemRemoveOptions,
    js_sys::{Array, Reflect},
};

use super::{FileHandle, Result};

#[derive(Debug, Clone)]
pub enum DirectoryEntry {
    File(FileHandle),
    Directory(DirectoryHandle),
}

/// Mirrors [`FileSystemDirectoryHandle`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemDirectoryHandle).
#[derive(Debug, Clone)]
pub struct DirectoryHandle(FileSystemDirectoryHandle);

impl From<FileSystemDirectoryHandle> for DirectoryHandle {
    fn from(handle: FileSystemDirectoryHandle) -> Self {
        Self(handle)
    }
}

impl DirectoryHandle {
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

    pub async fn get_file_handle(
        &self,
        name: &str,
        create: bool,
    ) -> Result<FileHandle> {
        let web_options = FileSystemGetFileOptions::new();
        web_options.set_create(create);
        let handle = JsFuture::from(self.0.get_file_handle_with_options(name, &web_options)).await?;
        Ok(FileHandle::from(FileSystemFileHandle::from(handle)))
    }

    pub async fn get_directory_handle(
        &self,
        name: &str,
        create: bool,
    ) -> Result<DirectoryHandle> {
        let web_options = FileSystemGetDirectoryOptions::new();
        web_options.set_create(create);
        let handle = JsFuture::from(self.0.get_directory_handle_with_options(name, &web_options)).await?;
        Ok(DirectoryHandle::from(FileSystemDirectoryHandle::from(handle)))
    }

    pub async fn remove_entry(
        &self,
        name: &str,
    ) -> Result<()> {
        JsFuture::from(self.0.remove_entry(name)).await?;
        Ok(())
    }

    pub async fn remove_entry_with_options(
        &self,
        name: &str,
        recursive: bool,
    ) -> Result<()> {
        let web_options = FileSystemRemoveOptions::new();
        web_options.set_recursive(recursive);
        JsFuture::from(self.0.remove_entry_with_options(name, &web_options)).await?;
        Ok(())
    }

    pub async fn entries(&self) -> Result<Vec<(String, DirectoryEntry)>> {
        let iterator = self.0.entries();
        let mut entries = Vec::new();
        loop {
            let step = JsFuture::from(iterator.next()?).await?;
            if Reflect::get(&step, &JsValue::from_str("done"))?.as_bool().unwrap_or(true) {
                break;
            }
            // Each value is a `[name, handle]` pair.
            let pair = Array::from(&Reflect::get(&step, &JsValue::from_str("value"))?);
            let name = pair.get(0).as_string().ok_or_else(|| JsValue::from_str("directory entry has no name"))?;
            let handle = pair.get(1);
            let entry = if handle.has_type::<FileSystemFileHandle>() {
                DirectoryEntry::File(FileHandle::from(FileSystemFileHandle::from(handle)))
            } else if handle.has_type::<FileSystemDirectoryHandle>() {
                DirectoryEntry::Directory(DirectoryHandle::from(FileSystemDirectoryHandle::from(handle)))
            } else {
                return Err(JsValue::from_str("unknown directory entry handle type"));
            };
            entries.push((name, entry));
        }
        Ok(entries)
    }
}
