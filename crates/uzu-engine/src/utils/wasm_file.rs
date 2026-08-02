use std::{
    io::{self, Error, ErrorKind},
    path::{Component, Path},
};

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{DomException, FileSystemDirectoryHandle, FileSystemFileHandle, js_sys::Uint8Array};

pub struct File(web_sys::File);

impl File {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let name =
            path.file_name().and_then(|name| name.to_str()).ok_or_else(|| Error::from(ErrorKind::InvalidInput))?;
        let directory = directory(path.parent().ok_or_else(|| Error::from(ErrorKind::InvalidInput))?).await?;
        let handle =
            FileSystemFileHandle::from(JsFuture::from(directory.get_file_handle(name)).await.map_err(io_error)?);
        let file = JsFuture::from(handle.get_file()).await.map_err(io_error)?.into();
        Ok(Self(file))
    }

    pub async fn read_exact_at(
        &self,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        let mut offset = offset;
        for chunk in buffer.chunks_mut(16 * 1024 * 1024) {
            let end = offset + chunk.len() as u64;
            let blob = self.0.slice_with_f64_and_f64(offset as f64, end as f64).map_err(io_error)?;
            let bytes = Uint8Array::new(&JsFuture::from(blob.array_buffer()).await.map_err(io_error)?);
            if bytes.length() as usize != chunk.len() {
                return Err(Error::from(ErrorKind::UnexpectedEof));
            }
            bytes.copy_to(chunk);
            offset = end;
        }
        Ok(())
    }
}

pub async fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let file = File::open(path).await?;
    let buffer = JsFuture::from(file.0.array_buffer()).await.map_err(io_error)?;
    Ok(Uint8Array::new(&buffer).to_vec())
}

pub async fn exists(path: impl AsRef<Path>) -> bool {
    directory(path.as_ref()).await.is_ok() || File::open(path).await.is_ok()
}

async fn directory(path: &Path) -> io::Result<FileSystemDirectoryHandle> {
    let window = web_sys::window().ok_or_else(|| Error::new(ErrorKind::Unsupported, "OPFS requires a window"))?;
    let mut directory = FileSystemDirectoryHandle::from(
        JsFuture::from(window.navigator().storage().get_directory()).await.map_err(io_error)?,
    );
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                let name = name.to_str().ok_or_else(|| Error::from(ErrorKind::InvalidInput))?;
                directory = JsFuture::from(directory.get_directory_handle(name)).await.map_err(io_error)?.into();
            },
            Component::RootDir | Component::CurDir => {},
            _ => return Err(Error::new(ErrorKind::InvalidInput, "OPFS paths cannot contain parent components")),
        }
    }
    Ok(directory)
}

fn io_error(error: JsValue) -> Error {
    if let Some(exception) = error.dyn_ref::<DomException>() {
        let kind = match exception.name().as_str() {
            "NotFoundError" => ErrorKind::NotFound,
            "NotAllowedError" | "SecurityError" => ErrorKind::PermissionDenied,
            "TypeMismatchError" | "TypeError" => ErrorKind::InvalidInput,
            _ => ErrorKind::Other,
        };
        Error::new(kind, exception.message())
    } else {
        Error::other(format!("{error:?}"))
    }
}
