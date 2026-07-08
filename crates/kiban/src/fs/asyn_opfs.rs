use std::{io, io::ErrorKind};

use web_sys::wasm_bindgen::JsValue;

use super::opfs::{self, WriteCommandType, WriteParams};
use crate::time::SystemTime;

pub(crate) async fn dir_create_all(path: &str) -> Result<(), io::Error> {
    let mut root = get_root_dir().await?;
    let segments = path.split('/').filter(|s| !s.is_empty());
    for segment in segments {
        root = get_dir_handle(&root, segment, true).await?;
    }
    Ok(())
}

pub(crate) async fn exists(path: &str) -> Result<bool, io::Error> {
    let root = match get_root_dir().await {
        Ok(root) => root,
        Err(_) => return Ok(false),
    };
    let (parent, name) = match resolve_parent(&root, path, false).await {
        Ok(path) => path,
        Err(_) => return Ok(false),
    };
    if get_file_handle(parent.clone(), &name, false).await.is_ok() {
        return Ok(true);
    }
    Ok(get_dir_handle(&parent, &name, false).await.is_ok())
}

pub(crate) async fn file_copy(
    src_path: &str,
    dst_path: &str,
) -> Result<(), io::Error> {
    if src_path == dst_path {
        return Err(io::Error::new(
            ErrorKind::InvalidFilename,
            "source and destination file paths are equal".to_string(),
        ));
    }

    let src_file = get_file_handle_root(src_path, false).await?;
    let total = src_file.size().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;

    let dst_file = get_file_handle_root(dst_path, true).await?;
    let writer = dst_file.create_writable(false).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;

    let chunk_size = 16 * 1024;
    let mut offset = 0usize;
    while offset < total {
        let end = (offset + chunk_size).min(total);
        let chunk =
            src_file.read_range(offset..end).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        if chunk.is_empty() {
            break;
        }

        let chunk_length = chunk.len();
        let params = &WriteParams {
            command_type: WriteCommandType::Write,
            data: Some(chunk),
            position: Some(offset),
            size: None,
        };
        match writer.write_with_params(params).await {
            Ok(_) => {},
            Err(err) => {
                writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
                return Err(js_value_to_io_error(&err, ErrorKind::Other));
            },
        }
        offset += chunk_length;
    }

    writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;

    Ok(())
}

pub(crate) async fn file_length(path: &str) -> Result<u64, io::Error> {
    let file = get_file_handle_root(path, false).await?;
    let size = file.size().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(size as u64)
}

pub(crate) async fn file_modified(path: &str) -> Result<SystemTime, io::Error> {
    let file_handle = get_file_handle_root(path, false).await?;
    let file = file_handle.get_file().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::InvalidInput))?;
    Ok(file.last_modified())
}

pub(crate) async fn file_read(path: &str) -> Result<Vec<u8>, io::Error> {
    let file = get_file_handle_root(path, false).await?;
    let result = file.read().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(result)
}

pub(crate) async fn file_remove(path: &str) -> Result<(), io::Error> {
    let root = get_root_dir().await?;
    let (parent, name) = resolve_parent(&root, path, false).await?;
    parent.remove_entry(&name).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(())
}

pub(crate) async fn file_write(
    path: &str,
    contents: &[u8],
) -> Result<(), io::Error> {
    let file = get_file_handle_root(path, true).await?;
    let writer = file.create_writable(false).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    writer.write(contents).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(())
}

pub(crate) async fn is_file(path: &str) -> bool {
    get_file_handle_root(path, false).await.is_ok()
}

pub(crate) async fn get_dir_handle(
    dir: &opfs::DirectoryHandle,
    path: &str,
    create: bool,
) -> Result<opfs::DirectoryHandle, io::Error> {
    dir.get_directory_handle(path, create).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))
}

pub(crate) async fn get_file_handle(
    dir_handle: opfs::DirectoryHandle,
    path: &str,
    create: bool,
) -> Result<opfs::FileHandle, io::Error> {
    dir_handle.get_file_handle(path, create).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))
}

pub(crate) async fn get_file_handle_root(
    path: &str,
    create: bool,
) -> Result<opfs::FileHandle, io::Error> {
    let root = get_root_dir().await?;
    let (parent, name) = resolve_parent(&root, path, create).await?;
    get_file_handle(parent, &name, create).await
}

pub(crate) async fn get_root_dir() -> Result<opfs::DirectoryHandle, io::Error> {
    opfs::get_storage_root().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))
}

pub(crate) async fn resolve_parent(
    root: &opfs::DirectoryHandle,
    path: &str,
    create: bool,
) -> Result<(opfs::DirectoryHandle, String), io::Error> {
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let name = parts.pop().ok_or(io::Error::new(ErrorKind::InvalidInput, "empty path"))?.to_string();
    let mut dir = root.clone();
    for segment in parts {
        dir = get_dir_handle(&dir, segment, create).await?;
    }
    Ok((dir, name))
}

pub(crate) fn js_value_to_io_error(
    value: &JsValue,
    kind: ErrorKind,
) -> io::Error {
    io::Error::new(kind, format!("{:?}", value))
}
