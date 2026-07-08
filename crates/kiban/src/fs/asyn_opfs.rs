use std::{
    io,
    io::ErrorKind,
    path::{Component, Path},
    sync::OnceLock,
};

use tokio::sync::Mutex;
use web_sys::{js_sys::Reflect, wasm_bindgen::JsValue};

use super::opfs::{self, WriteCommandType, WriteParams};
use crate::time::SystemTime;

pub(crate) async fn dir_create_all(path: &str) -> Result<(), io::Error> {
    let mut root = get_root_dir().await?;
    for segment in normalized_segments(path)? {
        root = get_dir_handle(&root, &segment, true).await?;
    }
    Ok(())
}

pub(crate) async fn exists(path: &str) -> Result<bool, io::Error> {
    let root = get_root_dir().await?;
    let (parent, name) = match resolve_parent(&root, path).await {
        Ok(path) => path,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err),
    };

    let file_err = match get_file_handle(parent.clone(), &name, false).await {
        Ok(_) => return Ok(true),
        Err(err) => err,
    };
    match get_dir_handle(&parent, &name, false).await {
        Ok(_) => Ok(true),
        Err(dir_err) if file_err.kind() == ErrorKind::NotFound && dir_err.kind() == ErrorKind::NotFound => Ok(false),
        Err(dir_err) if dir_err.kind() == ErrorKind::NotFound => Err(file_err),
        Err(dir_err) => Err(dir_err),
    }
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
    let mut offset = 0u64;
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
        offset += chunk_length as u64;
    }

    writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;

    Ok(())
}

pub(crate) async fn file_copy_if_absent(
    src_path: &str,
    dst_path: &str,
) -> Result<(), io::Error> {
    let _guard = create_if_absent_lock().lock().await;
    if exists(dst_path).await? {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "file already exists"));
    }
    file_copy(src_path, dst_path).await
}

pub(crate) async fn file_length(path: &str) -> Result<u64, io::Error> {
    let file = get_file_handle_root(path, false).await?;
    let size = file.size().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(size)
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
    let (parent, name) = resolve_parent(&root, path).await?;
    let _ = get_file_handle(parent.clone(), &name, false).await?;
    parent.remove_entry(&name).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
    Ok(())
}

pub(crate) async fn file_rename(
    src_path: &str,
    dst_path: &str,
) -> Result<(), io::Error> {
    if same_opfs_entry(src_path, dst_path)? {
        let _ = get_file_handle_root(src_path, false).await?;
        return Ok(());
    }

    file_copy(src_path, dst_path).await?;
    file_remove(src_path).await
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

pub(crate) async fn file_write_if_absent(
    path: &str,
    contents: &[u8],
) -> Result<(), io::Error> {
    let _guard = create_if_absent_lock().lock().await;
    if exists(path).await? {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "file already exists"));
    }
    file_write(path, contents).await
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
    let (parent, name) = resolve_parent(&root, path).await?;
    get_file_handle(parent, &name, create).await
}

pub(crate) async fn get_root_dir() -> Result<opfs::DirectoryHandle, io::Error> {
    opfs::get_storage_root().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))
}

pub(crate) async fn resolve_parent(
    root: &opfs::DirectoryHandle,
    path: &str,
) -> Result<(opfs::DirectoryHandle, String), io::Error> {
    let mut parts = normalized_segments(path)?;
    let name = parts.pop().ok_or(io::Error::new(ErrorKind::InvalidInput, "empty path"))?;
    let mut dir = root.clone();
    for segment in parts {
        dir = get_dir_handle(&dir, &segment, false).await?;
    }
    Ok((dir, name))
}

fn normalized_segments(path: &str) -> Result<Vec<String>, io::Error> {
    let mut segments = Vec::new();
    let mut rooted = false;

    for component in Path::new(path).components() {
        match component {
            Component::Prefix(_) => {
                return Err(io::Error::new(ErrorKind::InvalidInput, "OPFS paths do not support prefixes"));
            },
            Component::RootDir => {
                rooted = true;
                segments.clear();
            },
            Component::CurDir => {},
            Component::ParentDir => {
                if segments.pop().is_none() && !rooted {
                    return Err(io::Error::new(ErrorKind::InvalidInput, "OPFS path escapes storage root"));
                }
            },
            Component::Normal(segment) => {
                let segment = segment
                    .to_str()
                    .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "OPFS path is not valid UTF-8"))?;
                segments.push(segment.to_string());
            },
        }
    }

    Ok(segments)
}

fn same_opfs_entry(
    src_path: &str,
    dst_path: &str,
) -> Result<bool, io::Error> {
    Ok(normalized_segments(src_path)? == normalized_segments(dst_path)?)
}

fn create_if_absent_lock() -> &'static Mutex<()> {
    static CREATE_IF_ABSENT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    CREATE_IF_ABSENT_LOCK.get_or_init(|| Mutex::new(()))
}

pub(crate) fn js_value_to_io_error(
    value: &JsValue,
    kind: ErrorKind,
) -> io::Error {
    io::Error::new(js_value_to_io_error_kind(value, kind), format!("{:?}", value))
}

fn js_value_to_io_error_kind(
    value: &JsValue,
    fallback: ErrorKind,
) -> ErrorKind {
    let name = Reflect::get(value, &JsValue::from_str("name")).ok().and_then(|name| name.as_string());
    match name.as_deref() {
        Some("NotFoundError") => ErrorKind::NotFound,
        Some("NotAllowedError") => ErrorKind::PermissionDenied,
        Some("TypeMismatchError") => ErrorKind::NotADirectory,
        _ => fallback,
    }
}
