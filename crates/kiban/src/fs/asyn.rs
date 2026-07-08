use std::{io, path::Path};

use crate::time::SystemTime;

pub async fn create_dir_all(path: impl AsRef<Path>) -> Result<(), io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::dir_create_all(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::create_dir_all(path).await
}

pub async fn exists(path: impl AsRef<Path>) -> bool {
    try_exists(path.as_ref()).await.unwrap_or(false)
}

pub async fn file_length(path: impl AsRef<Path>) -> Result<u64, io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_length(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    Ok(tokio::fs::metadata(path.as_ref()).await?.len())
}

pub async fn file_modified(path: impl AsRef<Path>) -> Result<SystemTime, io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_modified(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::metadata(path.as_ref()).await?.modified()
}

pub async fn hard_link(
    original: impl AsRef<Path>,
    link: impl AsRef<Path>,
) -> Result<(), io::Error> {
    // OPFS doesn't support linking, that's why just copy
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_copy(original.as_ref().to_str().unwrap(), link.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::hard_link(original.as_ref(), link.as_ref()).await
}

pub async fn is_file(path: impl AsRef<Path>) -> bool {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::is_file(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::metadata(path).await.map(|m| m.is_file()).unwrap_or(false)
}

pub async fn read(path: impl AsRef<Path>) -> Result<Vec<u8>, io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_read(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::read(path.as_ref()).await
}

pub async fn read_to_string(path: impl AsRef<Path>) -> Result<String, io::Error> {
    #[cfg(target_family = "wasm")]
    {
        let content = read(path).await?;
        let string = String::from_utf8(content).map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        Ok(string)
    }

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::read_to_string(path.as_ref()).await
}

pub async fn remove_file(path: impl AsRef<Path>) -> Result<(), io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_remove(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::remove_file(path.as_ref()).await
}

pub async fn rename(
    src_file: impl AsRef<Path>,
    dst_file: impl AsRef<Path>,
) -> Result<(), io::Error> {
    #[cfg(target_family = "wasm")]
    {
        let src_file_name = src_file.as_ref().to_str().unwrap();
        let dst_file_name = dst_file.as_ref().to_str().unwrap();
        super::asyn_opfs::file_copy(src_file_name, dst_file_name).await?;
        super::asyn_opfs::file_remove(src_file_name).await?;
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::rename(src_file, dst_file.as_ref()).await
}

pub async fn try_exists(path: impl AsRef<Path>) -> Result<bool, io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_exists(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::try_exists(path.as_ref()).await
}

pub async fn write(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
) -> Result<(), io::Error> {
    #[cfg(target_family = "wasm")]
    return super::asyn_opfs::file_write(path.as_ref().to_str().unwrap(), contents.as_ref()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::write(path, contents).await
}

pub async fn write_with_sync_all(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
) -> Result<(), io::Error> {
    #[cfg(target_family = "wasm")]
    {
        if try_exists(path.as_ref()).await? {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "file already exists"));
        }
        return write(path, contents).await;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        use tokio::io::AsyncWriteExt;

        let mut file = tokio::fs::OpenOptions::new().write(true).create_new(true).open(&path).await?;
        file.write_all(contents.as_ref()).await?;
        file.sync_all().await
    }
}
