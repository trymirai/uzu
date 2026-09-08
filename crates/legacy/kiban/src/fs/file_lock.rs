use std::{io, path::Path};

pub struct FileLock {
    #[cfg(not(target_family = "wasm"))]
    file: std::fs::File,
    #[cfg(target_family = "wasm")]
    path: std::path::PathBuf,
}

impl FileLock {
    pub async fn try_acquire(path: &Path) -> Result<Option<Self>, io::Error> {
        if let Some(parent) = path.parent() {
            super::asyn::create_dir_all(parent).await?;
        }

        #[cfg(target_family = "wasm")]
        {
            if !super::asyn::try_exists(path).await? {
                super::asyn::write(path, b"").await?;
            }
            Ok(Some(Self {
                path: path.to_path_buf(),
            }))
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
            match file.try_lock() {
                Ok(()) => Ok(Some(Self {
                    file,
                })),
                Err(std::fs::TryLockError::WouldBlock) => Ok(None),
                Err(std::fs::TryLockError::Error(error)) => Err(error),
            }
        }
    }

    pub async fn write(
        &self,
        contents: &[u8],
    ) -> Result<(), io::Error> {
        #[cfg(target_family = "wasm")]
        {
            super::asyn::write(&self.path, contents).await
        }

        #[cfg(not(target_family = "wasm"))]
        {
            use std::io::{Seek, SeekFrom, Write};

            self.file.set_len(0)?;
            (&self.file).seek(SeekFrom::Start(0))?;
            (&self.file).write_all(contents)
        }
    }
}
