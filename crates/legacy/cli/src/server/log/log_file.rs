use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct LogFile {
    file: Arc<Mutex<File>>,
    path: PathBuf,
}

impl LogFile {
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            path: path.into(),
        })
    }

    pub fn add(
        &self,
        log: impl AsRef<str>,
    ) -> io::Result<()> {
        let mut file = self.file.lock().map_err(|_| io::Error::other("log file lock poisoned"))?;
        writeln!(file, "{}", log.as_ref())
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}
