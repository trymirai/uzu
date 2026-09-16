use std::{
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use tracing_subscriber::fmt::writer::{MakeWriter, MutexGuardWriter};

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

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl<'a> MakeWriter<'a> for LogFile {
    type Writer = MutexGuardWriter<'a, File>;

    fn make_writer(&'a self) -> Self::Writer {
        self.file.make_writer()
    }
}
