use std::{
    fs::{self},
    io::{self},
    path::{Path, PathBuf},
};

use jiff::{SignedDuration, Zoned, civil::DateTime};
use tracing_subscriber::fmt::writer::MakeWriter;
use uuid::Uuid;

use crate::server::log::log_file::LogFile;

const LOG_FILE_DATE_FORMAT: &str = "%Y-%m-%d-%H-%M-%S";
const LOG_FILE_EXTENSION: &str = "log";
const LOG_FILE_RETENTION: SignedDuration = SignedDuration::from_hours(7 * 24);

#[derive(Clone)]
pub struct FileLogger {
    file: LogFile,
}

impl FileLogger {
    pub fn new(logs_dir_path: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&logs_dir_path)?;
        Self::remove_expired_log_files(&logs_dir_path)?;

        let date = Zoned::now().strftime(LOG_FILE_DATE_FORMAT);
        let session_id = Uuid::new_v4().simple().to_string();
        let file_name = format!("{}-{}.{}", date, session_id, LOG_FILE_EXTENSION);
        let file = LogFile::new(logs_dir_path.join(file_name))?;
        Ok(Self {
            file,
        })
    }

    pub fn file_path(&self) -> &PathBuf {
        self.file.path()
    }

    fn remove_expired_log_files(logs_dir_path: &Path) -> io::Result<()> {
        let cutoff = Zoned::now().datetime() - LOG_FILE_RETENTION;

        for entry in fs::read_dir(logs_dir_path)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }

            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != LOG_FILE_EXTENSION) {
                continue;
            }

            let Some(file_stem) = path.file_stem().and_then(|file_stem| file_stem.to_str()) else {
                continue;
            };
            let Some((date, _session_id)) = file_stem.rsplit_once('-') else {
                continue;
            };
            let Ok(created_at) = DateTime::strptime(LOG_FILE_DATE_FORMAT, date) else {
                continue;
            };

            if created_at < cutoff {
                fs::remove_file(path)?;
            }
        }

        Ok(())
    }
}

impl<'a> MakeWriter<'a> for FileLogger {
    type Writer = <LogFile as MakeWriter<'a>>::Writer;

    fn make_writer(&'a self) -> Self::Writer {
        self.file.make_writer()
    }
}
