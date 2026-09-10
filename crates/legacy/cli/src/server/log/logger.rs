use std::{
    fs::{self},
    io::{self},
    path::{Path, PathBuf},
};

use chrono::{Local, NaiveDateTime, TimeDelta};
use uuid::Uuid;

use crate::server::log::log_file::LogFile;

const LOG_FILE_DATE_FORMAT: &str = "%Y-%m-%d-%H-%M-%S";
const LOG_FILE_EXTENSION: &str = "log";
const LOG_FILE_RETENTION: TimeDelta = TimeDelta::weeks(1);

#[derive(Clone)]
pub struct Logger {
    console: bool,
    file: Option<LogFile>,
}

impl Logger {
    pub fn new(
        console: bool,
        logs_dir_path: Option<PathBuf>,
    ) -> io::Result<Self> {
        let session_id = Uuid::new_v4().simple().to_string();
        let file = logs_dir_path
            .as_ref()
            .map(|logs_dir_path| {
                fs::create_dir_all(logs_dir_path)?;
                remove_expired_log_files(logs_dir_path)?;
                let date = Local::now().format(LOG_FILE_DATE_FORMAT);
                let file_name = format!("{}-{}.{}", date, session_id, LOG_FILE_EXTENSION);
                LogFile::new(logs_dir_path.join(file_name))
            })
            .transpose()?;

        Ok(Self {
            console,
            file,
        })
    }

    pub fn msg(
        &self,
        msg: impl Into<String>,
    ) {
        let msg_str = msg.into();
        if self.console {
            println!("{msg_str}")
        }
        self.add_str_to_file(&msg_str);
    }

    pub fn err(
        &self,
        msg: impl Into<String>,
    ) {
        let msg_str = format!("[error]: {}", msg.into());
        if self.console {
            eprintln!("{msg_str}",)
        }
        self.add_str_to_file(&msg_str);
    }

    fn add_str_to_file(
        &self,
        log: &str,
    ) {
        if let Some(file) = &self.file
            && let Err(error) = file.add(log)
        {
            eprintln!("Failed to write to log file in {}: {}", file.path().display(), error);
        }
    }
}

fn remove_expired_log_files(logs_dir_path: &Path) -> io::Result<()> {
    let cutoff = Local::now().naive_local() - LOG_FILE_RETENTION;

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
        let Ok(created_at) = NaiveDateTime::parse_from_str(date, LOG_FILE_DATE_FORMAT) else {
            continue;
        };

        if created_at < cutoff {
            fs::remove_file(path)?;
        }
    }

    Ok(())
}
