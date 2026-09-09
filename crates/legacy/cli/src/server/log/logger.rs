use std::{fmt::format, path::PathBuf};

use chrono::Local;
use uuid::Uuid;

use crate::server::log::log_file::LogFile;

#[derive(Clone)]
pub struct Logger {
    session_id: String,
    console: bool,
    logs_dir_path: Option<PathBuf>,
    file: Option<LogFile>,
}

impl Logger {
    pub fn new(
        console: bool,
        logs_dir_path: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let session_id = Uuid::new_v4().simple().to_string();
        let file = logs_dir_path
            .as_ref()
            .map(|logs_dir_path| {
                let date = Local::now().format("%Y-%m-%d-%H-%M-%S");
                let file_name = format!("{}-{}.log", date, session_id);
                LogFile::new(logs_dir_path.join(file_name))
            })
            .transpose()?;

        Ok(Self {
            session_id,
            console,
            logs_dir_path,
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
        self.add_to_file(&msg_str);
    }

    pub fn err(
        &self,
        msg: impl Into<String>,
    ) {
        let msg_str = format!("[error]: {}", msg.into());
        if self.console {
            eprintln!("{msg_str}",)
        }
        self.add_to_file(&msg_str);
    }

    fn add_to_file(
        &self,
        log: &str,
    ) {
        if let Some(file) = &self.file {
            if let Err(error) = file.add(log) {
                eprintln!("Failed to write to log file in {}: {}", file.path().display(), error);
            }
        }
    }
}
