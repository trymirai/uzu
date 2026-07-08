use std::{io, io::ErrorKind};

use super::PartFile;
use crate::fs::{
    asyn_opfs::{get_file_handle, get_root_dir, js_value_to_io_error, resolve_parent},
    opfs::{FileHandle, WritableFileStream, WriteCommandType, WriteParams},
};

pub struct OpfsPartFile {
    file: FileHandle,
    writer: WritableFileStream,
    position: u64,
}

impl OpfsPartFile {
    pub async fn new(
        path: &str,
        resume_from_bytes: u64,
    ) -> Result<Self, io::Error> {
        let root = get_root_dir().await?;
        let (dir, name) = resolve_parent(&root, path, true).await?;
        let file = get_file_handle(dir, &name, true).await?;
        let writer = file
            .create_writable(resume_from_bytes > 0)
            .await
            .map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        Ok(Self {
            file,
            writer,
            position: resume_from_bytes,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl PartFile for OpfsPartFile {
    async fn write_all(
        &mut self,
        buf: &[u8],
    ) -> Result<(), io::Error> {
        self.writer
            .write_with_params(&WriteParams {
                command_type: WriteCommandType::Write,
                data: Some(buf.to_vec()),
                position: Some(self.position as usize),
                size: None,
            })
            .await
            .map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        self.position += buf.len() as u64;
        Ok(())
    }

    /// OPFS doesn't have flush(), that's why writer is closed and then reopened
    async fn flush(&mut self) -> Result<(), io::Error> {
        self.writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        self.writer =
            self.file.create_writable(true).await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        Ok(())
    }
}
