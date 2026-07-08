use std::{io, io::ErrorKind};

use super::PartFile;
use crate::fs::{
    asyn_opfs::{get_file_handle, get_root_dir, js_value_to_io_error, resolve_parent},
    opfs::{WritableFileStream, WriteCommandType, WriteParams},
};

pub struct OpfsPartFile {
    writer: WritableFileStream,
    position: usize,
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
            writer,
            position: resume_from_bytes as usize,
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
                position: Some(self.position),
                size: None,
            })
            .await
            .map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))?;
        self.position += buf.len();
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), io::Error> {
        self.writer.close().await.map_err(|err| js_value_to_io_error(&err, ErrorKind::Other))
    }
}
