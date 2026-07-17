use std::{fs::File, io, path::Path};

use crate::utils::fs::{disable_page_cache, file_read_exact_at};

pub(crate) struct RowSource {
    file: File,
    row_bytes: usize,
    row_count: usize,
}

impl RowSource {
    pub(crate) fn open_exact(
        path: &Path,
        expected_len: usize,
        row_bytes: usize,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() != expected_len as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row file must contain exactly {expected_len} bytes"),
            ));
        }
        if row_bytes == 0 || !expected_len.is_multiple_of(row_bytes) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "parameter size is not a whole number of rows"));
        }
        Ok(Self {
            file,
            row_bytes,
            row_count: expected_len / row_bytes,
        })
    }

    pub(crate) fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            file: self.file.try_clone()?,
            row_bytes: self.row_bytes,
            row_count: self.row_count,
        })
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub(crate) fn disable_page_cache(&self) -> io::Result<()> {
        disable_page_cache(&self.file)
    }

    pub(crate) fn read_rows_while(
        &self,
        row_ids: &[u64],
        destination: &mut [u8],
        mut keep_going: impl FnMut() -> bool,
    ) -> io::Result<()> {
        let expected = row_ids
            .len()
            .checked_mul(self.row_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "row staging size overflow"))?;
        if destination.len() != expected {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "row staging destination has the wrong size"));
        }
        for (&row_id, destination) in row_ids.iter().zip(destination.chunks_exact_mut(self.row_bytes)) {
            if !keep_going() {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "parameter row read was cancelled"));
            }
            let row = usize::try_from(row_id)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "row ID does not fit in usize"))?;
            if row >= self.row_count {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "row ID is out of range"));
            }
            let offset = row
                .checked_mul(self.row_bytes)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "row offset overflow"))?;
            file_read_exact_at(&self.file, destination, offset as u64)?;
        }
        Ok(())
    }
}
