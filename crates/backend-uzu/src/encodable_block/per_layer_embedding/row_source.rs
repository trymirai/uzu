use std::io;

use crate::parameters::ParameterFile;

pub(crate) struct RowSource {
    file: ParameterFile,
    row_bytes: usize,
}

impl RowSource {
    pub(crate) fn new(
        file: ParameterFile,
        row_bytes: usize,
    ) -> io::Result<Self> {
        if row_bytes == 0 || !file.len().is_multiple_of(row_bytes) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "parameter size is not a whole number of rows"));
        }
        Ok(Self {
            file,
            row_bytes,
        })
    }

    pub(crate) fn try_clone(&self) -> io::Result<Self> {
        Self::new(self.file.try_clone()?, self.row_bytes)
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub(crate) fn read_rows(
        &self,
        row_ids: &[u64],
        destination: &mut [u8],
    ) -> io::Result<()> {
        self.read_rows_while(row_ids, destination, || true)
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
            let offset = row
                .checked_mul(self.row_bytes)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "row offset overflow"))?;
            self.file.read_exact_at(offset, destination)?;
        }
        Ok(())
    }
}
