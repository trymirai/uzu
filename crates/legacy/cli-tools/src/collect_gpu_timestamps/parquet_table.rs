use std::{fs::File, path::Path, sync::Arc};

use anyhow::{Result, anyhow};
use parquet::{
    basic::{Compression, Type, ZstdLevel},
    data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};

use super::{cell::Cell, column::Column};

const ROW_GROUP_ROWS: usize = 1_000_000;

pub struct ParquetTable {
    writer: SerializedFileWriter<File>,
    columns: Vec<Column>,
    rows: usize,
}

impl ParquetTable {
    pub fn new(
        path: &Path,
        schema: &str,
    ) -> Result<Self> {
        let schema = Arc::new(parse_message_type(schema)?);
        let columns = schema
            .get_fields()
            .iter()
            .map(|field| match field.get_physical_type() {
                Type::INT64 => Ok(Column::Int64(Vec::new())),
                Type::DOUBLE => Ok(Column::Double(Vec::new())),
                Type::BYTE_ARRAY => Ok(Column::Utf8(Vec::new())),
                other => Err(anyhow!("unsupported column type {other:?}")),
            })
            .collect::<Result<_>>()?;
        let properties = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::default())).build();
        Ok(Self {
            writer: SerializedFileWriter::new(File::create(path)?, schema, Arc::new(properties))?,
            columns,
            rows: 0,
        })
    }

    pub fn push_row<'a>(
        &mut self,
        cells: impl IntoIterator<Item = Cell<'a>>,
    ) -> Result<()> {
        for (column, cell) in self.columns.iter_mut().zip(cells) {
            match (column, cell) {
                (Column::Int64(values), Cell::Int64(value)) => values.push(value),
                (Column::Double(values), Cell::Double(value)) => values.push(value),
                (Column::Utf8(values), Cell::Utf8(value)) => values.push(ByteArray::from(value)),
                _ => return Err(anyhow!("cell does not match its column")),
            }
        }
        self.rows += 1;
        if self.rows >= ROW_GROUP_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let mut row_group = self.writer.next_row_group()?;
        for column in &mut self.columns {
            let mut writer = row_group.next_column()?.ok_or_else(|| anyhow!("more buffers than schema columns"))?;
            match column {
                Column::Int64(values) => {
                    writer.typed::<Int64Type>().write_batch(values, None, None)?;
                    values.clear();
                },
                Column::Double(values) => {
                    writer.typed::<DoubleType>().write_batch(values, None, None)?;
                    values.clear();
                },
                Column::Utf8(values) => {
                    writer.typed::<ByteArrayType>().write_batch(values, None, None)?;
                    values.clear();
                },
            }
            writer.close()?;
        }
        row_group.close()?;
        self.rows = 0;
        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        if self.rows > 0 {
            self.flush()?;
        }
        self.writer.close()?;
        Ok(())
    }
}
