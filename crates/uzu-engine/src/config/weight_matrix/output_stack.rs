use std::ops::Range;

use thiserror::Error;
use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::AnyWeightMatrixSpec;

#[uzu_config(super::WeightMatrixSpec)]
pub struct OutputStackSpec {
    pub parts: Box<[OutputStackPart]>,
}

#[uzu_config]
pub struct OutputStackPart {
    pub rows: u32,
    pub spec: AnyWeightMatrixSpec,
}

#[derive(Debug, Error)]
#[error("output stack has {actual_rows} rows, expected {expected_rows}")]
pub struct OutputStackError {
    pub actual_rows: u64,
    pub expected_rows: u32,
}

impl OutputStackSpec {
    pub fn into_parts(
        self,
        expected_rows: u32,
    ) -> Result<Vec<(Range<u32>, AnyWeightMatrixSpec)>, OutputStackError> {
        let actual_rows = self.parts.iter().map(|part| u64::from(part.rows)).sum();
        if actual_rows != u64::from(expected_rows) {
            return Err(OutputStackError {
                actual_rows,
                expected_rows,
            });
        }
        let mut parts = Vec::with_capacity(self.parts.len());
        let mut first_row = 0u32;
        for part in self.parts {
            let end = first_row + part.rows;
            parts.push((first_row..end, part.spec));
            first_row = end;
        }
        Ok(parts)
    }
}
