use std::ops::Range;

use uzu_engine_macros::uzu_config;

use super::AnyWeightMatrixSpec;

#[uzu_config(super::WeightMatrixSpec)]
pub struct OutputStackSpec {
    pub parts: Box<[OutputStackPart]>,
}

#[uzu_config]
pub struct OutputStackPart {
    pub rows: u32,
    pub spec: AnyWeightMatrixSpec,
}

impl OutputStackSpec {
    pub fn into_parts(
        self,
        expected_rows: u32,
    ) -> Result<Vec<(Range<u32>, AnyWeightMatrixSpec)>, String> {
        let mut parts = Vec::with_capacity(self.parts.len());
        let mut first_row = 0u32;
        for part in self.parts {
            let end =
                first_row.checked_add(part.rows).ok_or_else(|| "output stack row count overflows u32".to_owned())?;
            parts.push((first_row..end, part.spec));
            first_row = end;
        }
        if first_row != expected_rows {
            return Err(format!("output stack has {first_row} rows, expected {expected_rows}"));
        }
        Ok(parts)
    }
}
