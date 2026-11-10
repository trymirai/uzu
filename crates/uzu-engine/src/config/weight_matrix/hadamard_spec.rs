use uzu_engine_macros::uzu_config;

use super::IncoherenceProcessingMode;

#[uzu_config]
pub struct HadamardSpec {
    #[serde(rename = "type")]
    ty: monostate::MustBe!("HadamardSpec"),
    pub block_size: u32,
    pub processing_mode: IncoherenceProcessingMode,
}

impl HadamardSpec {
    pub fn new(
        block_size: u32,
        processing_mode: IncoherenceProcessingMode,
    ) -> Self {
        Self {
            ty: Default::default(),
            block_size,
            processing_mode,
        }
    }
}
