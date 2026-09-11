use json_transform::TransformSchema;
use serde::{Deserialize, Serialize};

use crate::{framing::FramingParserConfig, reduction::ReductionParserConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenStreamParserConfig {
    #[serde(default)]
    pub framing: Option<FramingParserConfig>,
    pub reduction: ReductionParserConfig,
    pub transformation: TransformSchema,
}

impl TokenStreamParserConfig {
    pub fn framing_config(&self) -> FramingParserConfig {
        self.framing.clone().unwrap_or_else(|| FramingParserConfig {
            tokens: self.reduction.collect_framing_tokens(),
        })
    }
}
