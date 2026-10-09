use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct MetalAstType {
    #[serde(rename = "qualType")]
    pub qual_type: Box<str>,
    #[serde(rename = "desugaredQualType")]
    pub desugared_qual_type: Option<Box<str>>,
}
