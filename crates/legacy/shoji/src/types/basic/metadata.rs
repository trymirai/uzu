use serde::{Deserialize, Serialize};

use crate::types::basic::Image;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Metadata {
    #[serde(rename = "id")]
    pub identifier: String,
    pub name: String,
    pub description: Option<String>,
    pub icons: Vec<Image>,
}

#[bindings::export(Implementation)]
impl Metadata {
    #[bindings::export(Method(Factory))]
    pub fn external(name: String) -> Self {
        Self {
            identifier: "external".to_string(),
            name,
            description: None,
            icons: vec![],
        }
    }
}
