use serde::{Deserialize, Serialize};

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub values: Vec<String>,
}

#[bindings::export(Implementation)]
impl Feature {
    #[bindings::export(Constructor)]
    pub fn new(
        name: String,
        values: Vec<String>,
    ) -> Self {
        Self {
            name,
            values,
        }
    }
}
