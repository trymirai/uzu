use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum PublicProjection {
    #[default]
    None,
    LockedByOther(String),
    StickyError(String),
}
