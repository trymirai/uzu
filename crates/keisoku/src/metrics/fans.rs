use serde::{Deserialize, Serialize};

use super::fan::Fan;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FanMetrics {
    pub fans: Vec<Fan>,
}
