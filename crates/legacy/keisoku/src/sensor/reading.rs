use serde::{Deserialize, Serialize};

use super::SensorKind;
use crate::component::Component;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sensor {
    pub name: String,
    pub value: f64,
    pub kind: SensorKind,
    pub component: Component,
    pub manufacturer: Option<String>,
    pub category: Option<String>,
    pub location_id: Option<i64>,
    pub registry_id: u64,
}
