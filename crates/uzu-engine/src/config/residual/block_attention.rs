use uzu_engine_macros::uzu_config;

use crate::config::linear::LinearConfig;

#[uzu_config(super::ResidualConfig)]
pub struct BlockAttnResConfig {
    pub epsilon: f32,
}

#[uzu_config]
pub struct MessageGateConfig {
    pub linear_config: LinearConfig,
    pub rank: u32,
}

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum QKVReadMode {
    Partial,
    Shared,
    Separate,
}

#[uzu_config(super::ConnectionConfig)]
pub struct AttnResReadConfig {
    pub message_gate_config: Option<MessageGateConfig>,
}

#[uzu_config(super::ConnectionConfig)]
pub struct AttnResQKVReadConfig {
    pub read_mode: QKVReadMode,
    pub starts_block: bool,
    pub message_gate_config: Option<MessageGateConfig>,
}
