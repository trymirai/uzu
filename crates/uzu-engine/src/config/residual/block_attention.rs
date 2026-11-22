use uzu_engine_macros::uzu_config;

use crate::config::linear::LinearConfig;

/// Mixes a message into a sublayer input as `(1 - tanh(forget)) * input + tanh(gain) * message`, with both gates
/// low-rank projections of the RMS-normalized input and message.
#[uzu_config]
pub struct MessageGateConfig {
    pub rank: u32,
    pub linear_config: LinearConfig,
}

/// Attention residuals over blocks of `block_size` layers.
///
/// Every sublayer input is a softmax mix of the RMS-normalized inputs of the completed blocks and of the partial
/// sum of the current block, weighted by learned queries. A sublayer adds its output to the partial sum; the mixer
/// of a layer that starts a block stores the partial sum as a completed block and restarts it from its output.
/// The first layer's mixer reads the embeddings directly. In repeated passes over looped layers, each input also
/// reads a message from the previous pass and mixes it in through a message gate.
#[uzu_config(super::ResidualConfig)]
pub struct BlockAttentionResidualConfig {
    pub block_size: u32,
    pub epsilon: f32,
    pub message_gate_config: Option<MessageGateConfig>,
}
