use uzu_engine_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum RoutingBias {
    Gating,
    Weights,
    Both,
}

#[uzu_config(super::RoutingFunction)]
pub struct SigmoidRouting {
    pub bias_mode: Option<RoutingBias>,
    pub weight_scale: f32,
}
