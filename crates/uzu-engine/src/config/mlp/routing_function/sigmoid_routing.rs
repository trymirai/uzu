use uzu_engine_macros::uzu_config;

/// Which routing scores the expert bias is added to.
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum RoutingBias {
    /// The bias steers which experts are selected; the selected experts keep their unbiased weights.
    Gating,
    /// The bias shifts the weights of the selected experts but not their selection.
    Weights,
    /// The bias shifts both the selection and the weights.
    Both,
}

/// Sigmoid routing: every expert scores `sigmoid(logit)`, the top `num_active_routed_experts` are selected, and their
/// weights are normalized to sum to one before scaling by `routed_scaling_factor`.
#[uzu_config(super::RoutingFunction)]
pub struct SigmoidRouting {
    pub bias_mode: Option<RoutingBias>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}
