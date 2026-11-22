use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct SeparableCausalConvConfig {
    pub has_biases: bool,
}

/// Stateful depthwise causal convolution computed in float32. `weights[channel][kernel_size - 1]` weights the
/// current token. With `num_dynamic_groups`, every token adds its own taps, shared by contiguous channel groups.
/// The identity adds the input before the output scale.
#[uzu_config]
pub struct CausalConvConfig {
    pub kernel_size: u32,
    pub num_dynamic_groups: Option<u32>,
    pub has_identity: bool,
    pub output_scale: Option<f32>,
}
