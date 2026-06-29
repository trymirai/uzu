//! Sampling-mode selection and its mapping to a uzu `SamplingMethod`.

use uzu::types::basic::SamplingMethod;

/// Sampling mode (mirrors ui-kit). `Default` uses the model's own config.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SamplingMode {
    Default,
    Argmax,
    Stochastic,
}

/// Map the UI sampling mode + params to a uzu `SamplingMethod`. `Default`
/// leaves it to the model's own config (`None`); a param of 0 means "off".
pub(super) fn sampling_method(
    mode: SamplingMode,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
) -> Option<SamplingMethod> {
    match mode {
        SamplingMode::Default => None,
        SamplingMode::Argmax => Some(SamplingMethod::Greedy {}),
        SamplingMode::Stochastic => Some(SamplingMethod::Stochastic {
            temperature: Some(temperature as f64),
            top_k: (top_k > 0).then_some(top_k as i64),
            top_p: (top_p > 0.0).then_some(top_p as f64),
            min_p: (min_p > 0.0).then_some(min_p as f64),
            repetition_penalty: None,
            suffix_repetition_length: None,
        }),
    }
}
