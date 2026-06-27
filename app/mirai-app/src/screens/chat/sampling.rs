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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mode_leaves_sampling_to_model() {
        assert!(sampling_method(SamplingMode::Default, 0.7, 0, 0.0, 0.0).is_none());
    }

    #[test]
    fn argmax_mode_is_greedy() {
        assert!(matches!(
            sampling_method(SamplingMode::Argmax, 0.7, 40, 0.9, 0.1),
            Some(SamplingMethod::Greedy {})
        ));
    }

    #[test]
    fn stochastic_zero_params_are_off() {
        match sampling_method(SamplingMode::Stochastic, 0.7, 0, 0.0, 0.0) {
            Some(SamplingMethod::Stochastic { temperature, top_k, top_p, min_p, .. }) => {
                assert_eq!(temperature, Some(0.7f32 as f64));
                assert_eq!(top_k, None);
                assert_eq!(top_p, None);
                assert_eq!(min_p, None);
            }
            _ => panic!("expected stochastic"),
        }
    }

    #[test]
    fn stochastic_nonzero_params_pass_through() {
        match sampling_method(SamplingMode::Stochastic, 0.8, 40, 0.9, 0.05) {
            Some(SamplingMethod::Stochastic { top_k, top_p, min_p, .. }) => {
                assert_eq!(top_k, Some(40));
                assert_eq!(top_p, Some(0.9f32 as f64));
                assert!(min_p.unwrap() > 0.0);
            }
            _ => panic!("expected stochastic"),
        }
    }
}
