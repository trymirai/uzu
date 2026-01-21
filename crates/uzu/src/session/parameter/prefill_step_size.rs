use crate::{
    config::LanguageModelConfig, session::parameter::ConfigResolvableValue,
};

fn env_prefill_step_size_default_override() -> Option<usize> {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> =
        std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("UZU_PREFILL_STEP_SIZE").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        trimmed.parse::<usize>().ok()
    })
}

#[derive(Debug, Clone, Copy)]
pub enum PrefillStepSize {
    Default,
    Maximal,
    Custom(usize),
}

impl Default for PrefillStepSize {
    fn default() -> Self {
        PrefillStepSize::Default
    }
}

impl ConfigResolvableValue<LanguageModelConfig, usize> for PrefillStepSize {
    fn resolve(
        &self,
        config: &LanguageModelConfig,
    ) -> usize {
        let default_limit: usize = 4096;
        let model_context_length =
            config.model_config.transformer_config.context_length;

        let minimal_sliding_window_size = config
            .decoder_config()
            .ok()
            .and_then(|dc| dc.sliding_window_sizes)
            .as_deref()
            .into_iter()
            .flatten()
            .flatten()
            .copied()
            .min()
            .unwrap_or(usize::MAX);

        let maximal_value =
            [model_context_length, minimal_sliding_window_size, default_limit]
                .into_iter()
                .min()
                .unwrap_or(default_limit);

        let proposed_value = match self {
            PrefillStepSize::Default => {
                if let Some(v) = env_prefill_step_size_default_override() {
                    v
                } else if cfg!(target_os = "ios") {
                    512
                } else {
                    2048
                }
            },
            PrefillStepSize::Maximal => maximal_value,
            PrefillStepSize::Custom(value) => *value,
        };
        std::cmp::min(proposed_value, maximal_value)
    }
}
