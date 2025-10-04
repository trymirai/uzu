use crate::{
    config::LanguageModelConfig, session::parameter::ConfigResolvableValue,
};

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
        let default_limit: usize = 1024;
        let model_context_length = config.decoder_config.context_length;
        let minimal_sliding_window_size = config
            .decoder_config
            .sliding_window_sizes
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
                if cfg!(target_os = "ios") {
                    return 1;
                } else {
                    return 1;
                }
            },
            PrefillStepSize::Maximal => maximal_value,
            PrefillStepSize::Custom(value) => *value,
        };
        std::cmp::min(proposed_value, maximal_value)
    }
}
