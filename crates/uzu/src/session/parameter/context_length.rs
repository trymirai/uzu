use crate::{
    config::LanguageModelConfig, session::parameter::ConfigResolvableValue,
};

#[derive(Debug, Clone, Copy)]
pub enum ContextLength {
    Default,
    Maximal,
    Custom(usize),
}

impl Default for ContextLength {
    fn default() -> Self {
        ContextLength::Default
    }
}

impl ConfigResolvableValue<LanguageModelConfig, usize> for ContextLength {
    fn resolve(
        &self,
        config: &LanguageModelConfig,
    ) -> usize {
        let model_context_length = config.decoder_config.context_length;
        let proposed_value = match self {
            ContextLength::Default => {
                if cfg!(target_os = "ios") {
                    return 8192;
                } else {
                    return 16384;
                }
            },
            ContextLength::Maximal => model_context_length,
            ContextLength::Custom(value) => *value,
        };
        std::cmp::min(proposed_value, model_context_length)
    }
}
