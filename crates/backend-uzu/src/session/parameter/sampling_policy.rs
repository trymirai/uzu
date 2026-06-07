use crate::{config::model::language_model::LanguageModelConfig, session::parameter::ConfigResolvableValue};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SamplingProcessingOrder {
    TemperatureThenFilters,
    FiltersThenTemperature,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Stochastic {
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        suffix_repetition_length: Option<usize>,
        processing_order: SamplingProcessingOrder,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingPolicy {
    Default,
    Custom {
        value: SamplingMethod,
    },
}

impl ConfigResolvableValue<LanguageModelConfig, SamplingMethod> for SamplingPolicy {
    fn resolve(
        &self,
        config: &LanguageModelConfig,
    ) -> SamplingMethod {
        let generation_config = &config.generation_config;
        match self {
            SamplingPolicy::Default => SamplingMethod::Stochastic {
                temperature: generation_config.temperature,
                top_k: generation_config.top_k,
                top_p: generation_config.top_p,
                min_p: generation_config.min_p,
                repetition_penalty: generation_config.repetition_penalty,
                suffix_repetition_length: generation_config.suffix_repetition_length,
                processing_order: SamplingProcessingOrder::TemperatureThenFilters,
            },
            SamplingPolicy::Custom {
                value,
            } => *value,
        }
    }
}
