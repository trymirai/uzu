use crate::{
    config::LanguageModelConfig, session::parameter::ConfigResolvableValue,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Stochastic {
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    },
}

impl Default for SamplingMethod {
    fn default() -> Self {
        SamplingMethod::Greedy
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingPolicy {
    Default,
    Custom {
        value: SamplingMethod,
    },
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        SamplingPolicy::Default
    }
}

impl ConfigResolvableValue<LanguageModelConfig, SamplingMethod>
    for SamplingPolicy
{
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
            },
            SamplingPolicy::Custom {
                value,
            } => *value,
        }
    }
}
