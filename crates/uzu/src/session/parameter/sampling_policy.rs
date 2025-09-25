use crate::{
    config::LanguageModelConfig, session::parameter::ConfigResolvableValue,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Temperature {
        temperature: f32,
    },
    TopP {
        top_p: f32,
    },
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
            SamplingPolicy::Default => {
                if let Some(top_p) = generation_config.top_p {
                    return SamplingMethod::TopP {
                        top_p: top_p,
                    };
                }
                if let Some(temperature) = generation_config.temperature {
                    return SamplingMethod::Temperature {
                        temperature: temperature,
                    };
                }
                return SamplingMethod::Greedy;
            },
            SamplingPolicy::Custom {
                value,
            } => *value,
        }
    }
}
