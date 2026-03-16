use crate::{config::LanguageModelConfig, session::parameter::ConfigResolvableValue};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    /// Sequential multi-kernel path: bitmask → temperature → top_k → top_p → min_p → gumbel → argmax.
    Stochastic {
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    },
    /// Unified single-pass path: all filtering and Gumbel-max in one kernel dispatch,
    /// operating on logits loaded into private registers.
    UnifiedStochastic {
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
    /// Like `Default` but forces the unified single-pass kernel path.
    DefaultUnified,
    Custom {
        value: SamplingMethod,
    },
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        SamplingPolicy::Default
    }
}

impl ConfigResolvableValue<LanguageModelConfig, SamplingMethod> for SamplingPolicy {
    fn resolve(
        &self,
        config: &LanguageModelConfig,
    ) -> SamplingMethod {
        let g = &config.generation_config;
        match self {
            SamplingPolicy::Default => SamplingMethod::Stochastic {
                temperature: g.temperature,
                top_k: g.top_k,
                top_p: g.top_p,
                min_p: g.min_p,
            },
            SamplingPolicy::DefaultUnified => SamplingMethod::UnifiedStochastic {
                temperature: g.temperature,
                top_k: g.top_k,
                top_p: g.top_p,
                min_p: g.min_p,
            },
            SamplingPolicy::Custom {
                value,
            } => *value,
        }
    }
}
