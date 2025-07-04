#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingConfig {
    Argmax,
    TopP {
        top_p: f32,
    },
    Categorical {
        temperature: f32,
    },
}

impl SamplingConfig {
    pub fn argmax() -> Self {
        Self::Argmax
    }

    pub fn top_p(top_p: f32) -> Self {
        Self::TopP {
            top_p,
        }
    }

    pub fn categorical(temperature: f32) -> Self {
        Self::Categorical {
            temperature,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::Argmax
    }
}
