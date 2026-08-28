use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyPowerStats {
    pub samples_count: i64,
    pub average_cpu_watts: f64,
    pub average_gpu_watts: f64,
    pub average_ane_watts: f64,
    pub average_ram_watts: f64,
    pub average_total_watts: f64,
    pub energy_joules: f64,
}

impl ChatReplyPowerStats {
    fn joules_per_token(
        &self,
        tokens_per_second: Option<f64>,
    ) -> Option<ChatReplyJoulesPerToken> {
        let tokens_per_second = tokens_per_second?;
        if !tokens_per_second.is_finite() || tokens_per_second <= 0.0 {
            return None;
        }

        Some(ChatReplyJoulesPerToken {
            cpu: self.average_cpu_watts / tokens_per_second,
            gpu: self.average_gpu_watts / tokens_per_second,
            ane: self.average_ane_watts / tokens_per_second,
            dram: self.average_ram_watts / tokens_per_second,
            combined: self.average_total_watts / tokens_per_second,
        })
    }
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyJoulesPerToken {
    pub cpu: f64,
    pub gpu: f64,
    pub ane: f64,
    pub dram: f64,
    pub combined: f64,
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplySpeculatorStats {
    pub tokens_per_forward_pass: f64,
    pub num_decode_forward_passes: u32,
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyStats {
    pub duration: f64,
    pub time_to_first_token: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub generate_tokens_per_second: Option<f64>,
    pub tokens_count_input: Option<u32>,
    pub tokens_count_input_cached: Option<u32>,
    pub tokens_count_output: Option<u32>,
    pub memory_used_bytes: Option<i64>,
    pub speculator_stats: Option<ChatReplySpeculatorStats>,
    pub power_stats: Option<ChatReplyPowerStats>,
}

#[bindings::export(Implementation)]
impl ChatReplyStats {
    #[bindings::export(Method(Getter))]
    pub fn tokens_count(&self) -> Option<u32> {
        self.tokens_count_input.and_then(|input| self.tokens_count_output.map(|output| input + output))
    }

    /// Energy spent per processed token, counting both input and output tokens.
    #[bindings::export(Method(Getter))]
    pub fn joules_per_token(&self) -> Option<f64> {
        let energy_joules = self.power_stats.as_ref()?.energy_joules;
        let tokens_count = self.tokens_count()?;
        (tokens_count > 0).then(|| energy_joules / f64::from(tokens_count))
    }

    /// Average energy per uncached input token, split by hardware component.
    #[bindings::export(Method(Getter))]
    pub fn input_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.power_stats.as_ref()?.joules_per_token(self.prefill_tokens_per_second)
    }

    /// Average energy per generated output token, split by hardware component.
    #[bindings::export(Method(Getter))]
    pub fn output_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.power_stats.as_ref()?.joules_per_token(self.generate_tokens_per_second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats() -> ChatReplyStats {
        ChatReplyStats {
            prefill_tokens_per_second: Some(20.0),
            generate_tokens_per_second: Some(5.0),
            power_stats: Some(ChatReplyPowerStats {
                average_cpu_watts: 10.0,
                average_gpu_watts: 20.0,
                average_ane_watts: 5.0,
                average_ram_watts: 4.0,
                average_total_watts: 39.0,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn reports_component_joules_per_input_and_output_token() {
        let stats = stats();

        assert_eq!(
            stats.input_joules_per_token(),
            Some(ChatReplyJoulesPerToken {
                cpu: 0.5,
                gpu: 1.0,
                ane: 0.25,
                dram: 0.2,
                combined: 1.95,
            })
        );
        assert_eq!(
            stats.output_joules_per_token(),
            Some(ChatReplyJoulesPerToken {
                cpu: 2.0,
                gpu: 4.0,
                ane: 1.0,
                dram: 0.8,
                combined: 7.8,
            })
        );
    }

    #[test]
    fn omits_component_joules_per_token_for_invalid_throughput() {
        for tokens_per_second in [None, Some(0.0), Some(-1.0), Some(f64::INFINITY), Some(f64::NAN)] {
            let mut stats = stats();
            stats.prefill_tokens_per_second = tokens_per_second;

            assert_eq!(stats.input_joules_per_token(), None);
        }
    }
}
