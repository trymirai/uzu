use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyPowerStats {
    pub samples_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_cpu_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_gpu_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_ane_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_ram_watts: Option<f64>,
    pub average_total_watts: f64,
    pub energy_joules: f64,
}

impl ChatReplyPowerStats {
    fn per_token(
        &self,
        tokens_per_second: Option<f64>,
    ) -> Option<ChatReplyJoulesPerToken> {
        let tokens_per_second = tokens_per_second?;
        if !tokens_per_second.is_finite() || tokens_per_second <= 0.0 {
            return None;
        }

        let component_watts =
            (self.average_cpu_watts, self.average_gpu_watts, self.average_ane_watts, self.average_ram_watts);
        let energy = match component_watts {
            (Some(cpu), Some(gpu), Some(ane), Some(dram)) => ChatReplyJoulesPerToken::Components {
                cpu: cpu / tokens_per_second,
                gpu: gpu / tokens_per_second,
                ane: ane / tokens_per_second,
                dram: dram / tokens_per_second,
            },
            _ => ChatReplyJoulesPerToken::Total {
                total: self.average_total_watts / tokens_per_second,
            },
        };
        Some(energy)
    }
}

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatReplyJoulesPerToken {
    Total {
        total: f64,
    },
    Components {
        cpu: f64,
        gpu: f64,
        ane: f64,
        dram: f64,
    },
}

// The generic exporter cannot attach methods to NAPI's union representation of data enums.
#[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
impl ChatReplyJoulesPerToken {
    /// Returns total energy per token, summing the component values when present.
    pub fn total(&self) -> f64 {
        match self {
            Self::Total {
                total,
            } => *total,
            Self::Components {
                cpu,
                gpu,
                ane,
                dram,
            } => cpu + gpu + ane + dram,
        }
    }
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

    /// Average energy per uncached input token, with a component breakdown when available.
    #[bindings::export(Method(Getter))]
    pub fn input_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.power_stats.as_ref()?.per_token(self.prefill_tokens_per_second)
    }

    /// Average energy per generated output token, with a component breakdown when available.
    #[bindings::export(Method(Getter))]
    pub fn output_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.power_stats.as_ref()?.per_token(self.generate_tokens_per_second)
    }
}
