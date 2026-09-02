use std::{
    fmt::{self, Display},
    ops::Add,
};

use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatReplyEnergy {
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

#[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
impl ChatReplyEnergy {
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

impl ChatReplyEnergy {
    fn per_token(
        &self,
        tokens_count: u32,
    ) -> Option<ChatReplyJoulesPerToken> {
        if tokens_count == 0 {
            return None;
        }
        let tokens_count = f64::from(tokens_count);

        Some(match self {
            Self::Total {
                total,
            } => ChatReplyJoulesPerToken::Total {
                total: total / tokens_count,
            },
            Self::Components {
                cpu,
                gpu,
                ane,
                dram,
            } => ChatReplyJoulesPerToken::Components {
                cpu: cpu / tokens_count,
                gpu: gpu / tokens_count,
                ane: ane / tokens_count,
                dram: dram / tokens_count,
            },
        })
    }
}

impl Add for ChatReplyEnergy {
    type Output = Self;

    fn add(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (
                Self::Components {
                    cpu: left_cpu,
                    gpu: left_gpu,
                    ane: left_ane,
                    dram: left_dram,
                },
                Self::Components {
                    cpu: right_cpu,
                    gpu: right_gpu,
                    ane: right_ane,
                    dram: right_dram,
                },
            ) => Self::Components {
                cpu: left_cpu + right_cpu,
                gpu: left_gpu + right_gpu,
                ane: left_ane + right_ane,
                dram: left_dram + right_dram,
            },
            (left, right) => Self::Total {
                total: left.total() + right.total(),
            },
        }
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

impl Display for ChatReplyJoulesPerToken {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "CPU+GPU+DRAM {:.2} J/tok", self.total())
    }
}

#[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
impl ChatReplyJoulesPerToken {
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
    pub input_energy: Option<ChatReplyEnergy>,
    pub output_energy: Option<ChatReplyEnergy>,
}

#[bindings::export(Implementation)]
impl ChatReplyStats {
    #[bindings::export(Method(Getter))]
    pub fn tokens_count(&self) -> Option<u32> {
        self.tokens_count_input.and_then(|input| self.tokens_count_output.map(|output| input + output))
    }

    #[bindings::export(Method(Getter))]
    pub fn total_joules(&self) -> Option<f64> {
        self.input_energy.iter().chain(self.output_energy.iter()).map(ChatReplyEnergy::total).reduce(f64::add)
    }

    #[bindings::export(Method(Getter))]
    pub fn input_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.input_energy.as_ref()?.per_token(self.tokens_count_input?)
    }

    #[bindings::export(Method(Getter))]
    pub fn output_joules_per_token(&self) -> Option<ChatReplyJoulesPerToken> {
        self.output_energy.as_ref()?.per_token(self.tokens_count_output?.checked_sub(1)?)
    }
}
