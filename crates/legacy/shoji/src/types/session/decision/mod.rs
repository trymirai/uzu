use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionOption {
    pub key: String,
    pub description: Option<String>,
}

impl DecisionOption {
    pub fn from(
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            description: Some(value.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionQuestion {
    Choice {
        instructions: Option<String>,
        options: Vec<DecisionOption>,
    },
    Score {
        instructions: Option<String>,
        levels: Vec<String>,
    },
    YesNo {
        instructions: Option<String>,
        no_description: Option<String>,
        yes_description: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionInput {
    pub state: String,
    pub questions: Vec<DecisionQuestion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Choice {
        index: i64,
        key: String,
        probabilities: Vec<f64>,
    },
    Score {
        probabilities: Vec<f64>,
        score: f64,
    },
    YesNo {
        yes_probability: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionStats {
    pub tokens_count: i64,
    pub total_duration: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionOutput {
    // One per question, in input order.
    pub answers: Vec<DecisionAnswer>,
    pub stats: DecisionStats,
}
