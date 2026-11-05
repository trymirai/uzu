use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shoji::types::session::decision::{DecisionOption, DecisionQuestion};

// A question of the System One request of vLLM Semantic Router, shared by the server and the bench.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionRequestQuestion {
    Choice {
        instructions: Option<String>,
        // Option order is answer-code order, so it must survive parsing.
        criteria: IndexMap<String, Option<String>>,
    },
    Noul {
        instructions: Option<String>,
        criteria: Option<NoulCriteria>,
    },
    Score {
        instructions: Option<String>,
        criteria: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "false")]
    no: Option<String>,
    #[serde(rename = "true")]
    yes: Option<String>,
}

impl From<DecisionRequestQuestion> for DecisionQuestion {
    fn from(question: DecisionRequestQuestion) -> Self {
        match question {
            DecisionRequestQuestion::Choice {
                instructions,
                criteria,
            } => DecisionQuestion::Choice {
                instructions,
                options: criteria
                    .into_iter()
                    .map(|(key, description)| DecisionOption {
                        key,
                        description,
                    })
                    .collect(),
            },
            DecisionRequestQuestion::Noul {
                instructions,
                criteria,
            } => {
                let (no_description, yes_description) =
                    criteria.map_or((None, None), |criteria| (criteria.no, criteria.yes));
                DecisionQuestion::YesNo {
                    instructions,
                    no_description,
                    yes_description,
                }
            },
            DecisionRequestQuestion::Score {
                instructions,
                criteria,
            } => DecisionQuestion::Score {
                instructions,
                levels: criteria,
            },
        }
    }
}
