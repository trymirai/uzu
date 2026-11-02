use indexmap::IndexMap;
use serde::Deserialize;
use shoji::types::session::decision::DecisionInput;

use crate::common::decision::DecisionRequestQuestion;

// The System One request of vLLM Semantic Router and d3's `d3_server.py`, text only:
// `state` must be a string and there are no `images` or `videos`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    // Accepted and not used: replies name the served model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    // Raw values, so a malformed question is reported under its id.
    pub questions: IndexMap<String, serde_json::Value>,
}

pub struct InvalidQuestion {
    pub id: String,
    pub message: String,
}

impl DecisionRequest {
    // The question ids in request order, and the input that decides them in the same order.
    pub fn into_input(self) -> Result<(Vec<String>, DecisionInput), InvalidQuestion> {
        let mut ids = Vec::with_capacity(self.questions.len());
        let mut questions = Vec::with_capacity(self.questions.len());
        for (id, question) in self.questions {
            let question =
                serde_json::from_value::<DecisionRequestQuestion>(question).map_err(|error| InvalidQuestion {
                    id: id.clone(),
                    message: error.to_string(),
                })?;
            ids.push(id);
            questions.push(question.into());
        }
        let input = DecisionInput {
            state: self.state.unwrap_or_default(),
            questions,
        };
        Ok((ids, input))
    }
}
