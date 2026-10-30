use indexmap::IndexMap;
use serde::Serialize;
use shoji::types::session::decision::{DecisionAnswer, DecisionOutput, DecisionQuestion};

#[derive(Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: IndexMap<String, DecisionResponseAnswer>,
    pub usage: DecisionResponseUsage,
}

#[derive(Serialize)]
pub struct DecisionResponseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

// The answer objects of the reference `d3_runtime.product_answer`.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionResponseAnswer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
    Score {
        score: f64,
        // Keyed by level index, as strings.
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        legend: IndexMap<String, String>,
    },
}

impl DecisionResponse {
    pub fn new(
        model: String,
        ids: Vec<String>,
        questions: &[DecisionQuestion],
        output: DecisionOutput,
    ) -> Self {
        let input_tokens = output.results.iter().map(|result| result.stats.tokens_count).sum();
        let answers = ids
            .into_iter()
            .zip(questions)
            .zip(output.results)
            .map(|((id, question), result)| (id, DecisionResponseAnswer::new(question, result.answer)))
            .collect();
        Self {
            model,
            answers,
            usage: DecisionResponseUsage {
                input_tokens,
                output_tokens: 0,
            },
        }
    }
}

impl DecisionResponseAnswer {
    fn new(
        question: &DecisionQuestion,
        answer: DecisionAnswer,
    ) -> Self {
        match (question, answer) {
            (
                DecisionQuestion::Choice {
                    options,
                    ..
                },
                DecisionAnswer::Choice {
                    key,
                    probabilities,
                    ..
                },
            ) => DecisionResponseAnswer::Choice {
                choice: key,
                confidence: confidence(&probabilities),
                probabilities: options.iter().map(|option| option.key.clone()).zip(probabilities).collect(),
            },
            (
                DecisionQuestion::Score {
                    levels,
                    ..
                },
                DecisionAnswer::Score {
                    probabilities,
                    score,
                },
            ) => DecisionResponseAnswer::Score {
                score,
                confidence: confidence(&probabilities),
                probabilities: probabilities
                    .into_iter()
                    .enumerate()
                    .map(|(index, probability)| (index.to_string(), probability))
                    .collect(),
                legend: levels.iter().enumerate().map(|(index, level)| (index.to_string(), level.clone())).collect(),
            },
            (
                DecisionQuestion::YesNo {
                    ..
                },
                DecisionAnswer::YesNo {
                    yes_probability,
                },
            ) => DecisionResponseAnswer::Noul {
                noul: yes_probability,
            },
            _ => unreachable!("answers follow the question types"),
        }
    }
}

// One minus the normalized Shannon entropy, as the reference `d3_runtime._confidence`.
fn confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f64>();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}
