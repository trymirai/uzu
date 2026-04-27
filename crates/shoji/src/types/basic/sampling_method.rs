use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingMethod {
    Greedy {},
    Stochastic {
        temperature: Option<f64>,
        top_k: Option<i64>,
        top_p: Option<f64>,
        min_p: Option<f64>,
    },
}
