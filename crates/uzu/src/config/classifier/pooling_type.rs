use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum PoolingType {
    /// [CLS] token — take the first token's hidden state as the pooled output.
    Cls,
    /// Mean over the sequence axis.
    Mean,
    /// No pooling — the prediction head runs per-token. Used by token-level
    /// classifiers (e.g. BIOES tag sequences, PII span taggers).
    None,
}
