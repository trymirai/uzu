use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyStats {
    pub duration: f64,
    pub time_to_first_token: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub generate_tokens_per_second: Option<f64>,
    pub tokens_count_input: Option<u32>,
    pub tokens_count_output: Option<u32>,
}

#[bindings::export(Implementation)]
impl ChatReplyStats {
    #[bindings::export(Method(Getter))]
    pub fn tokens_count(&self) -> Option<u32> {
        self.tokens_count_input.and_then(|input| self.tokens_count_output.map(|output| input + output))
    }
}
