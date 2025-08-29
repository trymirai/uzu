pub struct GeneratorEncodedTask {
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct GeneratorRunTask {
    pub token_ids: Vec<u64>,
    pub token_positions: Vec<usize>,
    pub expected_amount_of_new_tokens: usize,
}

impl GeneratorRunTask {
    pub fn speculate_next_task(&self) -> Self {
        GeneratorRunTask {
            token_ids: self.token_ids.clone(),
            token_positions: self.token_positions.clone(),
            expected_amount_of_new_tokens: self.expected_amount_of_new_tokens,
        }
    }

    pub fn encoded_task_key(
        &self,
        tokens_count: usize,
    ) -> String {
        format!(
            "tokens:{}_suffix:{}",
            tokens_count, self.expected_amount_of_new_tokens
        )
    }
}
