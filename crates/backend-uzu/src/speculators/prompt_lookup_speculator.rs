use std::collections::HashMap;

use crate::speculators::{speculator::Speculator, token_finder::TokenFinder};

pub struct PromptLookupSpeculator {
    max_ngram_size: usize,
}

impl Default for PromptLookupSpeculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptLookupSpeculator {
    pub fn new() -> Self {
        Self {
            max_ngram_size: 3,
        }
    }

    pub fn new_with_params(max_ngram_size: usize) -> Self {
        Self {
            max_ngram_size,
        }
    }
}

impl Speculator for PromptLookupSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let mut hm = HashMap::new();

        if let Some(tok) = TokenFinder::find_candidate_pred_token(prefix, self.max_ngram_size) {
            hm.insert(tok, 1.0);
        }

        hm
    }
}
