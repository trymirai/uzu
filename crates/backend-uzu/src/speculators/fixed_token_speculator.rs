use std::collections::HashMap;

use itertools::Itertools;

use crate::speculators::speculator::Speculator;

pub struct FixedTokensSpeculator {
    tokens: Vec<Vec<u64>>,
}

impl FixedTokensSpeculator {
    pub fn new(tokens: Vec<Vec<u64>>) -> Self {
        Self {
            tokens,
        }
    }

    pub fn max_trie_nodes(&self) -> usize {
        self.tokens.iter().flat_map(|sequence| (0..=sequence.len()).map(|len| &sequence[..len])).unique().count()
    }
}

impl Speculator for FixedTokensSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let last_prefix_token = prefix.last().copied();
        let mut plausible_tokens = HashMap::new();

        for proposed_seq in &self.tokens {
            let mut proposal_index = 0;

            for (index, &token) in proposed_seq.iter().enumerate() {
                if last_prefix_token == Some(token) {
                    proposal_index = index + 1;
                    break;
                }
            }

            if let Some(&proposed_token) = proposed_seq.get(proposal_index).or(proposed_seq.first()) {
                *plausible_tokens.entry(proposed_token).or_insert(0.0) += 1.0
            }
        }

        let sum: f32 = plausible_tokens.values().sum();
        plausible_tokens.into_iter().map(|(k, v)| (k, v / sum)).collect()
    }
}
