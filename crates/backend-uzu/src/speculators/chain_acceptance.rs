pub struct ChainProposal {
    pub token_ids: Box<[u64]>,
    pub token_positions: Box<[usize]>,
}

pub struct AcceptedProposal {
    pub committed_token_ids: Vec<u64>,
    pub committed_token_positions: Vec<usize>,
    pub num_verified_nodes: usize,
}

impl ChainProposal {
    /// `sampled_token_ids[i]` is the target's next token after node `i`.
    pub fn accept(
        &self,
        sampled_token_ids: &[u64],
        remaining_length: usize,
        eos_token_ids: &[u64],
    ) -> AcceptedProposal {
        let num_nodes = self.token_ids.len();
        debug_assert!(num_nodes > 0, "chain proposal must contain at least the last committed token");
        debug_assert_eq!(self.token_positions.len(), num_nodes, "token_positions length must match token_ids");
        debug_assert_eq!(sampled_token_ids.len(), num_nodes, "expected one sampled token per proposal node");

        let num_verified_drafts =
            (0..num_nodes - 1).take_while(|&i| self.token_ids[i + 1] == sampled_token_ids[i]).count();
        let num_verified_nodes = num_verified_drafts + 1;

        let mut committed_token_ids = Vec::with_capacity(num_verified_nodes);
        let mut committed_token_positions = Vec::with_capacity(num_verified_nodes);
        let mut has_draft_eos = false;
        for draft in 0..num_verified_drafts {
            let token_id = self.token_ids[draft + 1];
            committed_token_ids.push(token_id);
            committed_token_positions.push(self.token_positions[draft + 1]);
            if eos_token_ids.contains(&token_id) {
                has_draft_eos = true;
                break;
            }
        }

        // A draft EOS suppresses the bonus token; a bonus EOS is kept.
        if !has_draft_eos {
            committed_token_ids.push(sampled_token_ids[num_verified_drafts]);
            committed_token_positions.push(self.token_positions[num_verified_drafts] + 1);
        }

        committed_token_ids.truncate(remaining_length);
        committed_token_positions.truncate(remaining_length);

        AcceptedProposal {
            committed_token_ids,
            committed_token_positions,
            num_verified_nodes,
        }
    }
}

impl AcceptedProposal {
    pub fn next_context_nodes(&self) -> std::ops::Range<usize> {
        0..self.num_verified_nodes
    }

    pub fn last_token_id(&self) -> Option<u64> {
        self.committed_token_ids.last().copied()
    }

    /// One behind the last committed token (the bonus, not yet forward-passed).
    pub fn last_covered_position(&self) -> Option<usize> {
        if let Some(position) = self.committed_token_positions.last() {
            debug_assert!(*position >= 1, "committed token positions are 1-based");
            Some(position - 1)
        } else {
            None
        }
    }

    pub fn has_eos(
        &self,
        eos_token_ids: &[u64],
    ) -> bool {
        self.committed_token_ids.iter().any(|token| eos_token_ids.contains(token))
    }
}

#[cfg(test)]
#[path = "../../unit/speculators/chain_acceptance.rs"]
mod tests;
