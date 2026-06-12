//! Acceptance semantics for linear chain speculative-decoding proposals.
//!
//! Mirrors `lalamo/speculator/common.py` (`ChainProposal.accept` and
//! `AcceptedProposal`) from lalamo PR #307 at commit `2cb46e9`, restated for a
//! single sequence: lalamo's batch dimension and fixed-width padded arrays
//! (zero/`-1` sentinels plus validity masks) are JAX static-shape artifacts,
//! not part of the contract, so committed tokens here are plain vectors of
//! exactly the committed length.
//!
//! The contract separates three concepts that must not be conflated:
//!
//! - **Verified nodes** (`num_accepted_nodes`): the leading proposal nodes
//!   whose next-token prediction the target confirmed. EOS and the output
//!   budget do NOT shrink this count.
//! - **Committed tokens** (`token_ids` et al.): tokens actually emitted to the
//!   output stream. Can be fewer than verified nodes (EOS truncation,
//!   remaining output budget) and include the bonus token, which no verified
//!   node covers.
//! - **Next context nodes** ([`AcceptedProposal::accepted_node_indices`]): the
//!   nodes whose target hidden features advance the draft context state.
//!   Identical to the verified nodes for a chain proposal.
//!
//! Distinct from [`crate::trie::FlatTrie::accept`], which walks a speculation
//! tree and has none of the EOS/bonus/budget/verified-node semantics; the
//! DFlash integration should reconcile the two rather than grow both.

/// A linear chain proposal awaiting target verification.
///
/// `token_ids[0]` is the last committed token (sampled from the target but
/// never yet forward-passed through it); the remaining entries are draft
/// tokens. `token_positions[i]` is the 1-based sequence position of
/// `token_ids[i]`, using lalamo's convention where the position of the last
/// token appended to the draft context is `last_token_index + 1`
/// (`lalamo/speculator/common.py:59-64,152-155` @ `2cb46e9`).
pub struct ChainProposal {
    pub token_ids: Box<[u64]>,
    pub token_positions: Box<[usize]>,
}

/// Result of verifying a [`ChainProposal`] against the target's sampled
/// tokens (`lalamo/speculator/common.py:41-86` @ `2cb46e9`). The source node
/// of committed token `i` is always node `i` for a chain, so lalamo's
/// `source_indices` is not materialized.
pub struct AcceptedProposal {
    /// Committed tokens, in emission order.
    pub token_ids: Vec<u64>,
    /// Sequence position of each committed token.
    pub token_positions: Vec<usize>,
    /// Number of verified proposal nodes; the next context nodes are
    /// `0..num_accepted_nodes`. Unaffected by EOS or the output budget.
    pub num_accepted_nodes: usize,
}

impl ChainProposal {
    /// Verifies this proposal against the target's sampled tokens
    /// (`sampled_token_ids[i]` is the target's next token after node `i`).
    ///
    /// Mirrors `ChainProposal.accept` (`lalamo/speculator/common.py:120-185`
    /// @ `2cb46e9`); `remaining_length` is the remaining output budget
    /// (lalamo's `remaining_lengths`). Lalamo's `active_mask` only services
    /// inactive batch rows and has no single-sequence counterpart.
    pub fn accept(
        &self,
        sampled_token_ids: &[u64],
        remaining_length: usize,
        eos_token_ids: &[u64],
    ) -> AcceptedProposal {
        let num_nodes = self.token_ids.len();
        assert!(num_nodes > 0, "chain proposal must contain at least the last committed token");
        assert_eq!(self.token_positions.len(), num_nodes, "token_positions length must match token_ids");
        assert_eq!(sampled_token_ids.len(), num_nodes, "expected one sampled token per proposal node");

        let is_eos = |token: u64| eos_token_ids.contains(&token);

        // Greedy prefix match: draft token i+1 is verified iff it equals the
        // target's sample at node i (common.py:131-134).
        let num_accepted_drafts =
            (0..num_nodes - 1).take_while(|&i| self.token_ids[i + 1] == sampled_token_ids[i]).count();
        let num_accepted_nodes = num_accepted_drafts + 1;

        // Commit the verified draft tokens, truncating after the first EOS;
        // the EOS itself is kept, and a draft EOS also suppresses the bonus
        // token (common.py:141-150).
        let mut token_ids = Vec::with_capacity(num_accepted_nodes);
        let mut token_positions = Vec::with_capacity(num_accepted_nodes);
        let mut has_draft_eos = false;
        for source in 0..num_accepted_drafts {
            token_ids.push(self.token_ids[source + 1]);
            token_positions.push(self.token_positions[source + 1]);
            if is_eos(self.token_ids[source + 1]) {
                has_draft_eos = true;
                break;
            }
        }

        // Bonus token: the target's own sample at the first unverified node —
        // kept even when it is itself an EOS (common.py:152-157).
        if !has_draft_eos {
            let bonus_source = num_accepted_drafts;
            token_ids.push(sampled_token_ids[bonus_source]);
            token_positions.push(self.token_positions[bonus_source] + 1);
        }

        // Cap the committed output at the remaining budget. Lalamo applies the
        // EOS and capacity masks jointly per slot (common.py:166-170); both are
        // prefix-shaped, so truncation is equivalent.
        token_ids.truncate(remaining_length);
        token_positions.truncate(remaining_length);

        AcceptedProposal {
            token_ids,
            token_positions,
            num_accepted_nodes,
        }
    }
}

impl AcceptedProposal {
    /// Proposal nodes whose target hidden features advance the draft context
    /// state (the "next context nodes"). For a chain proposal these are
    /// always the leading nodes.
    pub fn accepted_node_indices(&self) -> std::ops::Range<usize> {
        0..self.num_accepted_nodes
    }

    /// ID of the last committed token, the anchor token for the next draft
    /// block; `None` when nothing was committed. Lalamo instead substitutes a
    /// caller-provided fallback for stopped batch rows (common.py:49-57).
    pub fn last_token_id(&self) -> Option<u64> {
        self.token_ids.last().copied()
    }

    /// Anchor index for the next draft block: the position of the last
    /// committed token minus one, i.e. the position of the last token whose
    /// features can already sit in the draft context (the bonus token was
    /// never forward-passed). Mirrors `last_token_indices`
    /// (common.py:59-64).
    pub fn last_token_index(&self) -> Option<usize> {
        if let Some(position) = self.token_positions.last() {
            debug_assert!(*position >= 1, "committed token positions are 1-based");
            Some(position - 1)
        } else {
            None
        }
    }

    /// Whether any committed token is an end-of-sequence token
    /// (common.py:66-70).
    pub fn has_eos(
        &self,
        eos_token_ids: &[u64],
    ) -> bool {
        self.token_ids.iter().any(|token| eos_token_ids.contains(token))
    }
}

#[cfg(test)]
#[path = "../../unit/speculators/chain_acceptance.rs"]
mod tests;
