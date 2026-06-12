//! Acceptance semantics for linear chain speculative-decoding proposals.
//!
//! Single-sequence port of `lalamo/speculator/common.py`
//! (`ChainProposal.accept`, `AcceptedProposal`) from lalamo PR #307 at commit
//! `2cb46e9`. Lalamo's batch dimension and fixed-width padded arrays
//! (sentinels plus validity masks) are JAX static-shape artifacts, not
//! contract, and are dropped. Names follow the integration plan's explicit
//! vocabulary instead of lalamo's: `num_accepted_nodes` ->
//! `num_verified_nodes`, `accepted_node_indices` -> `next_context_nodes`,
//! committed `token_ids`/`token_positions` -> `committed_token_ids`/
//! `committed_token_positions`.
//!
//! Verified nodes advance the draft context state; committed tokens are what
//! the session emits. The two counts differ (EOS truncation, output budget,
//! the bonus token) and must never be conflated.
//!
//! Distinct from [`crate::trie::FlatTrie::accept`], which walks a speculation
//! tree and has none of the EOS/bonus/budget/verified-node semantics; the
//! DFlash integration should reconcile the two rather than grow both.

/// A linear chain proposal awaiting target verification.
///
/// `token_ids[0]` is the last committed token (sampled from the target but
/// never yet forward-passed through it); the rest are draft tokens.
/// `token_positions[i]` is the 1-based sequence position of `token_ids[i]`
/// (`common.py:59-64,152-155` @ `2cb46e9`).
pub struct ChainProposal {
    pub token_ids: Box<[u64]>,
    pub token_positions: Box<[usize]>,
}

/// Result of verifying a [`ChainProposal`] against the target's sampled
/// tokens (`common.py:41-86` @ `2cb46e9`). The source node of committed token
/// `i` is always node `i` for a chain, so lalamo's `source_indices` is not
/// materialized.
pub struct AcceptedProposal {
    /// Committed tokens, in emission order.
    pub committed_token_ids: Vec<u64>,
    /// 1-based sequence position of each committed token.
    pub committed_token_positions: Vec<usize>,
    /// Leading proposal nodes whose next-token prediction the target
    /// confirmed — unaffected by EOS or the output budget, unlike the
    /// committed tokens.
    pub num_verified_nodes: usize,
}

impl ChainProposal {
    /// Verifies this proposal against the target's sampled tokens
    /// (`sampled_token_ids[i]` is the target's next token after node `i`);
    /// `remaining_length` is the output budget. Mirrors `ChainProposal.accept`
    /// (`common.py:120-185` @ `2cb46e9`) minus `active_mask`, which only
    /// services inactive batch rows.
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

        // common.py:131-134
        let num_verified_drafts =
            (0..num_nodes - 1).take_while(|&i| self.token_ids[i + 1] == sampled_token_ids[i]).count();
        let num_verified_nodes = num_verified_drafts + 1;

        // A draft EOS ends the committed output (the EOS itself is kept) and
        // suppresses the bonus token (common.py:141-150).
        let mut committed_token_ids = Vec::with_capacity(num_verified_nodes);
        let mut committed_token_positions = Vec::with_capacity(num_verified_nodes);
        let mut has_draft_eos = false;
        for draft in 0..num_verified_drafts {
            committed_token_ids.push(self.token_ids[draft + 1]);
            committed_token_positions.push(self.token_positions[draft + 1]);
            if is_eos(self.token_ids[draft + 1]) {
                has_draft_eos = true;
                break;
            }
        }

        // Bonus token: the target's own sample at the first unverified node —
        // committed even when it is itself an EOS (common.py:152-157).
        if !has_draft_eos {
            let bonus_source = num_verified_drafts;
            committed_token_ids.push(sampled_token_ids[bonus_source]);
            committed_token_positions.push(self.token_positions[bonus_source] + 1);
        }

        // Lalamo applies the EOS and budget masks jointly per slot
        // (common.py:166-170); both are prefix-shaped, so truncation is
        // equivalent.
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
    /// Proposal nodes whose target hidden features advance the draft context
    /// state — always the leading nodes for a chain.
    pub fn next_context_nodes(&self) -> std::ops::Range<usize> {
        0..self.num_verified_nodes
    }

    /// ID of the last committed token, the anchor token for the next draft
    /// block; `None` when nothing was committed (lalamo substitutes a
    /// caller-provided fallback for stopped batch rows, `common.py:49-57`).
    pub fn last_token_id(&self) -> Option<u64> {
        self.committed_token_ids.last().copied()
    }

    /// Anchor index for the next draft block: the position of the last
    /// committed token minus one, i.e. the last position whose features can
    /// already sit in the draft context — the bonus token was never
    /// forward-passed (`common.py:59-64`).
    pub fn last_token_index(&self) -> Option<usize> {
        if let Some(position) = self.committed_token_positions.last() {
            debug_assert!(*position >= 1, "committed token positions are 1-based");
            Some(position - 1)
        } else {
            None
        }
    }

    /// Whether any committed token is an end-of-sequence token
    /// (`common.py:66-70`).
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
