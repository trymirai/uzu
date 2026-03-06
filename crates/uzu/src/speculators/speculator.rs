use std::collections::HashMap;

use crate::trie::TrieCreationConfig;

pub trait Speculator: Send + Sync {
    /// Call once per trie build before any `speculate` calls.
    /// Stateful speculators (e.g. PARD) use this to run their forward pass and
    /// prime the cache; stateless speculators ignore it.
    fn prepare(
        &self,
        _prefix: &[u64],
    ) {
    }

    /// Returns the probability distribution over next tokens given `prefix`.
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32>;

    /// Returns the trie creation configuration for this speculator.
    ///
    /// N-gram and empty speculators use the default (width=4) to allow multiple
    /// candidates per position. Neural (PARD) speculators must return width=1 to
    /// produce a trunk-only chain: root → s0 → s1 → … so that the main model
    /// can verify the full speculation depth in a single forward pass.
    fn trie_creation_config(&self) -> TrieCreationConfig {
        TrieCreationConfig::default()
    }
}
