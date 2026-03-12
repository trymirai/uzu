use std::collections::HashMap;

use xxhash_rust::xxh3::Xxh3;

use crate::{
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    forward_pass::{
        cache_layers::{CacheLayer, CacheLayers},
        kv_cache_layer::KVCacheLayerState,
    },
};

/// Incremental hash of a token sequence, producing a per-position digest.
/// `hashes[i]` = xxh3 of `tokens[0..=i]`, enabling O(1) prefix-length comparison.
pub struct PrefixCacheKey {
    hashes: Vec<u64>,
}

impl PrefixCacheKey {
    pub fn from_tokens(tokens: &[u64]) -> Self {
        let mut hasher = Xxh3::new();
        let hashes = tokens
            .iter()
            .map(|token| {
                hasher.update(&token.to_le_bytes());
                hasher.digest()
            })
            .collect();
        Self { hashes }
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub fn hash_at(&self, position: usize) -> u64 {
        self.hashes[position]
    }

    /// Number of leading positions where both keys have identical hashes.
    pub fn common_prefix_len(&self, other: &PrefixCacheKey) -> usize {
        self.hashes
            .iter()
            .zip(other.hashes.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_memory_bytes: usize,
    pub min_prefix_len: usize,
    pub max_entries: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            min_prefix_len: 32,
            max_entries: 16,
        }
    }
}

pub struct KVLayerSnapshot<B: Backend> {
    pub keys: Array<B>,
    pub values: Array<B>,
    pub prefix_token_positions: Vec<usize>,
}

pub struct PrefixCacheEntry<B: Backend> {
    pub tokens: Vec<u64>,
    pub key: PrefixCacheKey,
    pub layer_snapshots: Vec<Option<KVLayerSnapshot<B>>>,
    pub prefix_len: usize,
    pub memory_bytes: usize,
    last_access: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub restored_tokens: u64,
    pub computed_tokens: u64,
    pub evictions: u64,
    pub total_memory_bytes: usize,
}

pub struct PrefixCacheStore<B: Backend> {
    entries: HashMap<u64, PrefixCacheEntry<B>>,
    config: PrefixCacheConfig,
    stats: PrefixCacheStats,
    access_counter: u64,
    current_memory_bytes: usize,
}

impl<B: Backend> PrefixCacheStore<B> {
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: PrefixCacheStats::default(),
            access_counter: 0,
            current_memory_bytes: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn config(&self) -> &PrefixCacheConfig {
        &self.config
    }

    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    pub fn next_access_counter(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Find the entry whose key shares the longest common prefix with `key`,
    /// provided the match length is at least `min_prefix_len`.
    pub fn find_longest_match(&mut self, key: &PrefixCacheKey) -> Option<(u64, usize)> {
        let min_len = self.config.min_prefix_len;
        let best = self
            .entries
            .iter()
            .map(|(hash, entry)| (*hash, key.common_prefix_len(&entry.key)))
            .filter(|(_, common)| *common >= min_len)
            .max_by_key(|(_, common)| *common);

        if let Some((hash, _)) = best {
            self.access_counter += 1;
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.last_access = self.access_counter;
            }
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        best
    }

    pub fn get(&self, hash: &u64) -> Option<&PrefixCacheEntry<B>> {
        self.entries.get(hash)
    }

    pub fn insert(&mut self, entry: PrefixCacheEntry<B>) {
        self.evict_if_needed(entry.memory_bytes);
        let hash = entry.key.hash_at(entry.key.len() - 1);
        self.current_memory_bytes += entry.memory_bytes;
        self.stats.total_memory_bytes = self.current_memory_bytes;
        self.entries.insert(hash, entry);
    }

    fn evict_if_needed(&mut self, incoming_bytes: usize) {
        while !self.entries.is_empty()
            && (self.entries.len() >= self.config.max_entries
                || self.current_memory_bytes + incoming_bytes > self.config.max_memory_bytes)
        {
            let lru_hash = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(h, _)| *h)
                .unwrap();
            if let Some(removed) = self.entries.remove(&lru_hash) {
                self.current_memory_bytes -= removed.memory_bytes;
                self.stats.evictions += 1;
            }
        }
    }

    /// Snapshot active KV cache into a new entry.
    pub fn snapshot_from_cache(
        context: &B::Context,
        cache_layers: &CacheLayers<B>,
        tokens: &[u64],
        key: PrefixCacheKey,
        access_counter: u64,
    ) -> PrefixCacheEntry<B> {
        let mut layer_snapshots = Vec::with_capacity(cache_layers.data.len());
        let mut memory_bytes = 0usize;

        for (layer_index, layer) in cache_layers.data.iter().enumerate() {
            match layer {
                CacheLayer::Transformer(kv) => {
                    let prefix_len = match &kv.state {
                        KVCacheLayerState::Full { prefix_len } => *prefix_len,
                        KVCacheLayerState::Windowed { .. } => {
                            layer_snapshots.push(None);
                            continue;
                        },
                    };
                    if prefix_len == 0 {
                        layer_snapshots.push(None);
                        continue;
                    }

                    let keys_borrow = kv.keys.borrow();
                    let shape = keys_borrow.shape();
                    let num_groups = shape[0];
                    let head_dim = shape[2];
                    let dtype = keys_borrow.data_type();

                    let snap_shape = [num_groups, prefix_len, head_dim];
                    let mut snap_keys = context.create_array(
                        &snap_shape,
                        dtype,
                        &format!("prefix_cache_snap_keys_{layer_index}"),
                    );
                    snap_keys.copy_slice(&keys_borrow, 1, 0..prefix_len, 0);
                    drop(keys_borrow);

                    let values_borrow = kv.values.borrow();
                    let mut snap_values = context.create_array(
                        &snap_shape,
                        dtype,
                        &format!("prefix_cache_snap_values_{layer_index}"),
                    );
                    snap_values.copy_slice(&values_borrow, 1, 0..prefix_len, 0);
                    drop(values_borrow);

                    memory_bytes += snap_keys.size() + snap_values.size();
                    let positions = kv.prefix_token_positions[..prefix_len].to_vec();
                    layer_snapshots.push(Some(KVLayerSnapshot {
                        keys: snap_keys,
                        values: snap_values,
                        prefix_token_positions: positions,
                    }));
                },
                _ => {
                    layer_snapshots.push(None);
                },
            }
        }

        let prefix_len = tokens.len();
        PrefixCacheEntry {
            tokens: tokens.to_vec(),
            key,
            layer_snapshots,
            prefix_len,
            memory_bytes,
            last_access: access_counter,
        }
    }

    /// Restore a cached entry into the active KV cache layers.
    pub fn restore_to_cache(
        entry: &PrefixCacheEntry<B>,
        cache_layers: &mut CacheLayers<B>,
        restore_len: usize,
    ) {
        for (layer, snapshot_opt) in cache_layers.data.iter_mut().zip(entry.layer_snapshots.iter()) {
            if let (CacheLayer::Transformer(kv), Some(snapshot)) = (layer, snapshot_opt) {
                let available = snapshot.keys.shape()[1];
                let len = restore_len.min(available);

                {
                    let mut dst_keys = kv.keys.borrow_mut();
                    dst_keys.copy_slice(&snapshot.keys, 1, 0..len, 0);
                }
                {
                    let mut dst_values = kv.values.borrow_mut();
                    dst_values.copy_slice(&snapshot.values, 1, 0..len, 0);
                }

                kv.state = KVCacheLayerState::Full { prefix_len: len };
                kv.prefix_token_positions = snapshot.prefix_token_positions[..len].to_vec();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_deterministic() {
        let tokens = vec![1u64, 2, 3, 4, 5];
        let k1 = PrefixCacheKey::from_tokens(&tokens);
        let k2 = PrefixCacheKey::from_tokens(&tokens);
        assert_eq!(k1.len(), 5);
        for i in 0..5 {
            assert_eq!(k1.hash_at(i), k2.hash_at(i));
        }
    }

    #[test]
    fn test_key_prefix_independent() {
        let k1 = PrefixCacheKey::from_tokens(&[10, 20, 30, 40, 50]);
        let k2 = PrefixCacheKey::from_tokens(&[10, 20, 30, 99, 100]);
        // First 3 positions must match.
        for i in 0..3 {
            assert_eq!(k1.hash_at(i), k2.hash_at(i));
        }
        // Position 3 onward must differ.
        assert_ne!(k1.hash_at(3), k2.hash_at(3));
        assert_ne!(k1.hash_at(4), k2.hash_at(4));
    }

    #[test]
    fn test_common_prefix_len_exact() {
        let k1 = PrefixCacheKey::from_tokens(&[1, 2, 3]);
        let k2 = PrefixCacheKey::from_tokens(&[1, 2, 3]);
        assert_eq!(k1.common_prefix_len(&k2), 3);
    }

    #[test]
    fn test_common_prefix_len_partial() {
        let k1 = PrefixCacheKey::from_tokens(&[1, 2, 3, 4]);
        let k2 = PrefixCacheKey::from_tokens(&[1, 2, 9, 10]);
        assert_eq!(k1.common_prefix_len(&k2), 2);
    }

    #[test]
    fn test_common_prefix_len_none() {
        let k1 = PrefixCacheKey::from_tokens(&[1, 2, 3]);
        let k2 = PrefixCacheKey::from_tokens(&[9, 8, 7]);
        assert_eq!(k1.common_prefix_len(&k2), 0);
    }

    #[test]
    fn test_common_prefix_len_different_lengths() {
        let k1 = PrefixCacheKey::from_tokens(&[1, 2, 3]);
        let k2 = PrefixCacheKey::from_tokens(&[1, 2, 3, 4, 5]);
        assert_eq!(k1.common_prefix_len(&k2), 3);
        assert_eq!(k2.common_prefix_len(&k1), 3);
    }

    #[test]
    fn test_empty_key() {
        let k = PrefixCacheKey::from_tokens(&[]);
        assert!(k.is_empty());
        assert_eq!(k.len(), 0);
        let k2 = PrefixCacheKey::from_tokens(&[1, 2]);
        assert_eq!(k.common_prefix_len(&k2), 0);
    }

    #[cfg(target_os = "macos")]
    mod store_tests {
        use super::super::*;
        use crate::backends::common::Backend;

        #[cfg(feature = "metal")]
        type B = crate::backends::metal::Metal;

        #[cfg(feature = "metal")]
        fn make_entry(tokens: &[u64], memory_bytes: usize) -> PrefixCacheEntry<B> {
            PrefixCacheEntry {
                tokens: tokens.to_vec(),
                key: PrefixCacheKey::from_tokens(tokens),
                layer_snapshots: vec![],
                prefix_len: tokens.len(),
                memory_bytes,
                last_access: 0,
            }
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_store_insert_and_find() {
            let config = PrefixCacheConfig {
                enabled: true,
                min_prefix_len: 2,
                max_entries: 16,
                max_memory_bytes: 1024 * 1024,
            };
            let mut store: PrefixCacheStore<B> = PrefixCacheStore::new(config);

            let entry = make_entry(&[1, 2, 3, 4, 5], 100);
            store.insert(entry);

            let query = PrefixCacheKey::from_tokens(&[1, 2, 3, 4, 5, 6]);
            let result = store.find_longest_match(&query);
            assert!(result.is_some());
            let (_, match_len) = result.unwrap();
            assert_eq!(match_len, 5);
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_store_no_match_below_min() {
            let config = PrefixCacheConfig {
                enabled: true,
                min_prefix_len: 10,
                max_entries: 16,
                max_memory_bytes: 1024 * 1024,
            };
            let mut store: PrefixCacheStore<B> = PrefixCacheStore::new(config);

            let entry = make_entry(&[1, 2, 3, 4, 5], 100);
            store.insert(entry);

            // Only 5 tokens match, but min is 10
            let query = PrefixCacheKey::from_tokens(&[1, 2, 3, 4, 5, 6]);
            assert!(store.find_longest_match(&query).is_none());
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_store_lru_eviction() {
            let config = PrefixCacheConfig {
                enabled: true,
                min_prefix_len: 2,
                max_entries: 2,
                max_memory_bytes: 1024 * 1024,
            };
            let mut store: PrefixCacheStore<B> = PrefixCacheStore::new(config);

            let mut entry_a = make_entry(&[1, 2, 3], 100);
            entry_a.last_access = store.next_access_counter();
            store.insert(entry_a);

            let mut entry_b = make_entry(&[4, 5, 6], 100);
            entry_b.last_access = store.next_access_counter();
            store.insert(entry_b);

            // Insert c — should evict a (oldest access)
            let mut entry_c = make_entry(&[7, 8, 9], 100);
            entry_c.last_access = store.next_access_counter();
            store.insert(entry_c);

            // a should be gone
            let query_a = PrefixCacheKey::from_tokens(&[1, 2, 3]);
            assert!(store.find_longest_match(&query_a).is_none());

            // b and c should still be present
            let query_b = PrefixCacheKey::from_tokens(&[4, 5, 6]);
            assert!(store.find_longest_match(&query_b).is_some());

            let query_c = PrefixCacheKey::from_tokens(&[7, 8, 9]);
            assert!(store.find_longest_match(&query_c).is_some());

            assert_eq!(store.stats().evictions, 1);
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_store_memory_budget_eviction() {
            let config = PrefixCacheConfig {
                enabled: true,
                min_prefix_len: 2,
                max_entries: 16,
                max_memory_bytes: 250, // Tight budget
            };
            let mut store: PrefixCacheStore<B> = PrefixCacheStore::new(config);

            let mut entry_a = make_entry(&[1, 2, 3], 100);
            entry_a.last_access = store.next_access_counter();
            store.insert(entry_a);

            let mut entry_b = make_entry(&[4, 5, 6], 100);
            entry_b.last_access = store.next_access_counter();
            store.insert(entry_b);

            // This pushes over 250 bytes, should evict a
            let mut entry_c = make_entry(&[7, 8, 9], 100);
            entry_c.last_access = store.next_access_counter();
            store.insert(entry_c);

            let query_a = PrefixCacheKey::from_tokens(&[1, 2, 3]);
            assert!(store.find_longest_match(&query_a).is_none());
            assert!(store.stats().evictions >= 1);
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_store_picks_longest_match() {
            let config = PrefixCacheConfig {
                enabled: true,
                min_prefix_len: 2,
                max_entries: 16,
                max_memory_bytes: 1024 * 1024,
            };
            let mut store: PrefixCacheStore<B> = PrefixCacheStore::new(config);

            // Short prefix
            let mut entry_short = make_entry(&[1, 2, 3], 100);
            entry_short.last_access = store.next_access_counter();
            store.insert(entry_short);

            // Longer prefix sharing same start
            let mut entry_long = make_entry(&[1, 2, 3, 4, 5], 100);
            entry_long.last_access = store.next_access_counter();
            store.insert(entry_long);

            let query = PrefixCacheKey::from_tokens(&[1, 2, 3, 4, 5, 6, 7]);
            let result = store.find_longest_match(&query);
            assert!(result.is_some());
            let (_, match_len) = result.unwrap();
            assert_eq!(match_len, 5); // Should pick the longer match
        }
    }
}
