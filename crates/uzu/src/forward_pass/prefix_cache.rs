use xxhash_rust::xxh3::Xxh3;

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
}
