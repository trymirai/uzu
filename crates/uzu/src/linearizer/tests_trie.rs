#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        linearizer::trie::{TokenTrie, TrieCreationConfig},
        llm::rng::DerivableSeed,
        speculators::speculator::Speculator,
    };

    struct StaticSpeculator {
        responses: HashMap<Vec<u64>, HashMap<u64, f32>>,
    }

    impl StaticSpeculator {
        fn new(responses: HashMap<Vec<u64>, HashMap<u64, f32>>) -> Self {
            Self {
                responses,
            }
        }
    }

    impl Speculator for StaticSpeculator {
        fn speculate(
            &self,
            prefix: &[u64],
        ) -> HashMap<u64, f32> {
            self.responses.get(prefix).cloned().unwrap_or_default()
        }
    }

    #[test]
    fn test_from_speculator_applies_prob_cutoff() {
        let mut responses = HashMap::new();
        responses.insert(
            vec![42],
            HashMap::from([(1, 0.9_f32), (2, 0.05_f32), (3, 0.2_f32)]),
        );
        responses.insert(vec![42, 1], HashMap::new());

        let speculator = StaticSpeculator::new(responses);
        let mut seed = DerivableSeed::new(1);

        let trie = TokenTrie::from_speculator(
            &[42],
            &mut seed,
            false,
            &speculator,
            &TrieCreationConfig::default(),
            5,
        );
        let suffix = trie.linearize(0, 5);

        assert_eq!(suffix.tokens, vec![1, 3]);
        assert!(!suffix.tokens.contains(&2));
        assert_eq!(suffix.transition_map.len(), 3);
    }

    #[test]
    fn test_from_speculator_respects_top_k_limit() {
        let mut responses = HashMap::new();
        responses.insert(vec![5], HashMap::from([(9, 0.8_f32), (10, 0.7_f32)]));
        responses.insert(vec![5, 9], HashMap::from([(11, 0.6_f32)]));
        responses.insert(vec![5, 9, 11], HashMap::new());

        let speculator = StaticSpeculator::new(responses);
        let mut seed = DerivableSeed::new(7);

        let trie = TokenTrie::from_speculator(
            &[5],
            &mut seed,
            false,
            &speculator,
            &TrieCreationConfig {
                top_k: 1,
                prob_cutoff: 0.0,
            },
            3,
        );

        let suffix = trie.linearize(0, 4);

        assert_eq!(suffix.tokens, vec![9, 11]);
        assert!(!suffix.tokens.contains(&10));
    }

    #[test]
    fn test_from_speculator_includes_last_prefix_token_and_seeds() {
        let mut responses = HashMap::new();
        responses.insert(vec![2, 4], HashMap::from([(8, 0.9_f32)]));
        responses.insert(vec![2, 4, 8], HashMap::new());

        let speculator = StaticSpeculator::new(responses);

        let mut seed = DerivableSeed::new(99);
        let mut expected_seed_stream = DerivableSeed::new(99);
        let expected_seeds =
            vec![expected_seed_stream.next(), expected_seed_stream.next()];

        let trie = TokenTrie::from_speculator(
            &[2, 4],
            &mut seed,
            true,
            &speculator,
            &TrieCreationConfig::default(),
            2,
        );

        let suffix = trie.linearize(0, 3);

        assert_eq!(suffix.tokens, vec![4, 8]);
        assert_eq!(suffix.seeds, expected_seeds);
        assert_eq!(suffix.indices, vec![0, 1]);
    }

    #[test]
    fn test_from_speculator_stops_at_max_length() {
        let mut responses = HashMap::new();
        responses.insert(
            vec![1],
            HashMap::from([(5, 0.9_f32), (6, 0.8_f32), (7, 0.7_f32)]),
        );
        responses
            .insert(vec![1, 5], HashMap::from([(8, 0.6_f32), (9, 0.5_f32)]));
        responses.insert(vec![1, 5, 8], HashMap::from([(10, 0.4_f32)]));

        let speculator = StaticSpeculator::new(responses);
        let mut seed = DerivableSeed::new(1234);

        let trie = TokenTrie::from_speculator(
            &[1],
            &mut seed,
            false,
            &speculator,
            &TrieCreationConfig::default(),
            3,
        );

        let suffix = trie.linearize(0, 5);

        assert_eq!(suffix.tokens.len(), 3);
        assert!(suffix.tokens.contains(&5));
        assert!(suffix.tokens.contains(&6));
        assert!(suffix.tokens.contains(&8));
        assert!(!suffix.tokens.contains(&9));
    }

    #[test]
    fn test_from_speculator_with_empty_responses() {
        let speculator = StaticSpeculator::new(HashMap::new());

        let mut seed = DerivableSeed::new(555);
        let mut expected_seed_stream = DerivableSeed::new(555);
        let expected_first_seed = expected_seed_stream.next();

        let trie = TokenTrie::from_speculator(
            &[12, 13],
            &mut seed,
            true,
            &speculator,
            &TrieCreationConfig::default(),
            4,
        );

        let suffix = trie.linearize(0, 5);

        // Only the last prefix token should be included; no speculated tokens.
        assert_eq!(suffix.tokens, vec![13]);
        assert_eq!(suffix.indices, vec![0]);
        assert_eq!(suffix.seeds, vec![expected_first_seed]);

        assert_eq!(suffix.transition_map.len(), 2);
        let root_transitions = &suffix.transition_map[&-1];
        assert_eq!(root_transitions.len(), 1);
        assert_eq!(root_transitions.get(&13), Some(&0));
    }

    #[test]
    fn test_linearize_empty_trie() {
        let trie = TokenTrie::new();
        let suffix = trie.linearize(0, 10);

        assert!(suffix.tokens.is_empty());
        assert!(suffix.indices.is_empty());
        assert!(suffix.transition_map.len() == 1);
    }

    #[test]
    fn test_linearize_single_sequence() {
        let mut trie = TokenTrie::new();
        trie.insert(&[1, 2, 3]);

        let suffix = trie.linearize(0, 5);

        assert_eq!(suffix.tokens.len(), 3);
        assert!(suffix.tokens.contains(&1));
        assert!(suffix.tokens.contains(&2));
        assert!(suffix.tokens.contains(&3));

        assert_eq!(suffix.transition_map.len(), 4);

        let root_transitions = &suffix.transition_map[&-1];
        assert!(root_transitions.contains_key(&1));

        let token1_idx = *root_transitions.get(&1).unwrap();
        let token1_transitions = &suffix.transition_map[&token1_idx];
        assert!(token1_transitions.contains_key(&2));

        let token2_idx = *token1_transitions.get(&2).unwrap();
        let token2_transitions = &suffix.transition_map[&token2_idx];
        assert!(token2_transitions.contains_key(&3));
    }

    #[test]
    fn test_linearize_branching_trie() {
        let mut trie = TokenTrie::new();
        trie.insert(&[1, 2, 3]);
        trie.insert(&[1, 2, 4]);
        trie.insert(&[5, 6]);

        let suffix = trie.linearize(10, 10);

        assert!(suffix.tokens.contains(&1));
        assert!(suffix.tokens.contains(&2));
        assert!(suffix.tokens.contains(&3));
        assert!(suffix.tokens.contains(&4));
        assert!(suffix.tokens.contains(&5));
        assert!(suffix.tokens.contains(&6));

        assert_eq!(suffix.tokens.len(), 6);

        assert_eq!(suffix.transition_map.len(), 7);

        let root_transitions = &suffix.transition_map[&-1];
        assert!(root_transitions.contains_key(&1));
        assert!(root_transitions.contains_key(&5));

        let token1_idx = *root_transitions.get(&1).unwrap();
        let token1_transitions = &suffix.transition_map[&token1_idx];
        assert!(token1_transitions.contains_key(&2));

        let token2_idx = *token1_transitions.get(&2).unwrap();
        let token2_transitions = &suffix.transition_map[&token2_idx];
        assert!(token2_transitions.contains_key(&3));
        assert!(token2_transitions.contains_key(&4));

        let token5_idx = *root_transitions.get(&5).unwrap();
        let token5_transitions = &suffix.transition_map[&token5_idx];
        assert!(token5_transitions.contains_key(&6));
    }

    #[test]
    fn test_linearize_max_length() {
        let mut trie = TokenTrie::new();
        trie.insert(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let suffix = trie.linearize(0, 5);

        assert_eq!(suffix.tokens.len(), 5);
        assert_eq!(suffix.indices.len(), 5);
        assert_eq!(suffix.transition_map.len(), 6);

        for i in 1..=5 {
            assert!(suffix.tokens.contains(&i));
        }

        let root_transitions = &suffix.transition_map[&-1];
        assert!(root_transitions.contains_key(&1));

        let mut current_idx = *root_transitions.get(&1).unwrap();
        for i in 2..=4 {
            let transitions = &suffix.transition_map[&current_idx];
            assert!(transitions.contains_key(&i));
            current_idx = *transitions.get(&i).unwrap();
        }
    }

    #[test]
    fn test_linearize_start_index() {
        let mut trie = TokenTrie::new();
        trie.insert(&[1, 2, 3]);

        let suffix = trie.linearize(100, 10);

        assert_eq!(suffix.tokens.len(), 3);
        assert!(suffix.tokens.contains(&1));
        assert!(suffix.tokens.contains(&2));
        assert!(suffix.tokens.contains(&3));

        for idx in suffix.indices.iter() {
            assert!(*idx >= 100 && *idx <= 102);
        }

        let root_transitions = &suffix.transition_map[&-1];
        assert!(root_transitions.contains_key(&1));
    }
}
