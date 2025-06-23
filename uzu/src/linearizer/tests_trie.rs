#[cfg(test)]
mod tests {
    use crate::linearizer::trie::TokenTrie;

    #[test]
    fn test_linearize_empty_trie() {
        let trie = TokenTrie::new();
        let suffix = trie.linearize(0, 10);

        assert!(suffix.tokens.is_empty());
        assert!(suffix.indices.is_empty());
        assert!(suffix.causal_mask.is_empty());
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
        assert_eq!(suffix.causal_mask.len(), 5);
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
