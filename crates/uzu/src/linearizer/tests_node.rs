#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::node::Node;

    #[test]
    fn test_new_node_has_no_transitions() {
        let node = Node::new();
        for i in 0..10 {
            assert!(!node.has_transition(i));
        }
    }

    #[test]
    fn test_has_transition() {
        let mut node = Node::new();
        node.get_or_insert_next(42);

        assert!(node.has_transition(42));
        assert!(!node.has_transition(43));
    }

    #[test]
    fn test_get_next() {
        let mut node = Node::new();
        node.get_or_insert_next(1);

        assert!(node.get_next(1).is_some());
        assert!(node.get_next(2).is_none());
    }

    #[test]
    fn test_get_or_insert_next() {
        let mut node = Node::new();

        // First insertion should create a new node
        let node1 = node.get_or_insert_next(1);
        assert!(!node1.has_transition(99));

        // Add a transition to the child node
        node1.get_or_insert_next(99);

        // Get the same node again
        let node1_again = node.get_or_insert_next(1);

        // Verify it's the same node by checking it has the transition we added
        assert!(node1_again.has_transition(99));

        // Different token should give a different node
        let node2 = node.get_or_insert_next(2);
        assert!(!node2.has_transition(99));
    }

    #[test]
    fn test_multiple_transitions() {
        let mut node = Node::new();

        // Insert several transitions
        for i in 0..5 {
            node.get_or_insert_next(i);
        }

        // Verify all transitions exist
        for i in 0..5 {
            assert!(node.has_transition(i));
        }

        // Verify non-existent transitions
        for i in 5..10 {
            assert!(!node.has_transition(i));
        }
    }

    #[test]
    fn test_dfs_with_path_empty_node() {
        let node = Node::new();
        let mut path: Vec<(isize, u64)> = Vec::new();
        let mut visited = Vec::new();

        node.dfs_with_path(&mut path, &mut |current_path: &[(
            isize,
            u64,
        )]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited.push(tokens);
        });

        assert!(visited.is_empty());
    }

    #[test]
    fn test_dfs_with_path_single_node() {
        let mut node = Node::new();
        node.get_or_insert_next(42);

        let mut path: Vec<(isize, u64)> = Vec::new();
        let mut visited = Vec::new();

        node.dfs_with_path(&mut path, &mut |current_path: &[(
            isize,
            u64,
        )]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited.push(tokens);
        });

        assert_eq!(visited.len(), 1);
        assert_eq!(visited[0], vec![42]);
    }

    #[test]
    fn test_dfs_with_path_complex_tree() {
        let mut root = Node::new();

        // Build a sample trie: [1 -> 2 -> 3], [1 -> 2 -> 4], [1 -> 5], [6]
        let node1 = root.get_or_insert_next(1);
        let node2 = node1.get_or_insert_next(2);
        node2.get_or_insert_next(3);
        node2.get_or_insert_next(4);
        node1.get_or_insert_next(5);
        root.get_or_insert_next(6);

        let mut path: Vec<(isize, u64)> = Vec::new();
        let mut visited = Vec::new();

        root.dfs_with_path(&mut path, &mut |current_path: &[(
            isize,
            u64,
        )]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited.push(tokens);
        });

        // Check all expected paths were visited
        assert_eq!(visited.len(), 6);
        assert!(visited.contains(&vec![1]));
        assert!(visited.contains(&vec![1, 2]));
        assert!(visited.contains(&vec![1, 2, 3]));
        assert!(visited.contains(&vec![1, 2, 4]));
        assert!(visited.contains(&vec![1, 5]));
        assert!(visited.contains(&vec![6]));
    }

    #[test]
    fn test_dfs_path_indices() {
        let mut root = Node::new();

        // Build a simple trie: [1], [2]
        root.get_or_insert_next(1);
        root.get_or_insert_next(2);

        let mut path: Vec<(isize, u64)> = Vec::new();
        let mut indices = Vec::new();

        root.dfs_with_path(&mut path, &mut |current_path: &[(
            isize,
            u64,
        )]| {
            if !current_path.is_empty() {
                indices.push(current_path.last().unwrap().0);
            }
        });

        // Check indices are unique and as expected
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
    }

    #[test]
    fn test_simplified_dfs_empty_node() {
        let node = Node::new();
        let mut visited = Vec::new();

        node.dfs(|current_path: &[(isize, u64)]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited.push(tokens);
        });

        assert!(visited.is_empty());
    }

    #[test]
    fn test_simplified_dfs_complex_tree() {
        let mut root = Node::new();

        // Build a sample trie: [1 -> 2 -> 3], [1 -> 2 -> 4], [1 -> 5], [6]
        let node1 = root.get_or_insert_next(1);
        let node2 = node1.get_or_insert_next(2);
        node2.get_or_insert_next(3);
        node2.get_or_insert_next(4);
        node1.get_or_insert_next(5);
        root.get_or_insert_next(6);

        let mut visited = Vec::new();

        root.dfs(|current_path: &[(isize, u64)]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited.push(tokens);
        });

        // Check all expected paths were visited
        assert_eq!(visited.len(), 6);
        assert!(visited.contains(&vec![1]));
        assert!(visited.contains(&vec![1, 2]));
        assert!(visited.contains(&vec![1, 2, 3]));
        assert!(visited.contains(&vec![1, 2, 4]));
        assert!(visited.contains(&vec![1, 5]));
        assert!(visited.contains(&vec![6]));
    }

    #[test]
    fn test_both_dfs_methods_equivalent() {
        let mut root = Node::new();

        // Build a more complex trie
        let node1 = root.get_or_insert_next(10);
        let node2 = node1.get_or_insert_next(20);
        node2.get_or_insert_next(30);
        node2.get_or_insert_next(40);
        node1.get_or_insert_next(50);
        let node3 = root.get_or_insert_next(60);
        node3.get_or_insert_next(70);

        // Get results from dfs_with_path
        let mut path: Vec<(isize, u64)> = Vec::new();
        let mut visited1 = Vec::new();

        root.dfs_with_path(&mut path, &mut |current_path: &[(
            isize,
            u64,
        )]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited1.push(tokens);
        });

        // Get results from simplified dfs
        let mut visited2 = Vec::new();

        root.dfs(|current_path: &[(isize, u64)]| {
            let tokens: Vec<u64> =
                current_path.iter().map(|(_, token)| *token).collect();
            visited2.push(tokens);
        });

        // Check both methods give the same results
        assert_eq!(visited1.len(), visited2.len());

        // Convert to sets to ignore order differences
        let set1: HashSet<Vec<u64>> = visited1.into_iter().collect();
        let set2: HashSet<Vec<u64>> = visited2.into_iter().collect();

        assert_eq!(set1, set2);
    }
}
