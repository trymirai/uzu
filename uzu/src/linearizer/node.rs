use super::transition::Transition;

#[derive(Debug)]
pub struct Node {
    transitions: Vec<Transition>,
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

impl Node {
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
        }
    }

    pub fn has_transition(
        &self,
        token: u64,
    ) -> bool {
        self.transitions.iter().any(|t| t.token() == token)
    }

    pub fn get_next(
        &self,
        token: u64,
    ) -> Option<&Node> {
        self.transitions.iter().find(|t| t.token() == token).map(|t| &t.node)
    }

    pub fn get_or_insert_next(
        &mut self,
        token: u64,
    ) -> &mut Node {
        if let Some(pos) =
            self.transitions.iter().position(|t| t.token() == token)
        {
            return &mut self.transitions[pos].node;
        }

        self.transitions.push(Transition::new(token));
        &mut self.transitions.last_mut().unwrap().node
    }

    pub fn dfs_with_path<F>(
        &self,
        path: &mut Vec<(isize, u64)>,
        callback: &mut F,
    ) -> isize
    where
        F: FnMut(&[(isize, u64)]),
    {
        let mut node_index = path.last().map(|(idx, _)| *idx).unwrap_or(-1);

        for transition in self.transitions.iter() {
            node_index += 1;
            path.push((node_index, transition.token()));
            callback(path);
            node_index = transition.node.dfs_with_path(path, callback);
            path.pop();
        }

        node_index
    }

    pub fn dfs<F>(
        &self,
        mut callback: F,
    ) where
        F: FnMut(&[(isize, u64)]),
    {
        let mut path: Vec<(isize, u64)> = Vec::new();
        self.dfs_with_path(&mut path, &mut callback);
    }
}
