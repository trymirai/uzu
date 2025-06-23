use super::node::Node;

#[derive(Debug)]
pub struct Transition {
    token: u64,
    pub node: Node,
}

impl Transition {
    pub fn new(token: u64) -> Self {
        Self {
            token,
            node: Node::new(),
        }
    }

    pub fn token(&self) -> u64 {
        self.token
    }
}
