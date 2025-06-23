use std::collections::HashMap;

#[derive(Debug)]
pub struct SpeculatedSuffix {
    pub tokens: Vec<u64>,
    pub indices: Vec<usize>,
    pub causal_mask: Vec<Vec<bool>>,
    pub transition_map: HashMap<isize, HashMap<u64, isize>>,
}
