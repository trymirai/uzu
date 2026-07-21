pub(crate) const MAX_CANDIDATES: usize = 512;

pub(crate) const FRONTIER_LANE_TOKEN: usize = 0;
pub(crate) const FRONTIER_LANE_PARENT: usize = 1;
pub(crate) const FRONTIER_LANE_DEPTH: usize = 2;
pub(crate) const FRONTIER_LANE_CUM: usize = 3;
pub(crate) const FRONTIER_LANE_LOGPROB: usize = 4;
pub(crate) const FRONTIER_LANE_KEY: usize = 5;
pub(crate) const FRONTIER_LANE_ACTIVE: usize = 6;

pub(crate) const TREE_LANE_TOKEN: usize = 0;
pub(crate) const TREE_LANE_PARENT: usize = 1;
pub(crate) const TREE_LANE_DEPTH: usize = 2;
pub(crate) const TREE_LANE_CUM: usize = 3;
pub(crate) const TREE_LANE_LOGPROB: usize = 4;
pub(crate) const TREE_LANE_MASK: usize = 5;

pub(crate) const METADATA_LANE_DEPTH: usize = 0;
pub(crate) const METADATA_LANE_ANCESTOR_COUNT: usize = 1;
pub(crate) const METADATA_LANE_NODE_INDEX: usize = 2;

pub(crate) const NO_WINNER: u32 = u32::MAX;
pub(crate) const MAX_FRONTIER_SLOTS: usize = 2048;
pub(crate) const MAX_FRONTIER_WIDTH: usize = 32;
