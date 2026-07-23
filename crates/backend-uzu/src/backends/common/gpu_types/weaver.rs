pub const CANDIDATES_MAX: usize = 512;
pub const TOP_CHILDREN_THREADS: usize = 256;
pub const TOP_CHILDREN_SIMDGROUPS: usize = 8;

pub const FRONTIER_LANE_TOKEN: usize = 0;
pub const FRONTIER_LANE_PARENT: usize = 1;
pub const FRONTIER_LANE_DEPTH: usize = 2;
pub const FRONTIER_LANE_CUM: usize = 3;
pub const FRONTIER_LANE_LOGPROB: usize = 4;
pub const FRONTIER_LANE_KEY: usize = 5;
pub const FRONTIER_LANE_ACTIVE: usize = 6;
pub const FRONTIER_LANE_COUNT: usize = FRONTIER_LANE_ACTIVE + 1;

pub const TREE_LANE_TOKEN: usize = 0;
pub const TREE_LANE_PARENT: usize = 1;
pub const TREE_LANE_DEPTH: usize = 2;
pub const TREE_LANE_CUM: usize = 3;
pub const TREE_LANE_LOGPROB: usize = 4;
pub const TREE_LANE_MASK: usize = 5;
pub const TREE_LANE_COUNT: usize = TREE_LANE_MASK + 1;

pub const METADATA_LANE_DEPTH: usize = 0;
pub const METADATA_LANE_ANCESTOR_COUNT: usize = 1;
pub const METADATA_LANE_NODE_INDEX: usize = 2;
pub const METADATA_LANE_COUNT: usize = METADATA_LANE_NODE_INDEX + 1;

pub const FRONTIER_NO_WINNER: u32 = !0;
pub const FRONTIER_SELECT_THREADS: usize = 256;
pub const FRONTIER_SELECT_SIMDGROUPS: usize = 8;
pub const FRONTIER_ENTRIES_PER_THREAD: usize = 8;
pub const FRONTIER_MAX_SLOTS: usize = 2048;
pub const FRONTIER_MAX_WIDTH: usize = 32;
