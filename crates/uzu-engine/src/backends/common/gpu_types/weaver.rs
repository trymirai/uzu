pub const CANDIDATES_MAX: u32 = 512;
pub const TOP_CHILDREN_THREADS: u32 = 256;
pub const TOP_CHILDREN_SIMDGROUPS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum FrontierIdx {
    TokenId,
    ParentSlot,
    Depth,
    PathLogprobBits,
    EdgeLogprobBits,
    PathScoreKey,
    Active,
}

impl FrontierIdx {
    pub const COUNT: usize = Self::Active as usize + 1;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum TreeIdx {
    TokenId,
    ParentSlot,
    Depth,
    PathLogprobBits,
    EdgeLogprobBits,
    Valid,
}

impl TreeIdx {
    pub const COUNT: usize = Self::Valid as usize + 1;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum MetadataIdx {
    Depth,
    AncestorCount,
    TreeSlot,
}

impl MetadataIdx {
    pub const COUNT: usize = Self::TreeSlot as usize + 1;
}

pub const FRONTIER_NO_WINNER: u32 = !0;
pub const FRONTIER_SELECT_THREADS: u32 = 256;
pub const FRONTIER_SELECT_SIMDGROUPS: u32 = 8;
pub const FRONTIER_ENTRIES_PER_THREAD: u32 = 8;
pub const FRONTIER_MAX_SLOTS: u32 = 2048;
pub const FRONTIER_MAX_WIDTH: u32 = 32;
pub const PADDING_DEPTH: u32 = 0;
