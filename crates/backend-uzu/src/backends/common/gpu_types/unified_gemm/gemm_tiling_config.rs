#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTilingConfig {
    pub threadgroup_m: u32,
    pub threadgroup_n: u32,
    pub threadgroup_k: u32,
    pub simdgroup_m: u32,
    pub simdgroup_n: u32,
    pub simdgroup_k: u32,
    pub fragment_m: u32,
    pub fragment_n: u32,
    pub fragment_k: u32,
    pub simdgroups_m: u32,
    pub simdgroups_n: u32,
}
