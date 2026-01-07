#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub name: String,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
    pub has_batch: bool,
    pub use_out_source: bool,
    pub do_axpby: bool,
}
