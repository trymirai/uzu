#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub name: String,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
}
