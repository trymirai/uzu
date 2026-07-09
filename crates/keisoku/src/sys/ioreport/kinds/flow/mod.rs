pub(crate) trait FlowKind: 'static {
    const FLOW: DramFlow;
    const TYPE_BIT_BYTES: u128;
    const TYPE_BIT_HIST: u128;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DramFlow {
    DramRead,
    DramWrite,
}

mod dram_read;
mod dram_write;

pub use dram_read::DramRead;
pub use dram_write::DramWrite;
