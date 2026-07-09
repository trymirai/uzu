//! Energy-rail and DRAM-flow kinds: each marker ZST maps to one enum variant.

mod flow;
mod rail;

pub(crate) use flow::FlowKind;
pub use flow::{DramFlow, DramRead, DramWrite};
pub(crate) use rail::RailKind;
pub use rail::{Ane, Cpu, Gpu, Rail, Ram};
