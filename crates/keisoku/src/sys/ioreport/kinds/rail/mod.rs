pub(crate) trait RailKind: 'static {
    const RAIL: Rail;
    const TYPE_BIT: u128;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rail {
    Cpu,
    Gpu,
    Ane,
    Ram,
}

mod ane;
mod cpu;
mod gpu;
mod ram;

pub use ane::Ane;
pub use cpu::Cpu;
pub use gpu::Gpu;
pub use ram::Ram;
