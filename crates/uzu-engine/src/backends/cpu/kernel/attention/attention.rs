use crate::backends::{common::kernel::attention::SinglePassAttention, cpu::Cpu};

pub type Attention = SinglePassAttention<Cpu>;
