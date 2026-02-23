mod argmax_single;
mod argmax_main;
mod argmax_final;
mod bitmask;
mod gumbel;
mod min_p;
mod temperature;
mod top_k;
mod top_p;

pub use argmax_single::ArgmaxSingleCpuKernel;
pub use argmax_main::ArgmaxMainCpuKernel;
pub use argmax_final::ArgmaxFinalCpuKernel;
pub use bitmask::BitmaskCpuKernel;
pub use gumbel::GumbelCpuKernel;
pub use min_p::MinPCpuKernel;
pub use temperature::TemperatureCpuKernel;
pub use top_k::TopKCpuKernel;
pub use top_p::TopPCpuKernel;
