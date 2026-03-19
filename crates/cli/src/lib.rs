pub mod handlers;
pub mod server;

use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub enum SamplerArg {
    /// Greedy argmax (deterministic)
    Greedy,
    /// Stochastic multi-kernel (temperature → top-k → top-p → min-p → gumbel)
    Stochastic,
    /// Stochastic single-pass kernel
    UnifiedStochastic,
}
