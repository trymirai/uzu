//! Sequential Monte Carlo Speculative Decoding (SMC-SD).
//!
//! Port of <https://github.com/abdelfattah-lab/smcsd> (arXiv:2604.15672).
//!
//! Phase 0: serial N=1 draft-extend + target-emit loop — a two-model plumbing
//! spike, not the real algorithm yet. Real importance-weighted resampling
//! lands in Phase 2; see `docs/smcsd/design.md` for the roadmap.
//!
//! Exposed as `pub mod smc` but the API is *unstable* until Phase 1
//! (paged KV cache) lands. Breaking changes expected between commits.

mod config;
mod error;
mod session;

pub use config::SmcConfig;
pub use error::SmcError;
pub use session::SmcSession;
