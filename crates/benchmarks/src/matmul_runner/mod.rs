mod runner;
pub mod types;

pub use runner::MatmulRunner;
pub use types::{MatmulBenchmarkResult, MatmulBenchmarkTask, MatmulDtypeCombo, MatmulShape};
