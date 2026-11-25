//! Tracing is unavailable on wasm (missing filesystem, env vars, threads, etc.)

use crate::utils::timestamps::TimestampSender;

pub fn sender_from_env() -> Option<TimestampSender> {
    None
}
