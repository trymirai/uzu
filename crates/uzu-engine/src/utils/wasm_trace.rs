//! Tracing is unavailable on wasm (missing filesystem, env vars, threads, etc.)

use std::sync::mpsc::Sender;

use crate::backends::common::TimestampSpan;

pub fn sender_from_env() -> Option<Sender<Box<[TimestampSpan]>>> {
    None
}
