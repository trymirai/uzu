//! Activation tracing, laid out to match lalamo's trace export so the two can
//! be diffed array-by-array with `lalamo compare-traces`.
//!
//! Capture points are the [`trace!`](crate::utils::trace) family of macros,
//! which compile to nothing unless the `trace` feature is on and do nothing
//! unless a [`Recorder`] is attached to the encoder.

mod array;
mod data_type;
mod error;
mod recorder;

use array::Array;
pub use error::Error;
pub use recorder::Recorder;
