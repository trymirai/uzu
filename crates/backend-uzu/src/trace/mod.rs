mod array;
mod data_type;
mod error;
mod record;
mod recorder;

use array::Array;
pub use error::Error;
pub use record::{TraceOutput, record_trace};
pub use recorder::Recorder;
