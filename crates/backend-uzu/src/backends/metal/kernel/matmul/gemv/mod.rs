mod kernel;
mod policy;

pub(crate) use kernel::{GemvDispatch, GemvSpecialization, max_gemv_batch_threshold};
