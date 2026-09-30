use std::{collections::HashMap, path::Path};

use super::{Error, trace_selection::TraceSelection};
use crate::{backends::select_backend, engine::resolve_model_type};

pub struct TraceOutput {
    pub array_count: usize,
}

pub fn record_trace(
    model_path: &Path,
    token_ids: &[u64],
    output_path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<TraceOutput, Error> {
    let model_type = resolve_model_type(model_path).map_err(Error::backend)?;
    select_backend(
        TraceSelection {
            model_type,
            model_path,
            token_ids,
            output_path,
            metadata,
        },
        Error::Backend("Unable to open any backend".to_owned()),
    )
}
