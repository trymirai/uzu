use std::{collections::HashMap, path::Path};

use shoji::types::model::ModelSpecialization;

use super::{Error, Recorder};
use crate::{
    backends::{common::Backend, select_backend},
    bridge::model_specialization,
    engine::Engine,
};

pub struct TraceOutput {
    pub array_count: usize,
}

pub fn record_trace(
    model_path: &Path,
    token_ids: &[u64],
    output_path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<TraceOutput, Error> {
    match model_specialization(model_path).map_err(Error::backend)? {
        ModelSpecialization::Chat {} => record_language_model(model_path, token_ids, output_path, metadata),
        ModelSpecialization::Classification {} => record_classifier(model_path, token_ids, output_path, metadata),
        other => Err(Error::Backend(format!("Tracing is not supported for {} models", other.name()))),
    }
}

fn record_language_model(
    model_path: &Path,
    token_ids: &[u64],
    output_path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<TraceOutput, Error> {
    select_backend!(
        {
            let engine = Engine::<B>::new().map_err(Error::backend)?;
            let model = engine.load_language_model(model_path).map_err(Error::backend)?;
            write(model.record_trace(token_ids).map_err(Error::backend)?, output_path, metadata)
        },
        Error::Backend("Unable to open any backend".to_owned())
    )
}

fn record_classifier(
    model_path: &Path,
    token_ids: &[u64],
    output_path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<TraceOutput, Error> {
    select_backend!(
        {
            let engine = Engine::<B>::new().map_err(Error::backend)?;
            let model = engine.load_classifier_model(model_path).map_err(Error::backend)?;
            write(model.record_trace(token_ids).map_err(Error::backend)?, output_path, metadata)
        },
        Error::Backend("Unable to open any backend".to_owned())
    )
}

fn write<B: Backend>(
    recorder: Recorder<B>,
    output_path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<TraceOutput, Error> {
    if let Some(parent) = output_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    recorder.write(output_path, metadata)?;

    Ok(TraceOutput {
        array_count: recorder.len(),
    })
}
