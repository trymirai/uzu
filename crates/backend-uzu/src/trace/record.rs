use std::{collections::HashMap, path::Path};

use shoji::types::model::ModelSpecialization;

use super::{ClassifierTapRequest, DecoderTapRequest, Error};
use crate::{backends::select_backend, bridge::model_specialization, engine::Engine};

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
            let mut model = engine.load_language_model(model_path).map_err(Error::backend)?;
            let array_count = model.record_trace(token_ids, &DecoderTapRequest::all()).map_err(Error::backend)?.len();
            model.write_trace(output_path, metadata)?;

            Ok(TraceOutput {
                array_count,
            })
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
            let mut model = engine.load_classifier_model(model_path).map_err(Error::backend)?;
            let array_count =
                model.record_trace(token_ids, &ClassifierTapRequest::all()).map_err(Error::backend)?.len();
            model.write_trace(output_path, metadata)?;

            Ok(TraceOutput {
                array_count,
            })
        },
        Error::Backend("Unable to open any backend".to_owned())
    )
}
