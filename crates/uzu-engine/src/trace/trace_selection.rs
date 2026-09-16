use std::{collections::HashMap, path::Path};

use super::{ClassifierTapRequest, DecoderTapRequest, Error, TraceOutput};
use crate::{
    backends::{BackendSelection, common::Backend},
    engine::{Engine, ModelType},
};

pub struct TraceSelection<'a> {
    pub model_type: ModelType,
    pub model_path: &'a Path,
    pub token_ids: &'a [u64],
    pub output_path: &'a Path,
    pub metadata: Option<HashMap<String, String>>,
}

impl BackendSelection for TraceSelection<'_> {
    type Output = TraceOutput;
    type Error = Error;

    fn select<B: Backend>(self) -> Result<TraceOutput, Error> {
        let engine = Engine::<B>::new().map_err(Error::backend)?;
        let array_count = match self.model_type {
            ModelType::LanguageModel => {
                let mut model = engine.load_language_model(self.model_path).map_err(Error::backend)?;
                let array_count =
                    model.record_trace(self.token_ids, &DecoderTapRequest::all()).map_err(Error::backend)?.len();
                model.write_trace(self.output_path, self.metadata)?;
                array_count
            },
            ModelType::Classifier => {
                let mut model = engine.load_classifier_model(self.model_path).map_err(Error::backend)?;
                let array_count =
                    model.record_trace(self.token_ids, &ClassifierTapRequest::all()).map_err(Error::backend)?.len();
                model.write_trace(self.output_path, self.metadata)?;
                array_count
            },
        };
        Ok(TraceOutput {
            array_count,
        })
    }
}
