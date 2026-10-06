use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use metal::MTLComputePipelineState;
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::metal::error::MetalError;

type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Clone)]
pub(crate) struct MetalPipeline {
    entry: Arc<MetalPipelineEntry>,
}

struct MetalPipelineEntry {
    result: OnceLock<Result<ComputePipelineState, String>>,
    function_name: Arc<str>,
}

impl MetalPipeline {
    fn pending(function_name: &str) -> Self {
        Self {
            entry: Arc::new(MetalPipelineEntry {
                result: OnceLock::new(),
                function_name: function_name.into(),
            }),
        }
    }

    pub(super) fn complete(
        &self,
        result: Result<ComputePipelineState, MetalError>,
    ) {
        let result = result.map_err(|error| match error {
            MetalError::CannotCreatePipelineState {
                error,
                ..
            } => error,
            error => error.to_string(),
        });
        let _ = self.entry.result.set(result);
    }

    pub(crate) fn wait(&self) -> Result<&ProtocolObject<dyn MTLComputePipelineState>, MetalError> {
        self.entry.result.wait().as_ref().map(Retained::as_ref).map_err(|error| MetalError::CannotCreatePipelineState {
            function_name: self.entry.function_name.to_string(),
            error: error.clone(),
        })
    }
}

#[derive(Default)]
pub(super) struct MetalPipelineCache {
    entries: Mutex<HashMap<String, MetalPipeline>>,
}

impl MetalPipelineCache {
    pub(super) fn get(
        &self,
        key: &str,
    ) -> Option<MetalPipeline> {
        self.entries.lock().get(key).cloned()
    }

    pub(super) fn insert_if_absent(
        &self,
        key: &str,
        function_name: &str,
    ) -> (MetalPipeline, bool) {
        let mut entries = self.entries.lock();
        match entries.get(key) {
            Some(pipeline) => (pipeline.clone(), false),
            None => {
                let pipeline = MetalPipeline::pending(function_name);
                entries.insert(key.to_owned(), pipeline.clone());
                (pipeline, true)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn cache_reuses_a_pending_pipeline_and_its_result() {
        let cache = MetalPipelineCache::default();
        let (first, inserted) = cache.insert_if_absent("key", "kernel");
        let (second, inserted_again) = cache.insert_if_absent("key", "kernel");

        assert!(inserted);
        assert!(!inserted_again);

        first.complete(Err(MetalError::CannotCreatePipelineState {
            function_name: "kernel".to_owned(),
            error: "failed".to_owned(),
        }));

        assert!(matches!(
            second.wait(),
            Err(MetalError::CannotCreatePipelineState {
                function_name,
                error,
            }) if function_name == "kernel" && error == "failed"
        ));
    }
}
