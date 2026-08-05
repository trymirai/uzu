use std::{path::PathBuf, pin::Pin, sync::Arc};

use futures::{StreamExt, stream};
use parking_lot::Mutex;
use shoji::traits::{
    State as ShojiState,
    backend::{
        Error as BackendError, Instance as BackendInstance, InstanceStream, NoMetricsStream,
        classification::{
            StreamConfig as ClassificationStreamConfig, StreamInput as ClassificationStreamInput,
            StreamMetrics as ClassificationStreamMetrics, StreamOutput as ClassificationStreamOutput,
        },
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    engine::{Engine, classifier_model::ClassifierModel},
};

pub struct UzuClassificationBackendInstance<B: Backend> {
    model: Arc<Mutex<ClassifierModel<B>>>,
}

impl<B: Backend> UzuClassificationBackendInstance<B> {
    pub fn new(model_path: String) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_classifier_model(&model_path).map_err(|err| err.to_string())?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

impl<B: Backend> BackendInstance for UzuClassificationBackendInstance<B> {
    type StreamConfig = ClassificationStreamConfig;
    type StreamInput = ClassificationStreamInput;
    type StreamOutput = ClassificationStreamOutput;
    type StreamMetrics = ClassificationStreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn ShojiState>, BackendError>> + Send + '_>> {
        Box::pin(async move { Ok(Box::new(State) as Box<dyn ShojiState>) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn ShojiState,
        _config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<
        Box<
            dyn InstanceStream<Item = Result<Self::StreamOutput, BackendError>, Metrics = ClassificationStreamMetrics>
                + Send
                + 'a,
        >,
    > {
        Box::pin(NoMetricsStream::new(
            stream::once(async move {
                let model_guard = self.model.lock();
                model_guard.classify(input).map_err(|err| BackendError::from(err.to_string()))
            })
            .take_until(cancel_token.cancelled_owned()),
        ))
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone)]
struct State;

impl ShojiState for State {}
