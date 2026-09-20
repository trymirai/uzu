use std::{any::Any, future::Future, pin::Pin};

use shoji::{
    traits::{
        Instance as InstanceTrait, State as StateTrait,
        backend::{
            Error as BackendError, InstanceStream, NoMetricsStream,
            chat_message::{StreamInput, StreamMetrics, StreamOutput},
        },
    },
    types::session::chat::ChatReplyConfig,
};
use tokio_util::sync::CancellationToken;

use crate::{
    engine::{EngineHandle, NeedleState},
    error::Error,
};

pub struct Instance {
    handle: EngineHandle,
    peak_memory_usage: Option<usize>,
}

impl Instance {
    pub fn new(
        handle: EngineHandle,
        peak_memory_usage: Option<usize>,
    ) -> Self {
        Self {
            handle,
            peak_memory_usage,
        }
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;
    type StreamMetrics = StreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        Box::pin(async { Ok(Box::new(NeedleState::new()) as Box<dyn StateTrait>) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn StateTrait,
        config: Self::StreamConfig,
        cancel: CancellationToken,
    ) -> Pin<
        Box<
            dyn InstanceStream<Item = Result<Self::StreamOutput, BackendError>, Metrics = Self::StreamMetrics>
                + Send
                + 'a,
        >,
    > {
        let Some(needle_state) = (state as &mut dyn Any).downcast_mut::<NeedleState>() else {
            return Box::pin(NoMetricsStream::new(futures::stream::once(async {
                Err(Box::new(Error::InvalidState) as BackendError)
            })));
        };
        if cancel.is_cancelled() {
            return Box::pin(NoMetricsStream::new(futures::stream::once(async {
                Err(Box::new(Error::Cancelled) as BackendError)
            })));
        }

        let handle = self.handle.clone();
        let input = input.clone();
        Box::pin(NoMetricsStream::new(futures::stream::once(async move {
            if cancel.is_cancelled() {
                return Err(Box::new(Error::Cancelled) as BackendError);
            }
            let mut local = needle_state.clone();
            let result = tokio::task::spawn_blocking(move || {
                handle.drive_turn(&mut local, &input, &config).map(|output| (output, local))
            })
            .await
            .map_err(|error| {
                Box::new(Error::LibraryLoad {
                    message: error.to_string(),
                }) as BackendError
            })?;
            let (output, updated) = result.map_err(|error| Box::new(error) as BackendError)?;
            *needle_state = updated;
            Ok(output)
        })))
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.peak_memory_usage
    }
}
