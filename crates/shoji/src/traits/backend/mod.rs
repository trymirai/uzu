pub mod chat_message;
pub mod chat_token;
pub mod classification;
pub mod text_to_speech;

use std::{
    any::Any,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use tokio_util::sync::CancellationToken;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub trait Backend: Send + Sync {
    fn identifier(&self) -> String;
    fn version(&self) -> String;

    fn as_chat_via_token_capable(&self) -> Option<&dyn chat_token::Backend> {
        None
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message::Backend> {
        None
    }

    fn as_classification_capable(&self) -> Option<&dyn classification::Backend> {
        None
    }

    fn as_text_to_speech_capable(&self) -> Option<&dyn text_to_speech::Backend> {
        None
    }
}

pub trait InstanceStream: Stream {
    type Metrics;

    fn metrics(&self) -> Self::Metrics;

    fn finish(self: Pin<&mut Self>) {}
}

pub struct NoMetricsStream<S, T> {
    inner: Pin<Box<S>>,
    _phantom: PhantomData<fn() -> T>,
}

impl<S, T> NoMetricsStream<S, T> {
    pub fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
            _phantom: PhantomData,
        }
    }
}

impl<S: Stream, T> Stream for NoMetricsStream<S, T> {
    type Item = S::Item;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

impl<S: Stream, T> InstanceStream for NoMetricsStream<S, T> {
    type Metrics = Option<T>;

    fn metrics(&self) -> Self::Metrics {
        None
    }
}

pub trait Instance: Send + Sync {
    type StreamConfig;
    type StreamInput;
    type StreamOutput;
    type StreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>>;

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<Self::StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>;

    fn peak_memory_usage(&self) -> Option<usize>;
}

pub trait State: Send + Sync + Any + 'static {}
