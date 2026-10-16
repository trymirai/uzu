use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    sync::{
        LazyLock,
        mpsc::{self, Receiver, Sender},
    },
};

use parking_lot::Mutex;
use thiserror::Error;

use crate::backends::common::{Backend, Context};

pub static CAPTURE_MATCHER: LazyLock<CaptureMatcher> = LazyLock::new(CaptureMatcher::new);

struct CaptureMatcherInner {
    starts: HashMap<String, (Sender<Box<dyn Send + Sync>>, PathBuf)>,
    ends: HashMap<String, Receiver<Box<dyn Send + Sync>>>,
}

pub struct CaptureMatcher {
    enabled: bool,
    inner: Mutex<CaptureMatcherInner>,
}

#[derive(Debug, Error)]
pub enum CaptureError<B: Backend> {
    #[error("Cannot start backend capture: {0}")]
    BackendCapture(#[source] B::Error),
    #[error("Attempted to start capture whose endpoint already happened")]
    StartAfterEnd,
}

impl CaptureMatcher {
    fn new() -> Self {
        let mut enabled = false;
        let mut starts = HashMap::new();
        let mut ends = HashMap::new();

        if let Some(capture) = env::var_os("UZU_CAPTURE").map(|s| s.into_string().unwrap()) {
            let spans = capture.split(',').filter(|s| !s.is_empty());
            let auto_paths = spans.clone().filter(|s| !s.contains('@')).count();
            let mut auto_index = 0;

            for span in spans {
                enabled = true;

                let (span, output_path) = match span.split_once('@') {
                    Some((span, path)) => (span, PathBuf::from(path)),
                    None if auto_paths == 1 => (span, PathBuf::from("/tmp/uzu.gputrace")),
                    None => {
                        let output_path = PathBuf::from(format!("/tmp/uzu-{auto_index}.gputrace"));
                        auto_index += 1;
                        (span, output_path)
                    },
                };
                let (start, end) = span.split_once("..").unwrap_or((span, span));

                let (sender, receiver) = mpsc::channel();

                assert!(!starts.contains_key(start), "duplicate capture start: {start:?}");
                starts.insert(start.to_owned(), (sender, output_path));

                assert!(!ends.contains_key(end), "duplicate capture end: {end:?}");
                ends.insert(end.to_owned(), receiver);
            }
        }

        Self {
            enabled,
            inner: Mutex::new(CaptureMatcherInner {
                starts,
                ends,
            }),
        }
    }

    #[cfg(backend = "metal")]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn capture<B: Backend>(
        &self,
        context: &B::Context,
        span: &str,
    ) -> Result<Option<CaptureMatcherSpan>, CaptureError<B>> {
        if self.enabled {
            let mut inner = self.inner.lock();
            if let Some((start, output_path)) = inner.starts.remove(span) {
                start
                    .send(Box::new(context.start_capture(&output_path).map_err(CaptureError::BackendCapture)?))
                    .map_err(|_| CaptureError::StartAfterEnd)?;
            }
            if let Some(end) = inner.ends.remove(span) {
                return Ok(Some(CaptureMatcherSpan {
                    _end: end,
                }));
            }
        }

        Ok(None)
    }
}

pub struct CaptureMatcherSpan {
    _end: Receiver<Box<dyn Send + Sync>>,
}
unsafe impl Sync for CaptureMatcherSpan {}
