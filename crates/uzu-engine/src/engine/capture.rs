use std::{
    path::PathBuf,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::backends::common::{Backend, Context};

static CAPTURE_FIRST_PREFILL: LazyLock<AtomicBool> =
    LazyLock::new(|| AtomicBool::new(std::env::var("UZU_CAPTURE_FIRST_PREFILL").is_ok()));
static CAPTURE_FIRST_DECODE: LazyLock<AtomicBool> =
    LazyLock::new(|| AtomicBool::new(std::env::var("UZU_CAPTURE_FIRST_DECODE").is_ok()));

pub struct CaptureRequest<B: Backend> {
    context: Arc<B::Context>,
    path: PathBuf,
}

impl<B: Backend> CaptureRequest<B> {
    fn new(
        context: Arc<B::Context>,
        path: PathBuf,
    ) -> Self {
        Self {
            context,
            path,
        }
    }

    pub fn start(self) -> Result<CaptureSpan<B>, B::Error> {
        self.context.start_capture(&self.path)?;
        eprintln!("Started capturing to {:?}", self.path);

        Ok(CaptureSpan {
            context: self.context,
            path: self.path,
        })
    }
}

pub struct CaptureSpan<B: Backend> {
    context: Arc<B::Context>,
    path: PathBuf,
}

impl<B: Backend> Drop for CaptureSpan<B> {
    fn drop(&mut self) {
        match self.context.stop_capture() {
            Ok(()) => {
                eprintln!("Successfully saved capture to {:?}", self.path);
            },
            Err(err) => {
                eprintln!("Error when saving capture to {:?}: {:?}", self.path, err);
            },
        }
    }
}

pub struct CaptureManager<B: Backend> {
    context: Arc<B::Context>,
}

impl<B: Backend> CaptureManager<B> {
    pub fn pre_load_enable() -> bool {
        let capture_enabled =
            CAPTURE_FIRST_PREFILL.load(Ordering::Relaxed) || CAPTURE_FIRST_DECODE.load(Ordering::Relaxed);

        if capture_enabled {
            <B::Context as Context>::enable_capture();
        }

        capture_enabled
    }

    pub fn new(context: Arc<B::Context>) -> Self {
        Self {
            context,
        }
    }

    fn capture_path(suffix: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("uzu-capture-{}-{}", B::NAME, suffix))
    }

    pub fn maybe_capture_prefill_step(&self) -> Option<CaptureRequest<B>> {
        CAPTURE_FIRST_PREFILL
            .swap(false, Ordering::Relaxed)
            .then(|| CaptureRequest::new(self.context.clone(), Self::capture_path("prefill")))
    }

    pub fn maybe_capture_decode_step(&self) -> Option<CaptureRequest<B>> {
        CAPTURE_FIRST_DECODE
            .swap(false, Ordering::Relaxed)
            .then(|| CaptureRequest::new(self.context.clone(), Self::capture_path("decode")))
    }
}
