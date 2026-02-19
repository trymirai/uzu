use std::{
    marker::PhantomData,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    backends::common::{Backend, Context},
    utils::env_utils::EnvVar,
};

pub struct GpuCaptureManager<B: Backend> {
    capture_prefill_enabled: bool,
    capture_decode_enabled: bool,
    first_prefill_captured: bool,
    first_decode_captured: bool,
    phantom: PhantomData<B>,
}

impl<B: Backend> GpuCaptureManager<B> {
    pub fn new() -> Self {
        let capture_prefill_enabled = EnvVar::CaptureFirstPrefill.is_enabled();
        let capture_decode_enabled = EnvVar::CaptureFirstDecode.is_enabled();

        // Enable backend capture BEFORE device creation if any capture is requested
        if capture_prefill_enabled || capture_decode_enabled {
            B::Context::enable_capture();
        }

        Self {
            capture_prefill_enabled,
            capture_decode_enabled,
            first_prefill_captured: false,
            first_decode_captured: false,
            phantom: PhantomData,
        }
    }

    pub fn should_capture_prefill(
        &self,
        is_first_prefill: bool,
    ) -> bool {
        self.capture_prefill_enabled && is_first_prefill && !self.first_prefill_captured
    }

    pub fn should_capture_decode(
        &self,
        is_first_decode: bool,
    ) -> bool {
        self.capture_decode_enabled && is_first_decode && !self.first_decode_captured
    }

    pub fn start_capture(
        &self,
        context: &B::Context,
        capture_type: &str,
    ) -> Result<PathBuf, String> {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| format!("Time error: {}", e))?.as_secs();

        let trace_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(format!("uzu_first_{}-{}.gputrace", capture_type, timestamp));

        context.start_capture(&trace_path).map_err(|err| err.to_string())?;

        println!("GPU capture started for first {}: {:?}", capture_type, trace_path);

        Ok(trace_path)
    }

    pub fn stop_capture(
        &mut self,
        context: &B::Context,
        capture_type: &str,
    ) -> Result<(), String> {
        context.stop_capture().map_err(|err| err.to_string())?;
        println!("GPU capture stopped for {}", capture_type);

        match capture_type {
            "prefill" => self.first_prefill_captured = true,
            "decode" => self.first_decode_captured = true,
            _ => {},
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.first_prefill_captured = false;
        self.first_decode_captured = false;
    }
}
