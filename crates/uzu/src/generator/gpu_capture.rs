use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use metal::{CaptureDescriptor, CaptureManager, MTLCaptureDestination};

use crate::{backends::metal::MTLContext, utils::env_utils::MetalEnvVar};

pub struct GpuCaptureManager {
    capture_prefill_enabled: bool,
    capture_decode_enabled: bool,
    first_prefill_captured: bool,
    first_decode_captured: bool,
}

impl GpuCaptureManager {
    pub fn new() -> Self {
        let capture_prefill_enabled = MetalEnvVar::CaptureFirstPrefill.is_enabled();
        let capture_decode_enabled = MetalEnvVar::CaptureFirstDecode.is_enabled();

        // Enable Metal capture layer BEFORE device creation if any capture is requested
        if capture_prefill_enabled || capture_decode_enabled {
            unsafe {
                std::env::set_var("METAL_CAPTURE_ENABLED", "1");
            }
        }

        Self {
            capture_prefill_enabled,
            capture_decode_enabled,
            first_prefill_captured: false,
            first_decode_captured: false,
        }
    }

    pub fn should_capture_prefill(&self, is_first_prefill: bool) -> bool {
        self.capture_prefill_enabled && is_first_prefill && !self.first_prefill_captured
    }

    pub fn should_capture_decode(&self, is_first_decode: bool) -> bool {
        self.capture_decode_enabled && is_first_decode && !self.first_decode_captured
    }

    pub fn start_capture(
        &self,
        mtl_context: &MTLContext,
        capture_type: &str,
    ) -> Result<PathBuf, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("Time error: {}", e))?
            .as_secs();

        let trace_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(format!("uzu_first_{}-{}.gputrace", capture_type, timestamp));

        let capture_manager = CaptureManager::shared();
        let capture_descriptor = CaptureDescriptor::new();
        capture_descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
        capture_descriptor.set_output_url(&trace_path);

        mtl_context.command_queue.set_label("uzu_command_queue");
        capture_descriptor.set_capture_command_queue(&mtl_context.command_queue);

        capture_manager
            .start_capture(&capture_descriptor)
            .map_err(|e| format!("Failed to start GPU capture: {}", e))?;

        println!("🔍 GPU capture started for first {}: {:?}", capture_type, trace_path);

        Ok(trace_path)
    }

    pub fn stop_capture(&mut self, capture_type: &str) {
        CaptureManager::shared().stop_capture();
        println!("✅ GPU capture stopped for {}", capture_type);

        match capture_type {
            "prefill" => self.first_prefill_captured = true,
            "decode" => self.first_decode_captured = true,
            _ => {},
        }
    }

    pub fn reset(&mut self) {
        self.first_prefill_captured = false;
        self.first_decode_captured = false;
    }
}
