use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanoCodecFsqRuntimeOptions {
    pub collect_command_buffer_profile: bool,
    pub profile_decoder_micro_stages: bool,
    pub capture_single_decode: bool,
    pub chunked_command_buffers: bool,
    pub micro_flush_min_elements: usize,
}

impl Default for NanoCodecFsqRuntimeOptions {
    fn default() -> Self {
        Self {
            collect_command_buffer_profile: false,
            profile_decoder_micro_stages: false,
            capture_single_decode: false,
            chunked_command_buffers: true,
            micro_flush_min_elements: 8_000_000,
        }
    }
}

impl NanoCodecFsqRuntimeOptions {
    pub(super) fn async_stream_delivery_enabled(self) -> bool {
        !(self.collect_command_buffer_profile || self.capture_single_decode)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioCommandBufferProfile {
    pub label: String,
    pub cpu_encode_ms: f64,
    pub cpu_wait_ms: f64,
    pub gpu_execution_ms: Option<f64>,
    pub estimated_macs: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioDecodeProfile {
    pub batch_size: usize,
    pub frames: usize,
    pub codebooks: usize,
    pub command_buffers: Vec<AudioCommandBufferProfile>,
    pub readback_cpu_ms: f64,
    pub trace_path: Option<PathBuf>,
}

pub struct SubmittedDecodedPaddedAudio<B: Backend> {
    pub(in crate::audio::nanocodec::runtime) output: Array<B>,
    pub(in crate::audio::nanocodec::runtime) channels: usize,
    pub(in crate::audio::nanocodec::runtime) frames: usize,
    pub(in crate::audio::nanocodec::runtime) lengths: Vec<usize>,
    pub(in crate::audio::nanocodec::runtime) final_command_buffer: Option<<B::CommandBuffer as CommandBuffer>::Pending>,
    pub(in crate::audio::nanocodec::runtime) final_command_label: Option<String>,
    pub(in crate::audio::nanocodec::runtime) final_cpu_encode_ms: f64,
    pub(in crate::audio::nanocodec::runtime) decode_profile: Option<AudioDecodeProfile>,
    pub(in crate::audio::nanocodec::runtime) capture: Option<AudioCaptureGuard<B>>,
}

impl<B: Backend> SubmittedDecodedPaddedAudio<B> {
    pub(in crate::audio::nanocodec::runtime) fn is_ready(&self) -> bool {
        self.final_command_buffer.as_ref().is_none_or(|command_buffer| command_buffer.is_completed())
    }

    pub(in crate::audio::nanocodec::runtime) fn resolve(
        mut self
    ) -> AudioResult<(DecodedPaddedAudio, Option<AudioDecodeProfile>)> {
        if let Some(command_buffer) = self.final_command_buffer.take() {
            let wait_start = self.decode_profile.is_some().then(Instant::now);
            let command_buffer = command_buffer.wait_until_completed().map_err(|err| {
                AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
            })?;
            let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
            if let Some(profile) = self.decode_profile.as_mut() {
                profile.command_buffers.push(AudioCommandBufferProfile {
                    label: self.final_command_label.clone().unwrap_or_else(|| "decoder_final".to_string()),
                    cpu_encode_ms: self.final_cpu_encode_ms,
                    cpu_wait_ms,
                    gpu_execution_ms: command_buffer.gpu_execution_time_ms(),
                    estimated_macs: None,
                });
            }
        }
        if let Some(capture) = self.capture.as_mut() {
            let trace_path = capture.stop()?;
            if let Some(profile) = self.decode_profile.as_mut() {
                profile.trace_path = Some(trace_path);
            }
        }
        let readback_start = self.decode_profile.is_some().then(Instant::now);
        let samples = read_array_to_f32_vec(&self.output)?;
        if let Some(profile) = self.decode_profile.as_mut() {
            profile.readback_cpu_ms = readback_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        }
        Ok((
            DecodedPaddedAudio {
                samples,
                channels: self.channels,
                frames: self.frames,
                lengths: self.lengths,
            },
            self.decode_profile,
        ))
    }
}

pub(crate) struct PendingStreamPcmChunk<B: Backend> {
    pub(in crate::audio::nanocodec::runtime) runtime: NanoCodecFsqRuntime<B>,
    pub(in crate::audio::nanocodec::runtime) submitted: SubmittedDecodedPaddedAudio<B>,
    pub(in crate::audio::nanocodec::runtime) previous_audio_lengths: Box<[usize]>,
    pub(in crate::audio::nanocodec::runtime) semantic_lengths: Box<[usize]>,
    pub(in crate::audio::nanocodec::runtime) audio_offset_frames: usize,
    pub(in crate::audio::nanocodec::runtime) upsample_factor: usize,
    pub(in crate::audio::nanocodec::runtime) step_stats: AudioDecodeStepStats,
}

impl<B: Backend> PendingStreamPcmChunk<B> {
    pub(crate) fn is_ready(&self) -> bool {
        self.submitted.is_ready()
    }

    pub(crate) fn step_stats(&self) -> AudioDecodeStepStats {
        self.step_stats
    }

    pub(crate) fn resolve(self) -> AudioResult<AudioPcmBatch> {
        let (decoded_window, decode_profile) = self.submitted.resolve()?;
        *self.runtime.last_decode_profile.borrow_mut() = decode_profile;
        let delta = extract_delta_from_padded_with_offset_snapshot(
            &decoded_window,
            &self.previous_audio_lengths,
            &self.semantic_lengths,
            self.audio_offset_frames,
            self.upsample_factor,
        )?;
        self.runtime.decoded_padded_to_pcm_batch(&delta)
    }
}

pub(in crate::audio::nanocodec::runtime) struct AudioCaptureGuard<B: Backend> {
    context: Rc<B::Context>,
    trace_path: PathBuf,
    active: bool,
}

impl<B: Backend> AudioCaptureGuard<B> {
    pub(in crate::audio::nanocodec::runtime) fn start() -> AudioResult<Self> {
        B::Context::enable_capture();
        let context =
            B::Context::new().map_err(|err| AudioError::Runtime(format!("failed to create capture context: {err}")))?;
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).map_err(|err| AudioError::Runtime(err.to_string()))?;
        let trace_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(format!("uzu_audio_decode-{}.gputrace", timestamp.as_secs()));
        context
            .start_capture(&trace_path)
            .map_err(|err| AudioError::Runtime(format!("failed to start audio GPU capture: {err}")))?;
        Ok(Self {
            context,
            trace_path,
            active: true,
        })
    }

    pub(in crate::audio::nanocodec::runtime) fn context(&self) -> Rc<B::Context> {
        Rc::clone(&self.context)
    }

    pub(in crate::audio::nanocodec::runtime) fn stop(&mut self) -> AudioResult<PathBuf> {
        if self.active {
            self.context
                .stop_capture()
                .map_err(|err| AudioError::Runtime(format!("failed to stop audio GPU capture: {err}")))?;
            self.active = false;
        }
        Ok(self.trace_path.clone())
    }
}

impl<B: Backend> Drop for AudioCaptureGuard<B> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.context.stop_capture();
            self.active = false;
        }
    }
}

pub(in crate::audio::nanocodec::runtime) fn push_audio_command_buffer_profile(
    profile: &mut Option<AudioDecodeProfile>,
    label: impl Into<String>,
    command_buffer: &impl CommandBufferCompleted,
    cpu_encode_ms: f64,
    cpu_wait_ms: f64,
    estimated_macs: Option<usize>,
) {
    if let Some(profile) = profile.as_mut() {
        profile.command_buffers.push(AudioCommandBufferProfile {
            label: label.into(),
            cpu_encode_ms,
            cpu_wait_ms,
            gpu_execution_ms: command_buffer.gpu_execution_time_ms(),
            estimated_macs,
        });
    }
}
