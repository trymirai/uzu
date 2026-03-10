use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use half::{bf16, f16};
use serde::Deserialize;

use super::{
    decoder::{
        CausalConv1dJson, CausalConvTranspose1dJson, NanoCodecDecoderGraph, NanoCodecDecoderJson,
        NanoCodecHiFiGanResBlockJson, NanoCodecHiFiGanResLayerJson, NanoCodecResidualBlockJson,
        NanoCodecUpsampleStageJson, Tensor3Json,
    },
    fsq::compute_dim_base_index,
};
use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking},
    backends::{
        common::{
            Backend, Context, CopyEncoder, Kernels,
            kernel::{
                ActivationKernel, AudioAddKernel, AudioCausalConv1dGroupedKernel,
                AudioCausalConv1dGroupedResidualKernel, AudioCausalConv1dKernel,
                AudioCausalConvTranspose1dCausalPadKernel, AudioConv1dKernel, AudioFsqDecodeKernel,
                AudioFsqEncodeKernel, AudioHalfSnakeKernel, AudioNormNcsKernel, AudioQuantizerDecodeKernel,
                AudioTransposeNscToNcsKernel,
            },
        },
        metal::Metal,
    },
    config::{ConfigDataType, EmbeddingConfig, EmbeddingConfigCommon, InnerModelConfig},
    encodable_block::{EncodableBlock, EncodingParameters, LayerExecutables, RMSNorm, Rope},
    forward_pass::{
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, RopeType, SharedBuffers},
    },
    parameters::{ParameterLoader, read_safetensors_metadata},
};

type MetalCommandBuffer = <Metal as Backend>::CommandBuffer;

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

fn usize_to_i32(
    value: usize,
    name: &str,
) -> AudioResult<i32> {
    i32::try_from(value).map_err(|_| AudioError::Runtime(format!("{name} exceeds i32 range")))
}

fn convert_lengths_to_i32(
    lengths: &[usize],
    frames: usize,
) -> AudioResult<Vec<i32>> {
    let mut out = Vec::with_capacity(lengths.len());
    for &length in lengths {
        if length > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length,
                frames,
            });
        }
        out.push(usize_to_i32(length, "length")?);
    }
    Ok(out)
}

fn checked_mul_i32(
    value: i32,
    mul: usize,
) -> AudioResult<i32> {
    i32::try_from(mul)
        .map_err(|_| AudioError::Runtime("length scaling factor exceeds i32 range".to_string()))?
        .checked_mul(value)
        .ok_or(AudioError::Runtime("scaled length overflow".to_string()))
}

fn checked_div_ceil(
    numerator: usize,
    denominator: usize,
) -> AudioResult<usize> {
    if denominator == 0 {
        return Err(AudioError::Runtime("division by zero".to_string()));
    }
    let addend = denominator.saturating_sub(1);
    numerator
        .checked_add(addend)
        .ok_or(AudioError::Runtime("ceil-division overflow".to_string()))
        .map(|value| value / denominator)
}

fn scale_lengths_i32_in_place(
    source: &[i32],
    destination: &mut [i32],
    factor: usize,
) -> AudioResult<()> {
    if destination.len() != source.len() {
        return Err(AudioError::Runtime(format!(
            "scaled length buffer mismatch: expected {}, got {}",
            source.len(),
            destination.len()
        )));
    }
    for (dst, &src) in destination.iter_mut().zip(source.iter()) {
        *dst = checked_mul_i32(src, factor)?;
    }
    Ok(())
}

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
    fn async_stream_delivery_enabled(self) -> bool {
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

#[derive(Clone)]
struct PendingAudioCommandBufferProfile {
    label: String,
    cpu_encode_ms: f64,
    cpu_wait_ms: f64,
    command_buffer: MetalCommandBuffer,
    estimated_macs: Option<usize>,
}

struct SubmittedDecodedPaddedAudio {
    output: Array<Metal>,
    channels: usize,
    frames: usize,
    lengths: Vec<usize>,
    final_command_buffer: Option<MetalCommandBuffer>,
    final_command_label: Option<String>,
    final_cpu_encode_ms: f64,
    submitted_command_buffers: Vec<PendingAudioCommandBufferProfile>,
    decode_profile: Option<AudioDecodeProfile>,
    capture: Option<AudioCaptureGuard>,
}

impl SubmittedDecodedPaddedAudio {
    fn is_ready(&self) -> bool {
        self.final_command_buffer.as_ref().is_none_or(|command_buffer| command_buffer.is_completed())
    }

    fn resolve(mut self) -> AudioResult<(super::decoder::DecodedPaddedAudio, Option<AudioDecodeProfile>)> {
        if let Some(command_buffer) = self.final_command_buffer.as_ref() {
            let wait_start = self.decode_profile.is_some().then(Instant::now);
            if !command_buffer.is_completed() {
                command_buffer.wait_until_completed().map_err(|err| {
                    AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
                })?;
            }
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
        if let Some(profile) = self.decode_profile.as_mut() {
            for pending in self.submitted_command_buffers.drain(..) {
                profile.command_buffers.push(AudioCommandBufferProfile {
                    label: pending.label,
                    cpu_encode_ms: pending.cpu_encode_ms,
                    cpu_wait_ms: pending.cpu_wait_ms,
                    gpu_execution_ms: pending.command_buffer.gpu_execution_time_ms(),
                    estimated_macs: pending.estimated_macs,
                });
            }
        }
        let readback_start = self.decode_profile.is_some().then(Instant::now);
        let samples = read_array_to_f32_vec(&self.output)?;
        if let Some(profile) = self.decode_profile.as_mut() {
            profile.readback_cpu_ms = readback_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        }
        Ok((
            super::decoder::DecodedPaddedAudio {
                samples,
                channels: self.channels,
                frames: self.frames,
                lengths: self.lengths,
            },
            self.decode_profile,
        ))
    }
}

pub(crate) struct PendingStreamPcmChunk {
    runtime: NanoCodecFsqRuntime,
    submitted: SubmittedDecodedPaddedAudio,
    previous_audio_lengths: Box<[usize]>,
    semantic_lengths: Box<[usize]>,
    audio_offset_frames: usize,
    upsample_factor: usize,
    step_stats: AudioDecodeStepStats,
}

impl PendingStreamPcmChunk {
    pub(crate) fn is_ready(&self) -> bool {
        self.submitted.is_ready()
    }

    pub(crate) fn step_stats(&self) -> AudioDecodeStepStats {
        self.step_stats
    }

    pub(crate) fn resolve(self) -> AudioResult<AudioPcmBatch> {
        let (decoded_window, decode_profile) = self.submitted.resolve()?;
        if let Ok(mut last_decode_profile) = self.runtime.last_decode_profile.write() {
            *last_decode_profile = decode_profile;
        }
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

struct AudioCaptureGuard {
    context: Rc<<Metal as Backend>::Context>,
    trace_path: PathBuf,
    active: bool,
}

impl AudioCaptureGuard {
    fn start() -> AudioResult<Self> {
        <Metal as Backend>::Context::enable_capture();
        let context = <Metal as Backend>::Context::new()
            .map_err(|err| AudioError::Runtime(format!("failed to create capture context: {err}")))?;
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

    fn context(&self) -> Rc<<Metal as Backend>::Context> {
        Rc::clone(&self.context)
    }

    fn stop(&mut self) -> AudioResult<PathBuf> {
        if self.active {
            self.context
                .stop_capture()
                .map_err(|err| AudioError::Runtime(format!("failed to stop audio GPU capture: {err}")))?;
            self.active = false;
        }
        Ok(self.trace_path.clone())
    }
}

impl Drop for AudioCaptureGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = self.context.stop_capture();
            self.active = false;
        }
    }
}

fn push_audio_command_buffer_profile(
    profile: &mut Option<AudioDecodeProfile>,
    label: impl Into<String>,
    command_buffer: &MetalCommandBuffer,
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

fn checked_add_usize(
    a: usize,
    b: usize,
    label: &str,
) -> AudioResult<usize> {
    a.checked_add(b).ok_or_else(|| AudioError::Runtime(format!("{label} overflow")))
}

fn conv1d_estimated_macs(
    batch_size: usize,
    seq_len: usize,
    layer: &FishAudioConv1dGpuLayer,
) -> AudioResult<usize> {
    let cin_per_group = layer
        .cin
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len, layer.cout, cin_per_group, layer.kernel_size])
}

fn convtranspose_estimated_macs(
    batch_size: usize,
    seq_len_in: usize,
    layer: &FishAudioConvTranspose1dGpuLayer,
) -> AudioResult<usize> {
    let cout_per_group = layer
        .cout
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped transpose-conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len_in, layer.cin, cout_per_group, layer.kernel_size])
}

fn residual_unit_estimated_macs(
    batch_size: usize,
    seq_len: usize,
    unit: &FishAudioResidualUnitGpuLayer,
) -> AudioResult<usize> {
    let conv1 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv1)?;
    let conv2 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv2)?;
    checked_add_usize(conv1, conv2, "residual-unit estimated MACs")
}

fn fishaudio_dtype_key(data_type: DataType) -> u8 {
    match data_type {
        DataType::F16 => 1,
        DataType::BF16 => 2,
        _ => 0,
    }
}

fn write_f32_slice_to_array(
    array: &mut Array<Metal>,
    values: &[f32],
) -> AudioResult<()> {
    if array.num_elements() != values.len() {
        return Err(AudioError::Runtime(format!(
            "array element count mismatch: expected {}, got {}",
            array.num_elements(),
            values.len()
        )));
    }
    match array.data_type() {
        DataType::F32 => {
            array.as_slice_mut::<f32>().copy_from_slice(values);
            Ok(())
        },
        DataType::F16 => {
            for (dst, &src) in array.as_slice_mut::<f16>().iter_mut().zip(values.iter()) {
                *dst = f16::from_f32(src);
            }
            Ok(())
        },
        DataType::BF16 => {
            for (dst, &src) in array.as_slice_mut::<bf16>().iter_mut().zip(values.iter()) {
                *dst = bf16::from_f32(src);
            }
            Ok(())
        },
        other => Err(AudioError::Runtime(format!("unsupported dtype for f32 array write: {other:?}"))),
    }
}

fn read_array_to_f32_vec(array: &Array<Metal>) -> AudioResult<Vec<f32>> {
    Ok(match array.data_type() {
        DataType::F32 => array.as_slice::<f32>().to_vec(),
        DataType::F16 => array.as_slice::<f16>().iter().copied().map(f32::from).collect(),
        DataType::BF16 => array.as_slice::<bf16>().iter().copied().map(f32::from).collect(),
        other => {
            return Err(AudioError::Runtime(format!("unsupported dtype for f32 array read: {other:?}")));
        },
    })
}

fn array_batch_view(
    array: &Array<Metal>,
    batch_index: usize,
    frames: usize,
    channels: usize,
    active_frames: usize,
) -> AudioResult<Array<Metal>> {
    let batch_stride_bytes = size_for_shape(&[frames, channels], array.data_type());
    let batch_offset = batch_index
        .checked_mul(batch_stride_bytes)
        .and_then(|value| value.checked_add(array.offset()))
        .ok_or(AudioError::Runtime("array batch view offset overflow".to_string()))?;
    if active_frames > frames {
        return Err(AudioError::Runtime("array batch view active_frames exceeds frames".to_string()));
    }
    Ok(unsafe { Array::from_parts(array.buffer_rc(), batch_offset, &[active_frames, channels], array.data_type()) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceLayout {
    Ncs,
    Nsc,
}

impl SequenceLayout {
    fn as_i32(self) -> i32 {
        match self {
            Self::Ncs => 0,
            Self::Nsc => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDecodeStreamingMode {
    IncrementalStateful,
    PrefixFallback,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioDecodeStepStats {
    pub input_frames: usize,
    pub total_semantic_frames: usize,
    pub decoded_window_start_frame: usize,
    pub decoded_window_frames: usize,
}

#[derive(Debug, Clone)]
pub struct AudioDecodeStreamState {
    batch_size: usize,
    codebooks: usize,
    max_workspace_frames: usize,
    mode: AudioDecodeStreamingMode,
    stored_frame_start: usize,
    total_frames_generated: usize,
    row_tokens: Vec<Vec<u32>>,
    flattened_tokens: Vec<u32>,
    window_lengths: Vec<usize>,
    semantic_lengths: Vec<usize>,
    emitted_semantic_lengths: Vec<usize>,
    emitted_audio_lengths: Vec<usize>,
    last_step_stats: AudioDecodeStepStats,
}

impl AudioDecodeStreamState {
    fn new(
        batch_size: usize,
        codebooks: usize,
        max_workspace_frames: usize,
        mode: AudioDecodeStreamingMode,
    ) -> AudioResult<Self> {
        if batch_size == 0 {
            return Err(AudioError::Runtime("stream state batch_size must be > 0".to_string()));
        }
        if codebooks == 0 {
            return Err(AudioError::Runtime("stream state codebooks must be > 0".to_string()));
        }
        if max_workspace_frames == 0 {
            return Err(AudioError::Runtime("stream state max_workspace_frames must be > 0".to_string()));
        }

        let row_count = batch_size
            .checked_mul(codebooks)
            .ok_or(AudioError::Runtime("stream state row count overflow".to_string()))?;
        let mut row_tokens = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            row_tokens.push(Vec::with_capacity(max_workspace_frames));
        }

        Ok(Self {
            batch_size,
            codebooks,
            max_workspace_frames,
            mode,
            stored_frame_start: 0,
            total_frames_generated: 0,
            row_tokens,
            flattened_tokens: Vec::with_capacity(
                row_count
                    .checked_mul(max_workspace_frames)
                    .ok_or(AudioError::Runtime("stream token capacity overflow".to_string()))?,
            ),
            window_lengths: Vec::with_capacity(batch_size),
            semantic_lengths: vec![0; batch_size],
            emitted_semantic_lengths: vec![0; batch_size],
            emitted_audio_lengths: vec![0; batch_size],
            last_step_stats: AudioDecodeStepStats::default(),
        })
    }

    fn total_frames(&self) -> usize {
        self.total_frames_generated
    }

    pub fn last_step_stats(&self) -> AudioDecodeStepStats {
        self.last_step_stats
    }

    fn stored_frames(&self) -> usize {
        self.row_tokens.first().map_or(0, Vec::len)
    }

    fn stored_frame_end(&self) -> usize {
        self.stored_frame_start.saturating_add(self.stored_frames())
    }

    fn append_delta(
        &mut self,
        delta_tokens: &AudioTokenGrid,
    ) -> AudioResult<()> {
        if delta_tokens.batch_size() != self.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                self.batch_size,
                delta_tokens.batch_size()
            )));
        }
        if delta_tokens.codebooks() != self.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                self.codebooks,
                delta_tokens.codebooks()
            )));
        }
        if delta_tokens.frames() == 0 {
            return Ok(());
        }

        let delta_codebook_major = delta_tokens.to_packing(AudioTokenPacking::CodebookMajor);
        let delta_frames = delta_codebook_major.frames();
        let tokens = delta_codebook_major.tokens();
        let target_frames = self.total_frames().saturating_add(delta_frames);
        if self.mode == AudioDecodeStreamingMode::PrefixFallback && target_frames > self.max_workspace_frames {
            return Err(AudioError::Runtime(format!(
                "stream workspace exceeded: target_frames={target_frames}, max_workspace_frames={}",
                self.max_workspace_frames
            )));
        }

        for batch in 0..self.batch_size {
            for codebook in 0..self.codebooks {
                let row_index = batch
                    .checked_mul(self.codebooks)
                    .and_then(|value| value.checked_add(codebook))
                    .ok_or(AudioError::Runtime("stream row index overflow".to_string()))?;
                let row = &mut self.row_tokens[row_index];
                let src_start = row_index
                    .checked_mul(delta_frames)
                    .ok_or(AudioError::Runtime("stream source index overflow".to_string()))?;
                let src_end = src_start
                    .checked_add(delta_frames)
                    .ok_or(AudioError::Runtime("stream source index overflow".to_string()))?;
                row.extend_from_slice(&tokens[src_start..src_end]);
            }
        }
        self.total_frames_generated = target_frames;

        let stored_frames = self.stored_frames();
        if stored_frames > self.max_workspace_frames {
            let evict = stored_frames - self.max_workspace_frames;
            for row in &mut self.row_tokens {
                row.drain(..evict);
            }
            self.stored_frame_start = self.stored_frame_start.saturating_add(evict);
        }

        for (length, &delta_len) in self.semantic_lengths.iter_mut().zip(delta_codebook_major.lengths().iter()) {
            *length = length
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream semantic length overflow".to_string()))?;
        }

        Ok(())
    }

    fn to_full_grid(&mut self) -> AudioResult<AudioTokenGrid> {
        if self.stored_frame_start != 0 {
            return Err(AudioError::Runtime(format!(
                "full-grid decode requires retained prefix, but {} frames were evicted",
                self.stored_frame_start
            )));
        }
        let total_frames = self.stored_frames();
        for row in &self.row_tokens {
            if row.len() != total_frames {
                return Err(AudioError::Runtime("stream row token length mismatch".to_string()));
            }
        }

        let expected_tokens = self
            .batch_size
            .checked_mul(self.codebooks)
            .and_then(|value| value.checked_mul(total_frames))
            .ok_or(AudioError::Runtime("stream token count overflow".to_string()))?;
        if self.flattened_tokens.capacity() < expected_tokens {
            return Err(AudioError::Runtime(format!(
                "stream flattened token capacity exceeded: required={expected_tokens}, capacity={}",
                self.flattened_tokens.capacity()
            )));
        }
        self.flattened_tokens.clear();
        for row in &self.row_tokens {
            self.flattened_tokens.extend_from_slice(row);
        }
        if self.flattened_tokens.len() != expected_tokens {
            return Err(AudioError::Runtime("stream flattened token count mismatch".to_string()));
        }

        AudioTokenGrid::new(
            self.flattened_tokens.clone().into_boxed_slice(),
            self.batch_size,
            self.codebooks,
            total_frames,
            self.semantic_lengths.clone().into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
    }

    fn flatten_window(
        &mut self,
        start_frame: usize,
        end_frame: usize,
    ) -> AudioResult<(&[u32], &[usize], usize)> {
        if end_frame < start_frame {
            return Err(AudioError::Runtime(format!(
                "invalid stream window: start_frame={start_frame}, end_frame={end_frame}"
            )));
        }
        let total_frames = self.total_frames();
        if end_frame > total_frames {
            return Err(AudioError::Runtime(format!(
                "stream window end exceeds token frames: end={end_frame}, total={total_frames}"
            )));
        }
        if start_frame < self.stored_frame_start || end_frame > self.stored_frame_end() {
            return Err(AudioError::Runtime(format!(
                "stream window [{start_frame}, {end_frame}) exceeds retained workspace [{}, {})",
                self.stored_frame_start,
                self.stored_frame_end()
            )));
        }

        let window_frames = end_frame.saturating_sub(start_frame);
        let local_start = start_frame.saturating_sub(self.stored_frame_start);
        let local_end = local_start.saturating_add(window_frames);
        let row_count = self
            .batch_size
            .checked_mul(self.codebooks)
            .ok_or(AudioError::Runtime("stream row count overflow".to_string()))?;
        let required_capacity = row_count
            .checked_mul(window_frames)
            .ok_or(AudioError::Runtime("stream window token capacity overflow".to_string()))?;

        if self.flattened_tokens.capacity() < required_capacity {
            return Err(AudioError::Runtime(format!(
                "stream flattened window capacity exceeded: required={required_capacity}, capacity={}",
                self.flattened_tokens.capacity()
            )));
        }
        self.flattened_tokens.clear();
        for row in &self.row_tokens {
            self.flattened_tokens.extend_from_slice(&row[local_start..local_end]);
        }

        self.window_lengths.clear();
        for &length in &self.semantic_lengths {
            self.window_lengths.push(length.saturating_sub(start_frame));
        }

        Ok((&self.flattened_tokens, &self.window_lengths, window_frames))
    }

    fn extract_delta_padded(
        &mut self,
        full_pcm: &AudioPcmBatch,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if full_pcm.batch_size() != self.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream decoded batch mismatch: expected {}, got {}",
                self.batch_size,
                full_pcm.batch_size()
            )));
        }
        let channels = full_pcm.channels();
        let mut delta_lengths = vec![0_usize; self.batch_size];
        let mut delta_unpacked = Vec::<f32>::new();

        let mut src_offset = 0usize;
        for batch in 0..self.batch_size {
            let full_frames = full_pcm.lengths()[batch];
            let previous_frames = self.emitted_audio_lengths[batch].min(full_frames);
            let delta_frames = full_frames.saturating_sub(previous_frames);
            delta_lengths[batch] = delta_frames;

            let batch_sample_count = full_frames
                .checked_mul(channels)
                .ok_or(AudioError::Runtime("stream batch sample count overflow".to_string()))?;
            let src_end = src_offset
                .checked_add(batch_sample_count)
                .ok_or(AudioError::Runtime("stream sample offset overflow".to_string()))?;
            let batch_slice = &full_pcm.samples()[src_offset..src_end];

            let delta_start = previous_frames
                .checked_mul(channels)
                .ok_or(AudioError::Runtime("stream delta offset overflow".to_string()))?;
            let delta_end = full_frames
                .checked_mul(channels)
                .ok_or(AudioError::Runtime("stream delta offset overflow".to_string()))?;
            delta_unpacked.extend_from_slice(&batch_slice[delta_start..delta_end]);

            src_offset = src_end;
            self.emitted_audio_lengths[batch] = full_frames;
            self.emitted_semantic_lengths[batch] = self.semantic_lengths[batch];
        }

        let (delta_padded, delta_frames) = pack_unpacked_to_padded(&delta_unpacked, channels, &delta_lengths)?;
        Ok(super::decoder::DecodedPaddedAudio {
            samples: delta_padded,
            channels,
            frames: delta_frames,
            lengths: delta_lengths,
        })
    }

    fn extract_delta_from_padded_with_offset(
        &mut self,
        full_padded: &super::decoder::DecodedPaddedAudio,
        audio_offset_frames: usize,
        upsample_factor: usize,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        let previous_audio_lengths = self.emitted_audio_lengths.clone();
        let semantic_lengths = self.semantic_lengths.clone();
        let delta = extract_delta_from_padded_with_offset_snapshot(
            full_padded,
            &previous_audio_lengths,
            &semantic_lengths,
            audio_offset_frames,
            upsample_factor,
        )?;
        for (index, &semantic_length) in semantic_lengths.iter().enumerate() {
            let full_audio_length = semantic_length
                .checked_mul(upsample_factor)
                .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
            self.emitted_audio_lengths[index] = full_audio_length;
            self.emitted_semantic_lengths[index] = semantic_length;
        }
        Ok(delta)
    }

    fn mark_submitted_audio_window(
        &mut self,
        semantic_lengths: &[usize],
        upsample_factor: usize,
    ) -> AudioResult<()> {
        if semantic_lengths.len() != self.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream semantic lengths mismatch: expected {}, got {}",
                self.batch_size,
                semantic_lengths.len()
            )));
        }
        for (index, &semantic_length) in semantic_lengths.iter().enumerate() {
            let full_audio_length = semantic_length
                .checked_mul(upsample_factor)
                .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
            self.emitted_audio_lengths[index] = full_audio_length;
            self.emitted_semantic_lengths[index] = semantic_length;
        }
        Ok(())
    }

    fn record_last_step_stats(
        &mut self,
        input_frames: usize,
        decoded_window_start_frame: usize,
        decoded_window_frames: usize,
    ) {
        self.last_step_stats = AudioDecodeStepStats {
            input_frames,
            total_semantic_frames: self.total_frames_generated,
            decoded_window_start_frame,
            decoded_window_frames,
        };
    }
}

fn extract_delta_from_padded_with_offset_snapshot(
    full_padded: &super::decoder::DecodedPaddedAudio,
    previous_audio_lengths: &[usize],
    semantic_lengths: &[usize],
    audio_offset_frames: usize,
    upsample_factor: usize,
) -> AudioResult<super::decoder::DecodedPaddedAudio> {
    if full_padded.lengths.len() != semantic_lengths.len() || previous_audio_lengths.len() != semantic_lengths.len() {
        return Err(AudioError::Runtime(format!(
            "stream decoded batch mismatch: full_lengths={}, previous_lengths={}, semantic_lengths={}",
            full_padded.lengths.len(),
            previous_audio_lengths.len(),
            semantic_lengths.len()
        )));
    }
    if upsample_factor == 0 {
        return Err(AudioError::Runtime("stream upsample_factor must be > 0".to_string()));
    }

    let batch_size = semantic_lengths.len();
    let channels = full_padded.channels;
    let decoded_frames = full_padded.frames;
    let mut delta_lengths = vec![0usize; batch_size];
    let mut max_delta_frames = 0usize;
    let mut local_delta_ranges = vec![(0usize, 0usize); batch_size];

    for batch in 0..batch_size {
        let full_audio_length = semantic_lengths[batch]
            .checked_mul(upsample_factor)
            .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
        let previous_audio_length = previous_audio_lengths[batch].min(full_audio_length);
        let decoded_batch_length = full_padded.lengths[batch];
        let target_end_local = full_audio_length.saturating_sub(audio_offset_frames).min(decoded_batch_length);
        let target_start_local = previous_audio_length.saturating_sub(audio_offset_frames).min(target_end_local);
        let delta = target_end_local.saturating_sub(target_start_local);
        local_delta_ranges[batch] = (target_start_local, target_end_local);
        delta_lengths[batch] = delta;
        max_delta_frames = max_delta_frames.max(delta);
    }

    let padded_len = checked_product(&[batch_size, channels, max_delta_frames])?;
    let mut delta_padded = vec![0.0_f32; padded_len];
    for (batch, &(start_local, end_local)) in local_delta_ranges.iter().enumerate() {
        let delta_len = end_local.saturating_sub(start_local);
        if delta_len == 0 {
            continue;
        }
        for channel in 0..channels {
            let src_start = (batch * channels + channel)
                .checked_mul(decoded_frames)
                .and_then(|value| value.checked_add(start_local))
                .ok_or(AudioError::Runtime("stream padded source offset overflow".to_string()))?;
            let src_end = src_start
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream padded source offset overflow".to_string()))?;
            let dst_start = (batch * channels + channel)
                .checked_mul(max_delta_frames)
                .ok_or(AudioError::Runtime("stream padded destination offset overflow".to_string()))?;
            let dst_end = dst_start
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream padded destination offset overflow".to_string()))?;
            delta_padded[dst_start..dst_end].copy_from_slice(&full_padded.samples[src_start..src_end]);
        }
    }

    Ok(super::decoder::DecodedPaddedAudio {
        samples: delta_padded,
        channels,
        frames: max_delta_frames,
        lengths: delta_lengths,
    })
}

fn pack_unpacked_to_padded(
    unpacked: &[f32],
    channels: usize,
    lengths: &[usize],
) -> AudioResult<(Vec<f32>, usize)> {
    if channels == 0 {
        return Err(AudioError::InvalidChannelCount);
    }
    let batch_size = lengths.len();
    let frames = lengths.iter().copied().max().unwrap_or(0);
    let expected_unpacked = lengths
        .iter()
        .try_fold(0usize, |acc, &length| acc.checked_add(length.checked_mul(channels)?))
        .ok_or(AudioError::Runtime("stream unpacked size overflow".to_string()))?;
    if unpacked.len() != expected_unpacked {
        return Err(AudioError::InvalidPcmShape {
            expected_samples: expected_unpacked,
            actual_samples: unpacked.len(),
        });
    }

    let padded_len = checked_product(&[batch_size, channels, frames])?;
    let mut padded = vec![0.0_f32; padded_len];
    let mut src_offset = 0usize;
    for (batch, &frame_count) in lengths.iter().enumerate() {
        let batch_samples = frame_count
            .checked_mul(channels)
            .ok_or(AudioError::Runtime("stream batch sample count overflow".to_string()))?;
        for frame in 0..frame_count {
            let src_frame_base = src_offset + frame * channels;
            for channel in 0..channels {
                let dst_index = (batch * channels + channel) * frames + frame;
                padded[dst_index] = unpacked[src_frame_base + channel];
            }
        }
        src_offset = src_offset
            .checked_add(batch_samples)
            .ok_or(AudioError::Runtime("stream source offset overflow".to_string()))?;
    }

    Ok((padded, frames))
}

fn pack_pcm_to_padded(
    pcm: &AudioPcmBatch,
    expected_channels: usize,
) -> AudioResult<(Vec<f32>, Vec<usize>, Vec<i32>, usize)> {
    if pcm.channels() != expected_channels {
        return Err(AudioError::Runtime(format!(
            "pcm channel mismatch: expected {expected_channels}, got {}",
            pcm.channels()
        )));
    }

    let lengths = pcm.lengths().to_vec();
    let frames = lengths.iter().copied().max().unwrap_or(0);
    let lengths_i32 = convert_lengths_to_i32(&lengths, frames)?;
    let padded_len = checked_product(&[pcm.batch_size(), expected_channels, frames])?;
    let mut padded = vec![0.0_f32; padded_len];

    let mut src_offset = 0usize;
    let samples = pcm.samples();
    for (batch, &frame_count) in lengths.iter().enumerate() {
        let sample_count = frame_count
            .checked_mul(expected_channels)
            .ok_or(AudioError::Runtime("pcm frame-count overflow".to_string()))?;
        let src_end =
            src_offset.checked_add(sample_count).ok_or(AudioError::Runtime("pcm indexing overflow".to_string()))?;
        if src_end > samples.len() {
            return Err(AudioError::Runtime("pcm indexing out of bounds".to_string()));
        }

        for frame in 0..frame_count {
            let src_frame_base = src_offset + frame * expected_channels;
            for channel in 0..expected_channels {
                let dst_index = (batch * expected_channels + channel) * frames + frame;
                padded[dst_index] = samples[src_frame_base + channel];
            }
        }

        src_offset = src_end;
    }

    Ok((padded, lengths, lengths_i32, frames))
}

fn unpack_padded_to_pcm(
    padded: &[f32],
    batch_size: usize,
    channels: usize,
    frames: usize,
    lengths: &[usize],
) -> AudioResult<Vec<f32>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }

    let expected_padded = checked_product(&[batch_size, channels, frames])?;
    if padded.len() != expected_padded {
        return Err(AudioError::InvalidPcmShape {
            expected_samples: expected_padded,
            actual_samples: padded.len(),
        });
    }

    let total_frames = lengths.iter().sum::<usize>();
    let packed_len = checked_product(&[total_frames, channels])?;
    let mut packed = vec![0.0_f32; packed_len];
    let mut dst_offset = 0usize;

    for (batch, &frame_count) in lengths.iter().enumerate() {
        if frame_count > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length: frame_count,
                frames,
            });
        }

        for frame in 0..frame_count {
            for channel in 0..channels {
                let src_index = (batch * channels + channel) * frames + frame;
                let dst_index = dst_offset + frame * channels + channel;
                packed[dst_index] = padded[src_index];
            }
        }
        dst_offset = dst_offset
            .checked_add(
                frame_count
                    .checked_mul(channels)
                    .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?,
            )
            .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?;
    }

    Ok(packed)
}

fn default_eps() -> f32 {
    1e-3
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimePacking {
    #[default]
    FrameMajor,
    CodebookMajor,
}

impl From<RuntimePacking> for AudioTokenPacking {
    fn from(value: RuntimePacking) -> Self {
        match value {
            RuntimePacking::FrameMajor => AudioTokenPacking::FrameMajor,
            RuntimePacking::CodebookMajor => AudioTokenPacking::CodebookMajor,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeConfigJson {
    #[serde(default)]
    r#type: Option<String>,
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
    #[serde(default)]
    output_packing: RuntimePacking,
    #[serde(default)]
    decoder: Option<NanoCodecDecoderJson>,
}

fn parse_runtime_config_json(tts_config: &serde_json::Value) -> AudioResult<RuntimeConfigJson> {
    let candidate = tts_config.get("audio_codec").unwrap_or(tts_config);
    let parsed: RuntimeConfigJson = serde_json::from_value(candidate.clone())
        .map_err(|err| AudioError::Runtime(format!("invalid tts audio codec config: {err}")))?;

    if let Some(ref codec_type) = parsed.r#type {
        if codec_type != "nanocodec_fsq" {
            return Err(AudioError::Runtime(format!(
                "unsupported audio codec type '{codec_type}', expected 'nanocodec_fsq'"
            )));
        }
    }

    Ok(parsed)
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoTtsConfigJson {
    audio_decoder_config: LalamoAudioDecoderConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoAudioDecoderConfigJson {
    samplerate: u32,
    quantizer_config: LalamoGroupedQuantizerConfigJson,
    decoder_config: LalamoDecoderConfigJson,
    base_channels: usize,
    up_sample_rates: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilations: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoGroupedQuantizerConfigJson {
    num_groups: usize,
    quantizer_config: LalamoQuantizerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoQuantizerConfigJson {
    num_levels: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoDecoderConfigJson {
    activation_config: LalamoActivationConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoActivationConfigJson {
    #[serde(default = "default_negative_slope")]
    leaky_relu_negative_slope: f32,
}

fn default_negative_slope() -> f32 {
    0.01
}

fn default_decoder_eps() -> f32 {
    1e-9
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioTtsConfigJson {
    #[serde(default)]
    activation_precision: Option<ConfigDataType>,
    audio_decoder_config: FishAudioAudioDecoderConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioAudioDecoderConfigJson {
    samplerate: u32,
    n_codebooks: usize,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    downsample_factor: Vec<usize>,
    decoder_rates: Vec<usize>,
    #[serde(default)]
    precision: Option<ConfigDataType>,
    quantizer_config: FishAudioQuantizerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioQuantizerConfigJson {
    #[serde(default)]
    precision: Option<ConfigDataType>,
    post_module_config: crate::config::TransformerConfig,
    upsampler_config: FishAudioUpsamplerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioUpsamplerConfigJson {
    block_configs: Vec<FishAudioUpsamplingBlockConfigJson>,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioUpsamplingBlockConfigJson {
    convnext_config: FishAudioConvNeXtConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioConvNeXtConfigJson {
    norm_config: FishAudioConvNeXtNormConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioConvNeXtNormConfigJson {
    epsilon: f32,
    #[serde(default)]
    subtract_mean: bool,
    #[serde(default)]
    use_bias: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct MatrixF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl MatrixF32 {
    #[cfg(test)]
    fn row(
        &self,
        index: usize,
    ) -> Option<&[f32]> {
        if index >= self.rows {
            return None;
        }
        let start = index.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.values.get(start..end)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioVectorQuantizer {
    codebook: MatrixF32,
    out_proj: MatrixF32,
    out_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConv1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConvTranspose1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioNormLayer {
    scales: Vec<f32>,
    biases: Option<Vec<f32>>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConvNeXtLayer {
    depthwise_conv: FishAudioConv1dLayer,
    norm: FishAudioNormLayer,
    pwconv1: MatrixF32,
    pwconv1_bias: Vec<f32>,
    pwconv2: MatrixF32,
    pwconv2_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioResidualUnitLayer {
    snake1_alpha: Vec<f32>,
    conv1: FishAudioConv1dLayer,
    snake2_alpha: Vec<f32>,
    conv2: FishAudioConv1dLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioDecoderBlockLayer {
    snake_alpha: Vec<f32>,
    trans_conv: FishAudioConvTranspose1dLayer,
    res_unit1: FishAudioResidualUnitLayer,
    res_unit2: FishAudioResidualUnitLayer,
    res_unit3: FishAudioResidualUnitLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioDecoderGraph {
    first_conv: FishAudioConv1dLayer,
    upsample_blocks: Vec<(FishAudioConvTranspose1dLayer, FishAudioConvNeXtLayer)>,
    decoder_blocks: Vec<FishAudioDecoderBlockLayer>,
    final_snake_alpha: Vec<f32>,
    final_conv: FishAudioConv1dLayer,
    upsample_factor: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioCodecGraph {
    semantic_quantizer: FishAudioVectorQuantizer,
    residual_quantizers: Vec<FishAudioVectorQuantizer>,
    post_module_model_dim: usize,
    post_module_transformer_config: crate::config::TransformerConfig,
    weights_path: String,
    decoder: FishAudioDecoderGraph,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    total_codebooks: usize,
    upsample_factor: usize,
    vocoder_data_type: DataType,
}

#[derive(Debug, Clone)]
struct FishAudioConv1dGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct FishAudioConvTranspose1dGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct FishAudioPointwiseConvGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
}

#[derive(Debug, Clone)]
struct FishAudioNormGpuLayer {
    scales: Array<Metal>,
    bias: Array<Metal>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone)]
struct FishAudioConvNeXtGpuLayer {
    depthwise_conv: FishAudioConv1dGpuLayer,
    norm: FishAudioNormGpuLayer,
    pwconv1: FishAudioPointwiseConvGpuLayer,
    pwconv2: FishAudioPointwiseConvGpuLayer,
}

#[derive(Debug, Clone)]
struct FishAudioResidualUnitGpuLayer {
    snake1_alpha: Array<Metal>,
    conv1: FishAudioConv1dGpuLayer,
    snake2_alpha: Array<Metal>,
    conv2: FishAudioConv1dGpuLayer,
}

#[derive(Debug, Clone)]
struct FishAudioDecoderBlockGpuLayer {
    snake_alpha: Array<Metal>,
    trans_conv: FishAudioConvTranspose1dGpuLayer,
    res_unit1: FishAudioResidualUnitGpuLayer,
    res_unit2: FishAudioResidualUnitGpuLayer,
    res_unit3: FishAudioResidualUnitGpuLayer,
}

#[derive(Debug, Clone)]
struct FishAudioDecoderGpuGraph {
    first_conv: FishAudioConv1dGpuLayer,
    upsample_blocks: Vec<(FishAudioConvTranspose1dGpuLayer, FishAudioConvNeXtGpuLayer)>,
    decoder_blocks: Vec<FishAudioDecoderBlockGpuLayer>,
    final_snake_alpha: Array<Metal>,
    final_conv: FishAudioConv1dGpuLayer,
}

struct FishAudioPostModuleRuntime {
    context: Rc<<Metal as Backend>::Context>,
    model_shape: ModelShape,
    scratch_buffers: ScratchBuffers<Metal>,
    shared_buffers: Rc<RefCell<SharedBuffers<Metal>>>,
    layers: Box<[LayerExecutables<Metal>]>,
    output_norm: RMSNorm<Metal>,
    max_sequence_length: usize,
}

struct FishAudioKernelCache {
    transpose_nsc_to_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioTransposeNscToNcsKernel,
    half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel,
    causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel,
    causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel,
    causal_conv1d_grouped_residual: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel,
    causal_conv_transpose1d_causal_pad:
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel,
    conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel,
    norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel,
    activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel,
    add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel,
}

struct FishAudioQuantizerGpuResources {
    data_type: DataType,
    codebook_dim: usize,
    residual_quantizers: usize,
    semantic_cardinality: usize,
    residual_cardinality: usize,
    kernel: <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel,
    semantic_codebook: Array<Metal>,
    semantic_out_proj: Array<Metal>,
    semantic_out_bias: Array<Metal>,
    residual_codebooks: Array<Metal>,
    residual_out_proj: Array<Metal>,
    residual_out_bias: Array<Metal>,
}

thread_local! {
    static FISHAUDIO_POST_MODULE_RUNTIME_CACHE: RefCell<HashMap<String, Rc<FishAudioPostModuleRuntime>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_DECODE_CONTEXT_CACHE: RefCell<HashMap<String, Rc<<Metal as Backend>::Context>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_KERNEL_CACHE: RefCell<HashMap<usize, Rc<FishAudioKernelCache>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_VOCODER_GRAPH_CACHE: RefCell<HashMap<usize, Rc<FishAudioDecoderGpuGraph>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_QUANTIZER_RESOURCES_CACHE: RefCell<HashMap<usize, Rc<FishAudioQuantizerGpuResources>>> =
        RefCell::new(HashMap::new());
}

#[derive(Debug, Clone)]
struct SafeTensorEntry {
    data_type: DataType,
    shape: Vec<usize>,
    offset: usize,
    size: usize,
}

#[derive(Debug)]
struct SafeTensorReader {
    file: File,
    entries: HashMap<String, SafeTensorEntry>,
}

impl SafeTensorReader {
    fn open(path: &Path) -> AudioResult<Self> {
        let file = File::open(path).map_err(|err| {
            AudioError::Runtime(format!("failed to open safetensors file '{}': {err}", path.display()))
        })?;
        let (global_offset, metadata) = read_safetensors_metadata(&file).map_err(|err| {
            AudioError::Runtime(format!("failed to parse safetensors metadata from '{}': {err}", path.display()))
        })?;

        let mut entries = HashMap::new();
        for (name, tensor) in metadata.tensors {
            let (local_begin, local_end) = tensor.data_offsets;
            let size = local_end
                .checked_sub(local_begin)
                .ok_or(AudioError::Runtime("invalid tensor data offsets in safetensors metadata".to_string()))?;
            let offset = global_offset
                .checked_add(local_begin)
                .ok_or(AudioError::Runtime("safetensors tensor offset overflow".to_string()))?;

            entries.insert(
                name,
                SafeTensorEntry {
                    data_type: tensor.dtype.into(),
                    shape: tensor.shape,
                    offset,
                    size,
                },
            );
        }

        Ok(Self {
            file,
            entries,
        })
    }

    fn read_tensor_f32(
        &self,
        key: &str,
    ) -> AudioResult<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| AudioError::Runtime(format!("missing tensor '{key}' in model.safetensors")))?;

        let num_elements = checked_product(&entry.shape)?;
        let expected_size = num_elements
            .checked_mul(entry.data_type.size_in_bytes())
            .ok_or(AudioError::Runtime(format!("tensor '{key}' byte-size overflow")))?;
        if entry.size != expected_size {
            return Err(AudioError::Runtime(format!(
                "tensor '{key}' size mismatch: expected {expected_size} bytes from shape {:?} and dtype {:?}, got {}",
                entry.shape, entry.data_type, entry.size
            )));
        }

        let mut bytes = vec![0_u8; entry.size];
        self.file.read_exact_at(&mut bytes, entry.offset as u64).map_err(|err| {
            AudioError::Runtime(format!("failed reading tensor '{key}' from model.safetensors: {err}"))
        })?;

        let values = match entry.data_type {
            DataType::F32 => decode_f32_bytes(&bytes),
            DataType::F16 => decode_f16_bytes(&bytes).into_iter().map(f16::to_f32).collect::<Vec<f32>>(),
            DataType::BF16 => decode_bf16_bytes(&bytes).into_iter().map(bf16::to_f32).collect::<Vec<f32>>(),
            other => {
                return Err(AudioError::Runtime(format!(
                    "unsupported tensor dtype for '{key}': {other:?} (expected F32/F16/BF16)"
                )));
            },
        };

        if values.len() != num_elements {
            return Err(AudioError::Runtime(format!(
                "decoded tensor '{key}' element count mismatch: expected {num_elements}, got {}",
                values.len()
            )));
        }

        Ok((entry.shape.clone(), values))
    }
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn decode_f16_bytes(bytes: &[u8]) -> Vec<f16> {
    bytes.chunks_exact(2).map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn decode_bf16_bytes(bytes: &[u8]) -> Vec<bf16> {
    bytes.chunks_exact(2).map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn read_tensor_1d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 1 {
        return Err(AudioError::Runtime(format!("expected rank-1 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(values)
}

fn read_tensor_3d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Tensor3Json> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(Tensor3Json {
        shape: [shape[0], shape[1], shape[2]],
        values,
    })
}

fn read_causal_conv_1d(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<CausalConv1dJson> {
    Ok(CausalConv1dJson {
        weight: read_tensor_3d(reader, &format!("{prefix}.weights"))?,
        bias: read_tensor_1d(reader, &format!("{prefix}.biases"))?,
        dilation,
    })
}

fn convert_lalamo_transpose_weight_oih_to_iog(
    weight: &Tensor3Json,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
) -> AudioResult<Tensor3Json> {
    let [out_channels_in_weight, in_channels_per_group, kernel_size] = weight.shape;
    if out_channels_in_weight != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv out-channel mismatch: expected {out_channels}, got {out_channels_in_weight}"
        )));
    }
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid transpose conv grouping: in_channels={in_channels}, out_channels={out_channels}, groups={groups}"
        )));
    }

    let expected_in_per_group = in_channels / groups;
    if in_channels_per_group != expected_in_per_group {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight shape mismatch: expected in_per_group={expected_in_per_group}, got {in_channels_per_group}"
        )));
    }

    let out_channels_per_group = out_channels / groups;
    let expected_weight_len = checked_product(&[out_channels, in_channels_per_group, kernel_size])?;
    if weight.values.len() != expected_weight_len {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight value length mismatch: expected {expected_weight_len}, got {}",
            weight.values.len()
        )));
    }

    let converted_len = checked_product(&[in_channels, out_channels_per_group, kernel_size])?;
    let mut converted = vec![0.0_f32; converted_len];

    for group in 0..groups {
        let in_base = group * in_channels_per_group;
        let out_base = group * out_channels_per_group;

        for out_idx in 0..out_channels_per_group {
            for in_idx in 0..in_channels_per_group {
                for k in 0..kernel_size {
                    let src_index = ((out_base + out_idx) * in_channels_per_group + in_idx) * kernel_size + k;
                    let dst_index = ((in_base + in_idx) * out_channels_per_group + out_idx) * kernel_size + k;
                    converted[dst_index] = weight.values[src_index];
                }
            }
        }
    }

    Ok(Tensor3Json {
        shape: [in_channels, out_channels_per_group, kernel_size],
        values: converted,
    })
}

fn parse_lalamo_tts_config_json(tts_config: &serde_json::Value) -> AudioResult<LalamoTtsConfigJson> {
    serde_json::from_value(tts_config.clone())
        .map_err(|err| AudioError::Runtime(format!("invalid Lalamo tts_config for NanoCodec runtime import: {err}")))
}

fn build_decoder_json_from_lalamo_export(
    tts_config: &LalamoTtsConfigJson,
    model_weights_path: &Path,
) -> AudioResult<NanoCodecDecoderJson> {
    let cfg = &tts_config.audio_decoder_config;
    let reader = SafeTensorReader::open(model_weights_path)?;

    let pre_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.pre_conv", 1)?;
    let mut stages = Vec::with_capacity(cfg.up_sample_rates.len());

    let mut current_channels = cfg.base_channels;
    for (stage_index, &stride) in cfg.up_sample_rates.iter().enumerate() {
        if stride == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid upsample stride at stage {stage_index}: stride must be > 0"
            )));
        }
        let out_channels = current_channels
            .checked_div(2)
            .ok_or(AudioError::Runtime("decoder channel progression overflow".to_string()))?;
        if out_channels == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid decoder stage {stage_index}: current_channels={current_channels}, expected > 1"
            )));
        }
        let groups = out_channels;

        let activation_alpha =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.activations.{stage_index}.snake.alpha"))?;

        let upsample_weight_oih =
            read_tensor_3d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.weights"))?;
        let upsample_bias =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.biases"))?;
        let upsample_weight =
            convert_lalamo_transpose_weight_oih_to_iog(&upsample_weight_oih, current_channels, out_channels, groups)?;
        let upsample_conv = CausalConvTranspose1dJson {
            weight: upsample_weight,
            bias: upsample_bias,
            stride,
            groups,
        };

        let mut res_blocks = Vec::with_capacity(cfg.resblock_kernel_sizes.len());
        for (res_block_index, _) in cfg.resblock_kernel_sizes.iter().enumerate() {
            let mut residuals = Vec::with_capacity(cfg.resblock_dilations.len());
            for (residual_index, &dilation) in cfg.resblock_dilations.iter().enumerate() {
                let prefix = format!(
                    "audio_decoder.decoder.res_layers.{stage_index}.res_blocks.{res_block_index}.res_blocks.{residual_index}"
                );
                residuals.push(NanoCodecResidualBlockJson {
                    input_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.input_activation.snake.alpha"))?,
                    input_conv: read_causal_conv_1d(&reader, &format!("{prefix}.input_conv"), dilation)?,
                    skip_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.skip_activation.snake.alpha"))?,
                    skip_conv: read_causal_conv_1d(&reader, &format!("{prefix}.skip_conv"), 1)?,
                });
            }
            res_blocks.push(NanoCodecHiFiGanResBlockJson {
                res_blocks: residuals,
            });
        }

        stages.push(NanoCodecUpsampleStageJson {
            activation_alpha,
            upsample_conv,
            res_layer: Some(NanoCodecHiFiGanResLayerJson {
                res_blocks,
            }),
        });

        current_channels = out_channels;
    }

    let post_activation_alpha = read_tensor_1d(&reader, "audio_decoder.decoder.post_activation.snake.alpha")?;
    let post_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.post_conv", 1)?;

    Ok(NanoCodecDecoderJson {
        pre_conv,
        stages,
        post_activation_alpha,
        post_conv,
        negative_slope: cfg.decoder_config.activation_config.leaky_relu_negative_slope,
        eps: default_decoder_eps(),
    })
}

fn parse_fishaudio_tts_config_json(tts_config: &serde_json::Value) -> AudioResult<FishAudioTtsConfigJson> {
    serde_json::from_value(tts_config.clone()).map_err(|err| {
        AudioError::Runtime(format!("invalid Lalamo tts_config for FishAudio DAC runtime import: {err}"))
    })
}

fn resolve_fishaudio_vocoder_data_type(tts_config: &FishAudioTtsConfigJson) -> AudioResult<DataType> {
    let mut resolved_precision: Option<ConfigDataType> = None;
    for (field_name, precision) in [
        ("tts_config.activation_precision", tts_config.activation_precision),
        ("tts_config.audio_decoder_config.precision", tts_config.audio_decoder_config.precision),
        (
            "tts_config.audio_decoder_config.quantizer_config.precision",
            tts_config.audio_decoder_config.quantizer_config.precision,
        ),
    ] {
        if let Some(precision) = precision {
            if let Some(existing) = resolved_precision {
                if existing != precision {
                    return Err(AudioError::Runtime(format!(
                        "conflicting FishAudio precision in Lalamo export: {field_name}={precision:?} conflicts with {existing:?}"
                    )));
                }
            } else {
                resolved_precision = Some(precision);
            }
        }
    }

    let precision = resolved_precision.ok_or(AudioError::Runtime(
        "missing FishAudio precision in Lalamo export; expected one of tts_config.activation_precision, \
tts_config.audio_decoder_config.precision, or tts_config.audio_decoder_config.quantizer_config.precision"
            .to_string(),
    ))?;
    let data_type: DataType = precision.into();
    if !matches!(data_type, DataType::F32 | DataType::F16 | DataType::BF16) {
        return Err(AudioError::Runtime(format!(
            "unsupported FishAudio vocoder precision in Lalamo export: {precision:?} (expected float32/float16/bfloat16)"
        )));
    }
    Ok(data_type)
}

fn read_matrix_f32(
    reader: &SafeTensorReader,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(AudioError::Runtime(format!(
            "tensor '{key}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {shape:?}"
        )));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_matrix_f32_any(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_fishaudio_vector_quantizer(
    reader: &SafeTensorReader,
    prefix: &str,
    codebook_size: usize,
    output_dim: usize,
) -> AudioResult<FishAudioVectorQuantizer> {
    let codebook_key = format!("{prefix}.codebook.weights");
    let (shape, values) = reader.read_tensor_f32(&codebook_key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{codebook_key}', got shape {shape:?}")));
    }
    if shape[0] != codebook_size {
        return Err(AudioError::Runtime(format!(
            "codebook rows mismatch for '{codebook_key}': expected {codebook_size}, got {}",
            shape[0]
        )));
    }
    let code_dim = shape[1];
    if code_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let out_proj = read_matrix_f32(reader, &format!("{prefix}.out_proj.weights"), output_dim, code_dim)?;
    let out_bias = read_tensor_1d(reader, &format!("{prefix}.out_proj.biases"))?;
    if out_bias.len() != output_dim {
        return Err(AudioError::Runtime(format!(
            "out_proj bias shape mismatch for '{prefix}': expected {output_dim}, got {}",
            out_bias.len()
        )));
    }

    Ok(FishAudioVectorQuantizer {
        codebook: MatrixF32 {
            rows: shape[0],
            cols: shape[1],
            values,
        },
        out_proj,
        out_bias,
    })
}

fn read_fishaudio_conv1d_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
    groups: usize,
) -> AudioResult<FishAudioConv1dLayer> {
    let weight_key = format!("{prefix}.weights");
    let bias_key = format!("{prefix}.biases");
    let (shape, values) = reader.read_tensor_f32(&weight_key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{weight_key}', got shape {shape:?}")));
    }
    let bias = read_tensor_1d(reader, &bias_key)?;
    if bias.len() != shape[0] {
        return Err(AudioError::Runtime(format!(
            "bias shape mismatch for '{bias_key}': expected {}, got {}",
            shape[0],
            bias.len()
        )));
    }
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if shape[0] % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv weights for '{weight_key}': out_channels {} not divisible by groups {groups}",
            shape[0]
        )));
    }

    Ok(FishAudioConv1dLayer {
        weight: values,
        bias,
        cin: shape[1].checked_mul(groups).ok_or(AudioError::Runtime("conv input channel overflow".to_string()))?,
        cout: shape[0],
        kernel_size: shape[2],
        dilation,
        groups,
    })
}

fn read_fishaudio_transpose_conv_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    stride: usize,
    groups: usize,
) -> AudioResult<FishAudioConvTranspose1dLayer> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight_oih = read_tensor_3d(reader, &format!("{prefix}.weights"))?;
    let bias = read_tensor_1d(reader, &format!("{prefix}.biases"))?;
    let out_channels = weight_oih.shape[0];
    if bias.len() != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv bias mismatch for '{prefix}.biases': expected {out_channels}, got {}",
            bias.len()
        )));
    }
    let in_channels = weight_oih.shape[1]
        .checked_mul(groups)
        .ok_or(AudioError::Runtime("transpose conv input channel overflow".to_string()))?;
    let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, in_channels, out_channels, groups)?;
    Ok(FishAudioConvTranspose1dLayer {
        weight: converted.values,
        bias,
        cin: in_channels,
        cout: out_channels,
        kernel_size: converted.shape[2],
        stride,
        groups,
    })
}

fn read_fishaudio_residual_unit_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<FishAudioResidualUnitLayer> {
    Ok(FishAudioResidualUnitLayer {
        snake1_alpha: read_tensor_1d(reader, &format!("{prefix}.snake1.alpha"))?,
        conv1: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv1"), dilation, 1)?,
        snake2_alpha: read_tensor_1d(reader, &format!("{prefix}.snake2.alpha"))?,
        conv2: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv2"), 1, 1)?,
    })
}

fn read_fishaudio_norm_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<FishAudioNormLayer> {
    let scales = read_tensor_1d(reader, &format!("{prefix}.scales"))?;
    let biases = if use_bias {
        Some(read_tensor_1d(reader, &format!("{prefix}.biases"))?)
    } else {
        None
    };
    if let Some(biases) = &biases {
        if biases.len() != scales.len() {
            return Err(AudioError::Runtime(format!(
                "norm scale/bias length mismatch at '{prefix}': {} vs {}",
                scales.len(),
                biases.len()
            )));
        }
    }
    Ok(FishAudioNormLayer {
        scales,
        biases,
        epsilon,
        subtract_mean,
    })
}

fn read_fishaudio_convnext_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    norm_config: &FishAudioConvNeXtNormConfigJson,
) -> AudioResult<FishAudioConvNeXtLayer> {
    let depthwise_weight = read_tensor_3d(reader, &format!("{prefix}.dwconv.weights"))?;
    let depthwise_bias = read_tensor_1d(reader, &format!("{prefix}.dwconv.biases"))?;
    if depthwise_weight.shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at {prefix}, got {}",
            depthwise_weight.shape[1]
        )));
    }
    let depthwise_conv = FishAudioConv1dLayer {
        weight: depthwise_weight.values,
        bias: depthwise_bias,
        cin: depthwise_weight.shape[0],
        cout: depthwise_weight.shape[0],
        kernel_size: depthwise_weight.shape[2],
        dilation: 1,
        groups: depthwise_weight.shape[0],
    };
    if depthwise_conv.bias.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise bias mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            depthwise_conv.bias.len()
        )));
    }
    if depthwise_conv.cout == 0 || depthwise_conv.kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if depthwise_conv.cin != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise conv expects cin==cout, got {} vs {} at {prefix}",
            depthwise_conv.cin, depthwise_conv.cout
        )));
    }

    let norm = read_fishaudio_norm_layer(
        reader,
        &format!("{prefix}.norm"),
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
    )?;
    if norm.scales.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt norm channels mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            norm.scales.len()
        )));
    }

    let pwconv1 = read_matrix_f32_any(reader, &format!("{prefix}.pwconv1.weights"))?;
    let pwconv1_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv1.biases"))?;
    if pwconv1.cols != depthwise_conv.cout || pwconv1.rows != pwconv1_bias.len() {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 shape mismatch at {prefix}: weight [{}, {}], bias {}",
            pwconv1.rows,
            pwconv1.cols,
            pwconv1_bias.len()
        )));
    }

    let pwconv2 = read_matrix_f32(reader, &format!("{prefix}.pwconv2.weights"), depthwise_conv.cout, pwconv1.rows)?;
    let pwconv2_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv2.biases"))?;
    if pwconv2_bias.len() != pwconv2.rows {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv2 bias mismatch at {prefix}: expected {}, got {}",
            pwconv2.rows,
            pwconv2_bias.len()
        )));
    }

    Ok(FishAudioConvNeXtLayer {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv1_bias,
        pwconv2,
        pwconv2_bias,
    })
}

fn build_fishaudio_decoder_graph(
    reader: &SafeTensorReader,
    cfg: &FishAudioAudioDecoderConfigJson,
) -> AudioResult<FishAudioDecoderGraph> {
    if cfg.decoder_rates.is_empty() {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.upsampler_config.block_configs.len() != cfg.downsample_factor.len() {
        return Err(AudioError::Runtime(format!(
            "FishAudio upsampler config mismatch: {} block configs for {} downsample factors",
            cfg.quantizer_config.upsampler_config.block_configs.len(),
            cfg.downsample_factor.len()
        )));
    }

    let mut upsample_blocks = Vec::with_capacity(cfg.downsample_factor.len());
    for (index, &stride) in cfg.downsample_factor.iter().rev().enumerate() {
        let trans_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.trans_conv");
        let convnext_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.convnext");
        let trans_conv = read_fishaudio_transpose_conv_layer(reader, &trans_prefix, stride, 1)?;
        let convnext = read_fishaudio_convnext_layer(
            reader,
            &convnext_prefix,
            &cfg.quantizer_config.upsampler_config.block_configs[index].convnext_config.norm_config,
        )?;
        if convnext.depthwise_conv.cin != trans_conv.cout {
            return Err(AudioError::Runtime(format!(
                "FishAudio upsampler convnext channel mismatch at block {index}: trans_conv out {} vs convnext in {}",
                trans_conv.cout, convnext.depthwise_conv.cin
            )));
        }
        upsample_blocks.push((trans_conv, convnext));
    }

    let first_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.first_conv", 1, 1)?;
    let mut decoder_blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
        let base = format!("audio_decoder.decoder.decoder_blocks.{index}");
        decoder_blocks.push(FishAudioDecoderBlockLayer {
            snake_alpha: read_tensor_1d(reader, &format!("{base}.snake.alpha"))?,
            trans_conv: read_fishaudio_transpose_conv_layer(reader, &format!("{base}.trans_conv"), stride, 1)?,
            res_unit1: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit1"), 1)?,
            res_unit2: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit2"), 3)?,
            res_unit3: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit3"), 9)?,
        });
    }

    let final_snake_alpha = read_tensor_1d(reader, "audio_decoder.decoder.final_snake.alpha")?;
    let final_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.final_conv", 1, 1)?;

    let upsample_factor = cfg
        .downsample_factor
        .iter()
        .chain(cfg.decoder_rates.iter())
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("FishAudio decoder upsample factor overflow".to_string()))?;

    Ok(FishAudioDecoderGraph {
        first_conv,
        upsample_blocks,
        decoder_blocks,
        final_snake_alpha,
        final_conv,
        upsample_factor,
    })
}

fn build_fishaudio_codec_graph(
    tts_config: &FishAudioTtsConfigJson,
    model_weights_path: &Path,
) -> AudioResult<FishAudioCodecGraph> {
    let cfg = &tts_config.audio_decoder_config;
    let reader = SafeTensorReader::open(model_weights_path)?;
    if cfg.n_codebooks == 0 || cfg.codebook_size <= 1 || cfg.semantic_codebook_size <= 1 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.post_module_config.model_dim != cfg.input_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio post_module model_dim mismatch: expected {}, got {}",
            cfg.input_dim, cfg.quantizer_config.post_module_config.model_dim
        )));
    }

    let semantic_quantizer = read_fishaudio_vector_quantizer(
        &reader,
        "audio_decoder.quantizer.semantic_quantizer.quantizers.0",
        cfg.semantic_codebook_size,
        cfg.input_dim,
    )?;
    let mut residual_quantizers = Vec::with_capacity(cfg.n_codebooks);
    for index in 0..cfg.n_codebooks {
        let prefix = format!("audio_decoder.quantizer.quantizer.quantizers.{index}");
        residual_quantizers.push(read_fishaudio_vector_quantizer(&reader, &prefix, cfg.codebook_size, cfg.input_dim)?);
    }
    let decoder = build_fishaudio_decoder_graph(&reader, cfg)?;
    let total_codebooks =
        cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime("FishAudio codebook count overflow".to_string()))?;
    let vocoder_data_type = resolve_fishaudio_vocoder_data_type(tts_config)?;

    Ok(FishAudioCodecGraph {
        semantic_quantizer,
        residual_quantizers,
        post_module_model_dim: cfg.quantizer_config.post_module_config.model_dim,
        post_module_transformer_config: cfg.quantizer_config.post_module_config.clone(),
        weights_path: model_weights_path.display().to_string(),
        decoder,
        codebook_size: cfg.codebook_size,
        semantic_codebook_size: cfg.semantic_codebook_size,
        input_dim: cfg.input_dim,
        total_codebooks,
        upsample_factor: cfg
            .downsample_factor
            .iter()
            .chain(cfg.decoder_rates.iter())
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or(AudioError::Runtime("FishAudio upsample factor overflow".to_string()))?,
        vocoder_data_type,
    })
}

fn write_i32_slice_to_array(
    array: &mut Array<Metal>,
    values: &[i32],
    label: &str,
) -> AudioResult<()> {
    if array.data_type() != DataType::I32 {
        return Err(AudioError::Runtime(format!("{label} expected I32 array, got {:?}", array.data_type())));
    }
    if array.num_elements() != values.len() {
        return Err(AudioError::Runtime(format!(
            "{label} length mismatch: expected {}, got {}",
            array.num_elements(),
            values.len()
        )));
    }
    if !values.is_empty() {
        array.as_slice_mut::<i32>().copy_from_slice(values);
    }
    Ok(())
}

fn create_data_array(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    shape: &[usize],
    values: &[f32],
    label: &str,
) -> AudioResult<Array<Metal>> {
    let expected = checked_product(shape)?;
    if expected != values.len() {
        return Err(AudioError::Runtime(format!(
            "tensor '{label}' shape/value mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    let mut array = context.create_array(shape, data_type, label);
    write_f32_slice_to_array(&mut array, values)?;
    Ok(array)
}

fn create_alpha_gpu_array(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    channels: usize,
    alpha: &[f32],
    label: &str,
) -> AudioResult<Array<Metal>> {
    if alpha.len() != channels {
        return Err(AudioError::Runtime(format!(
            "alpha length mismatch for '{label}': expected {channels}, got {}",
            alpha.len()
        )));
    }
    create_data_array(context, data_type, &[channels], alpha, label)
}

fn create_conv1d_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &FishAudioConv1dLayer,
    label_prefix: &str,
) -> AudioResult<FishAudioConv1dGpuLayer> {
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let cin_per_group = layer.cin / layer.groups;
    let weight_shape = [layer.cout, cin_per_group, layer.kernel_size];
    let weight =
        create_data_array(context, data_type, &weight_shape, &layer.weight, &format!("{label_prefix}_weight"))?;
    let bias = create_data_array(context, data_type, &[layer.cout], &layer.bias, &format!("{label_prefix}_bias"))?;
    Ok(FishAudioConv1dGpuLayer {
        weight,
        bias,
        cin: layer.cin,
        cout: layer.cout,
        kernel_size: layer.kernel_size,
        dilation: layer.dilation,
        groups: layer.groups,
    })
}

fn create_conv_transpose1d_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &FishAudioConvTranspose1dLayer,
    label_prefix: &str,
) -> AudioResult<FishAudioConvTranspose1dGpuLayer> {
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight_plane = checked_product(&[layer.cin, layer.cout / layer.groups])?;
    if weight_plane == 0 || layer.weight.len() % weight_plane != 0 {
        return Err(AudioError::Runtime(format!("transpose layer '{label_prefix}' has invalid weight shape")));
    }
    let kernel_size = layer.kernel_size;
    if kernel_size == 0
        || layer.weight.len()
            != weight_plane
                .checked_mul(kernel_size)
                .ok_or(AudioError::Runtime(format!("transpose layer '{label_prefix}' kernel shape overflow")))?
    {
        return Err(AudioError::Runtime(format!(
            "transpose layer '{label_prefix}' has invalid kernel_size={kernel_size} for weight len {}",
            layer.weight.len()
        )));
    }
    let weight = create_data_array(
        context,
        data_type,
        &[layer.cin, layer.cout / layer.groups, kernel_size],
        &layer.weight,
        &format!("{label_prefix}_weight"),
    )?;
    let bias = create_data_array(context, data_type, &[layer.cout], &layer.bias, &format!("{label_prefix}_bias"))?;
    Ok(FishAudioConvTranspose1dGpuLayer {
        weight,
        bias,
        cin: layer.cin,
        cout: layer.cout,
        kernel_size,
        stride: layer.stride,
        groups: layer.groups,
    })
}

fn create_pointwise_conv_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    weight: &MatrixF32,
    bias: &[f32],
    label_prefix: &str,
) -> AudioResult<FishAudioPointwiseConvGpuLayer> {
    if bias.len() != weight.rows {
        return Err(AudioError::Runtime(format!(
            "pointwise layer '{label_prefix}' bias mismatch: expected {}, got {}",
            weight.rows,
            bias.len()
        )));
    }
    let weight_array = create_data_array(
        context,
        data_type,
        &[weight.rows, weight.cols, 1],
        &weight.values,
        &format!("{label_prefix}_weight"),
    )?;
    let bias_array = create_data_array(context, data_type, &[weight.rows], bias, &format!("{label_prefix}_bias"))?;
    Ok(FishAudioPointwiseConvGpuLayer {
        weight: weight_array,
        bias: bias_array,
        cin: weight.cols,
        cout: weight.rows,
    })
}

fn create_norm_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &FishAudioNormLayer,
    channels: usize,
    label_prefix: &str,
) -> AudioResult<FishAudioNormGpuLayer> {
    if layer.scales.len() != channels {
        return Err(AudioError::Runtime(format!(
            "norm layer '{label_prefix}' scale mismatch: expected {channels}, got {}",
            layer.scales.len()
        )));
    }
    let mut bias = vec![0.0_f32; channels];
    if let Some(bias_values) = &layer.biases {
        if bias_values.len() != channels {
            return Err(AudioError::Runtime(format!(
                "norm layer '{label_prefix}' bias mismatch: expected {channels}, got {}",
                bias_values.len()
            )));
        }
        bias.copy_from_slice(bias_values);
    }
    Ok(FishAudioNormGpuLayer {
        scales: create_data_array(context, data_type, &[channels], &layer.scales, &format!("{label_prefix}_scales"))?,
        bias: create_data_array(context, data_type, &[channels], &bias, &format!("{label_prefix}_bias"))?,
        epsilon: layer.epsilon,
        subtract_mean: layer.subtract_mean,
    })
}

fn create_convnext_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &FishAudioConvNeXtLayer,
    label_prefix: &str,
) -> AudioResult<FishAudioConvNeXtGpuLayer> {
    let channels = layer.depthwise_conv.cout;
    Ok(FishAudioConvNeXtGpuLayer {
        depthwise_conv: create_conv1d_gpu_layer(
            context,
            data_type,
            &layer.depthwise_conv,
            &format!("{label_prefix}_dwconv"),
        )?,
        norm: create_norm_gpu_layer(context, data_type, &layer.norm, channels, &format!("{label_prefix}_norm"))?,
        pwconv1: create_pointwise_conv_gpu_layer(
            context,
            data_type,
            &layer.pwconv1,
            &layer.pwconv1_bias,
            &format!("{label_prefix}_pwconv1"),
        )?,
        pwconv2: create_pointwise_conv_gpu_layer(
            context,
            data_type,
            &layer.pwconv2,
            &layer.pwconv2_bias,
            &format!("{label_prefix}_pwconv2"),
        )?,
    })
}

fn create_residual_unit_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &FishAudioResidualUnitLayer,
    channels: usize,
    label_prefix: &str,
) -> AudioResult<FishAudioResidualUnitGpuLayer> {
    Ok(FishAudioResidualUnitGpuLayer {
        snake1_alpha: create_alpha_gpu_array(
            context,
            data_type,
            channels,
            &layer.snake1_alpha,
            &format!("{label_prefix}_snake1"),
        )?,
        conv1: create_conv1d_gpu_layer(context, data_type, &layer.conv1, &format!("{label_prefix}_conv1"))?,
        snake2_alpha: create_alpha_gpu_array(
            context,
            data_type,
            channels,
            &layer.snake2_alpha,
            &format!("{label_prefix}_snake2"),
        )?,
        conv2: create_conv1d_gpu_layer(context, data_type, &layer.conv2, &format!("{label_prefix}_conv2"))?,
    })
}

fn fishaudio_kernels(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
) -> AudioResult<Rc<FishAudioKernelCache>> {
    let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(data_type));
    FISHAUDIO_KERNEL_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Ok(existing.clone());
        }

        let created = Rc::new(FishAudioKernelCache {
            transpose_nsc_to_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioTransposeNscToNcsKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize NSC->NCS transpose kernel: {err}")))?,
            half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize snake1d kernel: {err}")))?,
            causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped_residual:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize grouped residual conv1d kernel: {err}"))
                })?,
            causal_conv_transpose1d_causal_pad:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize causal-pad conv transpose kernel: {err}"))
                })?,
            conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize pointwise conv1d kernel: {err}")))?,
            norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize norm kernel: {err}")))?,
            activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(
                context.as_ref(),
                data_type,
                false,
            )
                .map_err(|err| AudioError::Runtime(format!("failed to initialize activation kernel: {err}")))?,
            add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize add kernel: {err}")))?,
        });

        cache.borrow_mut().insert(key, created.clone());
        Ok(created)
    })
}

fn transpose_nsc_to_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
    channels: usize,
) -> AudioResult<Array<Metal>> {
    let expected = checked_product(&[batch_size, seq_len, channels])?;
    if input.num_elements() != expected {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected,
            actual_tokens: input.num_elements(),
        });
    }

    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_ncs_output");
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let channels_i32 = usize_to_i32(channels, "channels")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.transpose_nsc_to_ncs.encode(
            input.buffer(),
            output.buffer(),
            seq_len_i32,
            channels_i32,
            batch_i32,
            compute_encoder,
        );
    });
    Ok(output)
}

fn snake1d_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    alpha: &Array<Metal>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    if alpha.shape() != [channels] {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: channels,
            actual_tokens: alpha.num_elements(),
        });
    }

    let data_type = input.data_type();
    if alpha.data_type() != data_type {
        return Err(AudioError::Runtime(format!(
            "snake alpha dtype mismatch: expected {:?}, got {:?}",
            data_type,
            alpha.data_type()
        )));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_snake_output");

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.half_snake.encode(
            input.buffer(),
            alpha.buffer(),
            output.buffer(),
            channels_i32,
            seq_len_i32,
            channels_i32,
            0.0,
            1e-9,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn causal_conv1d_grouped_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    layer: &FishAudioConv1dGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cout, layer.cin / layer.groups, layer.kernel_size];
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal conv1d bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("causal conv1d dtype mismatch".to_string()));
    }
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_causal_conv1d_output");

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let kernels = fishaudio_kernels(context, data_type)?;
    if layer.groups == 1 {
        command_buffer.with_compute_encoder(|compute_encoder| {
            kernels.causal_conv1d.encode(
                input.buffer(),
                layer.weight.buffer(),
                layer.bias.buffer(),
                output.buffer(),
                lengths_array.buffer(),
                cin_i32,
                cout_i32,
                seq_len_i32,
                kernel_size_i32,
                dilation_i32,
                batch_i32,
                compute_encoder,
            );
        });
    } else {
        let groups_i32 = usize_to_i32(layer.groups, "groups")?;
        command_buffer.with_compute_encoder(|compute_encoder| {
            kernels.causal_conv1d_grouped.encode(
                input.buffer(),
                layer.weight.buffer(),
                layer.bias.buffer(),
                output.buffer(),
                lengths_array.buffer(),
                cin_i32,
                cout_i32,
                seq_len_i32,
                kernel_size_i32,
                dilation_i32,
                groups_i32,
                batch_i32,
                compute_encoder,
            );
        });
    }

    Ok(output)
}

fn causal_conv1d_grouped_residual_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    residual: &Array<Metal>,
    layer: &FishAudioConv1dGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cout, layer.cin / layer.groups, layer.kernel_size];
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal conv1d bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    let expected_residual = checked_product(&[batch_size, layer.cout, seq_len])?;
    if residual.num_elements() != expected_residual {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_residual,
            actual_tokens: residual.num_elements(),
        });
    }
    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let data_type = input.data_type();
    if residual.data_type() != data_type || layer.weight.data_type() != data_type || layer.bias.data_type() != data_type
    {
        return Err(AudioError::Runtime("causal conv1d residual dtype mismatch".to_string()));
    }
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_causal_conv1d_output");
    let kernels = fishaudio_kernels(context, data_type)?;

    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.causal_conv1d_grouped_residual.encode(
            input.buffer(),
            residual.buffer(),
            layer.weight.buffer(),
            layer.bias.buffer(),
            output.buffer(),
            lengths_array.buffer(),
            cin_i32,
            cout_i32,
            seq_len_i32,
            kernel_size_i32,
            dilation_i32,
            groups_i32,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn causal_conv_transpose1d_causal_pad_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    layer: &FishAudioConvTranspose1dGpuLayer,
    lengths: &[i32],
    batch_size: usize,
    seq_len_in: usize,
    seq_len_out: usize,
    input_layout: SequenceLayout,
    lengths_array: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len_in])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cin, layer.cout / layer.groups, layer.kernel_size];
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal transpose weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal transpose bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }

    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("causal transpose dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(
        &[batch_size, layer.cout, seq_len_out],
        data_type,
        "fishaudio_causal_conv_transpose_output",
    );

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_in_i32 = usize_to_i32(seq_len_in, "seq_len_in")?;
    let seq_out_i32 = usize_to_i32(seq_len_out, "seq_len_out")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let stride_i32 = usize_to_i32(layer.stride, "stride")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let input_layout_i32 = input_layout.as_i32();
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;

    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.causal_conv_transpose1d_causal_pad.encode(
            input.buffer(),
            layer.weight.buffer(),
            layer.bias.buffer(),
            output.buffer(),
            lengths_array.buffer(),
            cin_i32,
            cout_i32,
            seq_in_i32,
            seq_out_i32,
            kernel_size_i32,
            stride_i32,
            groups_i32,
            input_layout_i32,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn conv1d_pointwise_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    layer: &FishAudioPointwiseConvGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::Runtime(format!(
            "pointwise conv input shape mismatch: expected {expected_input} elements ([batch={batch_size}, cin={}, seq_len={seq_len}]), got {}",
            layer.cin,
            input.num_elements()
        )));
    }
    if layer.weight.shape() != [layer.cout, layer.cin, 1] {
        return Err(AudioError::Runtime(format!(
            "pointwise conv weight shape mismatch: expected [{}, {}, 1], got {:?}",
            layer.cout,
            layer.cin,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "pointwise conv bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }

    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("pointwise conv dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_pwconv_output");

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.conv1d.encode(
            input.buffer(),
            layer.weight.buffer(),
            layer.bias.buffer(),
            output.buffer(),
            lengths_array.buffer(),
            cin_i32,
            cout_i32,
            seq_len_i32,
            seq_len_i32,
            1,
            1,
            1,
            0,
            0,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn norm_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
    norm: &FishAudioNormGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if norm.scales.shape() != [channels] {
        return Err(AudioError::Runtime(format!(
            "norm scale shape mismatch: expected [{channels}], got {:?}",
            norm.scales.shape()
        )));
    }
    if norm.bias.shape() != [channels] {
        return Err(AudioError::Runtime(format!(
            "norm bias shape mismatch: expected [{channels}], got {:?}",
            norm.bias.shape()
        )));
    }

    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }

    let data_type = input.data_type();
    if norm.scales.data_type() != data_type || norm.bias.data_type() != data_type {
        return Err(AudioError::Runtime("norm dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_norm_output");

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let subtract_mean = if norm.subtract_mean {
        1
    } else {
        0
    };
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.norm_ncs.encode(
            input.buffer(),
            norm.scales.buffer(),
            norm.bias.buffer(),
            output.buffer(),
            lengths_array.buffer(),
            channels_i32,
            seq_len_i32,
            norm.epsilon,
            subtract_mean,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn gelu_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(input.shape(), data_type, "fishaudio_gelu_output");
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("gelu element count exceeds u32 range".to_string()))?;
    let gelu_id = 1_u32;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels
            .activation
            .encode(Some(input.buffer()), output.buffer(), n_u32, gelu_id, compute_encoder);
    });
    Ok(output)
}

fn add_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    a: &Array<Metal>,
    b: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    if a.num_elements() != b.num_elements() {
        return Err(AudioError::Runtime(format!(
            "elementwise add shape mismatch: {} vs {}",
            a.num_elements(),
            b.num_elements()
        )));
    }
    if a.data_type() != b.data_type() {
        return Err(AudioError::Runtime(format!(
            "elementwise add dtype mismatch: {:?} vs {:?}",
            a.data_type(),
            b.data_type()
        )));
    }
    let data_type = a.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(a.shape(), data_type, "fishaudio_add_out");
    let n_i32 = usize_to_i32(a.num_elements(), "n")?;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.add.encode(a.buffer(), b.buffer(), output.buffer(), n_i32, compute_encoder);
    });
    Ok(output)
}

fn tanh_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut <Metal as Backend>::CommandBuffer,
    input: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(input.shape(), data_type, "fishaudio_tanh_output");
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("tanh element count exceeds u32 range".to_string()))?;
    let tanh_id = 2_u32;
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels
            .activation
            .encode(Some(input.buffer()), output.buffer(), n_u32, tanh_id, compute_encoder);
    });
    Ok(output)
}

impl FishAudioCodecGraph {
    fn conv1d_input_context(layer: &FishAudioConv1dLayer) -> AudioResult<usize> {
        layer
            .kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(layer.dilation))
            .ok_or(AudioError::Runtime("conv1d streaming context overflow".to_string()))
    }

    fn convtranspose_input_context(
        output_context: usize,
        layer: &FishAudioConvTranspose1dLayer,
    ) -> AudioResult<usize> {
        let kernel_minus_one = layer.kernel_size.saturating_sub(1);
        let numerator = output_context
            .checked_add(kernel_minus_one)
            .ok_or(AudioError::Runtime("transpose streaming context overflow".to_string()))?;
        checked_div_ceil(numerator, layer.stride)
    }

    fn residual_unit_input_context(
        output_context: usize,
        layer: &FishAudioResidualUnitLayer,
    ) -> AudioResult<usize> {
        let conv2 = Self::conv1d_input_context(&layer.conv2)?;
        let conv1 = Self::conv1d_input_context(&layer.conv1)?;
        output_context
            .checked_add(conv2)
            .and_then(|value| value.checked_add(conv1))
            .ok_or(AudioError::Runtime("residual-unit streaming context overflow".to_string()))
    }

    fn convnext_input_context(
        output_context: usize,
        layer: &FishAudioConvNeXtLayer,
    ) -> AudioResult<usize> {
        output_context
            .checked_add(Self::conv1d_input_context(&layer.depthwise_conv)?)
            .ok_or(AudioError::Runtime("convnext streaming context overflow".to_string()))
    }

    fn decoder_block_input_context(
        output_context: usize,
        layer: &FishAudioDecoderBlockLayer,
    ) -> AudioResult<usize> {
        let after_res3 = Self::residual_unit_input_context(output_context, &layer.res_unit3)?;
        let after_res2 = Self::residual_unit_input_context(after_res3, &layer.res_unit2)?;
        let after_res1 = Self::residual_unit_input_context(after_res2, &layer.res_unit1)?;
        Self::convtranspose_input_context(after_res1, &layer.trans_conv)
    }

    fn streaming_vocoder_context_frames(&self) -> AudioResult<usize> {
        let mut context = Self::conv1d_input_context(&self.decoder.final_conv)?;

        for block in self.decoder.decoder_blocks.iter().rev() {
            context = Self::decoder_block_input_context(context, block)?;
        }

        context = context
            .checked_add(Self::conv1d_input_context(&self.decoder.first_conv)?)
            .ok_or(AudioError::Runtime("decoder-first streaming context overflow".to_string()))?;

        for (trans_conv, convnext) in self.decoder.upsample_blocks.iter().rev() {
            context = Self::convnext_input_context(context, convnext)?;
            context = Self::convtranspose_input_context(context, trans_conv)?;
        }

        Ok(context)
    }

    fn post_module_streaming_context_frames(&self) -> Option<usize> {
        let mut context = 0usize;
        for layer in &self.post_module_transformer_config.layer_configs {
            let window = layer.mixer_config.sliding_window_size()?;
            context = context.max(window.saturating_sub(1));
        }
        Some(context)
    }

    fn streaming_decode_context_frames(&self) -> AudioResult<Option<usize>> {
        let vocoder_context = self.streaming_vocoder_context_frames()?;
        let Some(post_module_context) = self.post_module_streaming_context_frames() else {
            return Ok(None);
        };
        Ok(Some(vocoder_context.max(post_module_context)))
    }

    #[cfg(test)]
    fn add_quantized_code_to_latent(
        target: &mut [f32],
        quantizer: &FishAudioVectorQuantizer,
        code_index: usize,
    ) -> AudioResult<()> {
        let code_row = quantizer
            .codebook
            .row(code_index)
            .ok_or(AudioError::Runtime("quantizer code index out of range".to_string()))?;
        if quantizer.out_proj.cols != code_row.len() || quantizer.out_proj.rows != target.len() {
            return Err(AudioError::Runtime("quantizer projection shape mismatch".to_string()));
        }
        for (row_index, row) in quantizer.out_proj.values.chunks_exact(quantizer.out_proj.cols).enumerate() {
            let mut acc = quantizer.out_bias[row_index];
            for (&w, &x) in row.iter().zip(code_row.iter()) {
                acc += w * x;
            }
            target[row_index] += acc;
        }
        Ok(())
    }

    #[cfg(test)]
    fn decode_quantizer_to_nsc_reference(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio codebook mismatch: expected {}, got {codebooks}",
                self.total_codebooks
            )));
        }
        let expected_tokens = checked_product(&[batch_size, codebooks, frames])?;
        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }

        let mut latent_nsc = vec![0.0_f32; checked_product(&[batch_size, frames, self.input_dim])?];
        for batch in 0..batch_size {
            let active_frames = lengths[batch];
            if active_frames > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_frames,
                    frames,
                });
            }
            for frame in 0..active_frames {
                let row_start = (batch * frames + frame) * self.input_dim;
                let row_end = row_start + self.input_dim;
                let target = &mut latent_nsc[row_start..row_end];

                let semantic_token_index = ((batch * codebooks) * frames) + frame;
                let semantic_token = tokens[semantic_token_index] as usize;
                let semantic_clamped = semantic_token.min(self.semantic_codebook_size.saturating_sub(1));
                Self::add_quantized_code_to_latent(target, &self.semantic_quantizer, semantic_clamped)?;

                for (residual_index, quantizer) in self.residual_quantizers.iter().enumerate() {
                    let token_index = ((batch * codebooks + (residual_index + 1)) * frames) + frame;
                    let token = tokens[token_index] as usize;
                    let clamped = token.min(self.codebook_size.saturating_sub(1));
                    Self::add_quantized_code_to_latent(target, quantizer, clamped)?;
                }
            }
        }

        Ok(latent_nsc)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn decode_quantizer_to_nsc(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        let context = self.decode_context()?;
        self.decode_quantizer_to_nsc_on_context(&context, tokens, lengths, batch_size, codebooks, frames)
    }

    fn decode_quantizer_to_nsc_array_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio codebook mismatch: expected {}, got {codebooks}",
                self.total_codebooks
            )));
        }
        let expected_tokens = checked_product(&[batch_size, codebooks, frames])?;
        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }
        for &length in lengths {
            if length > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length,
                    frames,
                });
            }
        }
        let quantizer_resources = self.quantizer_gpu_resources(context)?;
        if quantizer_resources.residual_quantizers + 1 != codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio residual quantizer count mismatch: expected {}, got {}",
                codebooks.saturating_sub(1),
                quantizer_resources.residual_quantizers
            )));
        }

        let tokens_i32 = tokens
            .iter()
            .copied()
            .map(|token| {
                i32::try_from(token).map_err(|_| AudioError::Runtime(format!("token id out of i32 range: {token}")))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let lengths_i32 = convert_lengths_to_i32(lengths, frames)?;

        let mut tokens_array =
            context.create_array(&[batch_size, codebooks, frames], DataType::I32, "fishaudio_quantizer_tokens");
        tokens_array.as_slice_mut::<i32>().copy_from_slice(&tokens_i32);
        let mut lengths_array = context.create_array(&[batch_size], DataType::I32, "fishaudio_quantizer_lengths");
        lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let output = context.create_array(
            &[batch_size, frames, self.input_dim],
            quantizer_resources.data_type,
            "fishaudio_quantizer_output_nsc",
        );
        let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
        let codebooks_i32 = usize_to_i32(codebooks, "codebooks")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let input_dim_i32 = usize_to_i32(self.input_dim, "input_dim")?;
        let codebook_dim_i32 = usize_to_i32(quantizer_resources.codebook_dim, "codebook_dim")?;
        let residual_quantizers_i32 = usize_to_i32(quantizer_resources.residual_quantizers, "residual_quantizers")?;
        let semantic_cardinality_i32 = usize_to_i32(quantizer_resources.semantic_cardinality, "semantic_cardinality")?;
        let residual_cardinality_i32 = usize_to_i32(quantizer_resources.residual_cardinality, "residual_cardinality")?;

        let encode_start = profile.is_some().then(Instant::now);
        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create quantizer command buffer: {err}")))?;
        command_buffer.with_compute_encoder(|compute_encoder| {
            quantizer_resources.kernel.encode(
                tokens_array.buffer(),
                lengths_array.buffer(),
                quantizer_resources.semantic_codebook.buffer(),
                quantizer_resources.semantic_out_proj.buffer(),
                quantizer_resources.semantic_out_bias.buffer(),
                quantizer_resources.residual_codebooks.buffer(),
                quantizer_resources.residual_out_proj.buffer(),
                quantizer_resources.residual_out_bias.buffer(),
                output.buffer(),
                batch_i32,
                codebooks_i32,
                frames_i32,
                input_dim_i32,
                codebook_dim_i32,
                residual_quantizers_i32,
                semantic_cardinality_i32,
                residual_cardinality_i32,
                compute_encoder,
            );
        });
        let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        command_buffer.submit();
        let wait_start = profile.is_some().then(Instant::now);
        command_buffer.wait_until_completed().map_err(|err| {
            AudioError::Runtime(format!("failed to wait for quantizer command buffer: {err}"))
        })?;
        let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        push_audio_command_buffer_profile(profile, "quantizer", &command_buffer, cpu_encode_ms, cpu_wait_ms, None);

        Ok(output)
    }

    fn decode_quantizer_to_nsc_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        let mut profile = None;
        let output = self.decode_quantizer_to_nsc_array_on_context(
            context,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
            &mut profile,
        )?;
        read_array_to_f32_vec(&output)
    }

    #[cfg(test)]
    fn apply_norm_sequence(
        values: &mut [f32],
        seq_len: usize,
        channels: usize,
        norm: &FishAudioNormLayer,
    ) -> AudioResult<()> {
        if norm.scales.len() != channels {
            return Err(AudioError::Runtime(format!(
                "norm scale length mismatch: expected {channels}, got {}",
                norm.scales.len()
            )));
        }
        if let Some(biases) = &norm.biases {
            if biases.len() != channels {
                return Err(AudioError::Runtime(format!(
                    "norm bias length mismatch: expected {channels}, got {}",
                    biases.len()
                )));
            }
        }
        let expected = checked_product(&[seq_len, channels])?;
        if values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: values.len(),
            });
        }

        for token in 0..seq_len {
            let row_start = token * channels;
            let row = &mut values[row_start..row_start + channels];
            let mean = if norm.subtract_mean {
                row.iter().sum::<f32>() / channels as f32
            } else {
                0.0
            };
            let variance_sum = if norm.subtract_mean {
                row.iter()
                    .map(|&value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
            } else {
                row.iter().map(|&value| value * value).sum::<f32>()
            };
            let variance = variance_sum / channels as f32;

            let inv_std = 1.0 / (variance + norm.epsilon).sqrt();
            for channel in 0..channels {
                let mut normalized = if norm.subtract_mean {
                    row[channel] - mean
                } else {
                    row[channel]
                };
                normalized *= inv_std * norm.scales[channel];
                if let Some(biases) = &norm.biases {
                    normalized += biases[channel];
                }
                row[channel] = normalized;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn linear_sequence(
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        weight: &MatrixF32,
        bias: Option<&[f32]>,
    ) -> AudioResult<Vec<f32>> {
        let expected_input = checked_product(&[seq_len, in_dim])?;
        if input.len() != expected_input {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_input,
                actual_tokens: input.len(),
            });
        }
        if weight.cols != in_dim {
            return Err(AudioError::Runtime(format!(
                "linear input dim mismatch: expected {}, got {}",
                weight.cols, in_dim
            )));
        }
        if let Some(bias_values) = bias {
            if bias_values.len() != weight.rows {
                return Err(AudioError::Runtime(format!(
                    "linear bias shape mismatch: expected {}, got {}",
                    weight.rows,
                    bias_values.len()
                )));
            }
        }

        let mut output = vec![0.0_f32; checked_product(&[seq_len, weight.rows])?];
        for token in 0..seq_len {
            let input_row = &input[token * in_dim..(token + 1) * in_dim];
            let output_row = &mut output[token * weight.rows..(token + 1) * weight.rows];
            for row_index in 0..weight.rows {
                let mut acc = bias.map_or(0.0, |bias_values| bias_values[row_index]);
                let row = &weight.values[row_index * weight.cols..(row_index + 1) * weight.cols];
                for (&weight_value, &input_value) in row.iter().zip(input_row.iter()) {
                    acc += weight_value * input_value;
                }
                output_row[row_index] = acc;
            }
        }
        Ok(output)
    }

    fn apply_convnext_ncs_enqueued(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        command_buffer: &mut <Metal as Backend>::CommandBuffer,
        input: &Array<Metal>,
        layer: &FishAudioConvNeXtGpuLayer,
        lengths: &[i32],
        lengths_array: &Array<Metal>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Array<Metal>> {
        let residual = input.clone();
        let x = causal_conv1d_grouped_enqueue(
            context,
            command_buffer,
            input,
            &layer.depthwise_conv,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = norm_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.norm,
            lengths,
            lengths_array,
            batch_size,
            channels,
            seq_len,
        )?;
        let x = conv1d_pointwise_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.pwconv1,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = gelu_enqueue(context, command_buffer, &x)?;
        let x = conv1d_pointwise_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.pwconv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        add_enqueue(context, command_buffer, &x, &residual)
    }

    fn build_post_module_runtime(
        &self,
        context: Rc<<Metal as Backend>::Context>,
        required_sequence_length: usize,
    ) -> AudioResult<FishAudioPostModuleRuntime> {
        let inner_model_config = InnerModelConfig {
            embedding_config: EmbeddingConfig::Untied {
                common: EmbeddingConfigCommon {
                    input_scale: None,
                    logit_soft_cap: None,
                },
                precision: self.vocoder_data_type.into(),
            },
            transformer_config: self.post_module_transformer_config.clone(),
            vocab_size: 1,
        };
        let decoder_config =
            Rc::new(inner_model_config.to_decoder_config().map_err(|_| {
                AudioError::Runtime("failed to build FishAudio post_module decoder config".to_string())
            })?);
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!("failed to open post_module weights '{}': {err}", self.weights_path))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!("failed to load post_module weights '{}': {err}", self.weights_path))
        })?;
        let root_loader_view = loader.tree();
        let transformer_subtree_name = "audio_decoder.quantizer.post_module";
        let transformer_tree = root_loader_view
            .subtree(transformer_subtree_name)
            .map_err(|err| AudioError::Runtime(format!("missing FishAudio post_module subtree: {err}")))?;

        let max_sequence_length = decoder_config.context_length.max(required_sequence_length.max(1));
        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        shared_buffers.borrow_mut().update_data_with_transformer_subtree(&root_loader_view, transformer_subtree_name);
        let scratch_buffers = ScratchBuffers::new(
            context.as_ref(),
            &decoder_config,
            &model_shape,
            max_sequence_length,
            max_sequence_length,
        );

        let attention_data_type = (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config
                .attention_config()
                .map(|attention_config| attention_config.qkv_projection_config.activation_precision().into())
        });
        let attention_data_type =
            attention_data_type.ok_or(AudioError::Runtime("post_module has no attention layers".to_string()))?;

        let create_rope_block = |rope_type: RopeType| -> AudioResult<Rc<Box<dyn EncodableBlock<Metal>>>> {
            let rope = Rope::<Metal>::new(context.as_ref(), attention_data_type, rope_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize post_module rope block: {err}")))?;
            Ok(Rc::new(Box::new(rope)))
        };
        let global_rope =
            decoder_config.global_rope_config.as_ref().map(|_| create_rope_block(RopeType::Global)).transpose()?;

        let mut layers = Vec::with_capacity(decoder_config.num_layers);
        for layer_index in 0..decoder_config.num_layers {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            let layer_type = model_shape.layer_type(layer_index);
            let rope_for_layer = match layer_type {
                crate::config::DecoderLayerType::Transformer => Some(
                    global_rope.clone().ok_or(AudioError::Runtime("post_module missing global rope".to_string()))?,
                ),
                _ => None,
            };
            let layer_loader = transformer_tree
                .subtree(&format!("layers.{layer_index}"))
                .map_err(|err| AudioError::Runtime(format!("failed to load post_module layer {layer_index}: {err}")))?;

            layers.push(LayerExecutables::new(
                context.clone(),
                layer_config,
                layer_type,
                layer_index,
                decoder_config.model_dim,
                decoder_config.hidden_dim,
                decoder_config.num_heads,
                decoder_config.head_dim,
                decoder_config.num_groups,
                decoder_config.attention_scale,
                &layer_loader,
                rope_for_layer,
            ));
        }

        let norm_reference_layer =
            decoder_config.layer_configs.as_ref().map(|configs| &configs[0]).unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config {
            crate::config::MixerConfig::Attention(attention_config) => {
                attention_config.qkv_projection_config.activation_precision().into()
            },
            crate::config::MixerConfig::Mamba(mamba_config) => {
                mamba_config.in_projection_config.activation_precision().into()
            },
            crate::config::MixerConfig::ShortConv(short_conv_config) => {
                short_conv_config.in_projection_config.activation_precision().into()
            },
        };
        let output_norm_tree = transformer_tree
            .subtree("output_norm")
            .map_err(|err| AudioError::Runtime(format!("failed to load post_module output_norm: {err}")))?;
        let output_norm = RMSNorm::new(
            context.as_ref(),
            norm_data_type,
            decoder_config.output_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &output_norm_tree,
        )
        .map_err(|err| AudioError::Runtime(format!("failed to build post_module output_norm: {err}")))?;

        Ok(FishAudioPostModuleRuntime {
            context,
            model_shape,
            scratch_buffers,
            shared_buffers,
            layers: layers.into_boxed_slice(),
            output_norm,
            max_sequence_length,
        })
    }

    fn post_module_runtime_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        required_sequence_length: usize,
    ) -> AudioResult<Rc<FishAudioPostModuleRuntime>> {
        let cache_key = format!("{}::{}", self.weights_path, Rc::as_ptr(context) as usize);
        FISHAUDIO_POST_MODULE_RUNTIME_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(runtime) = cache.get(cache_key.as_str()) {
                if runtime.max_sequence_length >= required_sequence_length.max(1) {
                    return Ok(runtime.clone());
                }
            }

            let runtime = Rc::new(self.build_post_module_runtime(context.clone(), required_sequence_length)?);
            cache.insert(cache_key, runtime.clone());
            Ok(runtime)
        })
    }

    fn decode_context(&self) -> AudioResult<Rc<<Metal as Backend>::Context>> {
        FISHAUDIO_DECODE_CONTEXT_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&self.weights_path).cloned() {
                return Ok(existing);
            }
            let created = <Metal as Backend>::Context::new()
                .map_err(|err| AudioError::Runtime(format!("failed to create FishAudio decode context: {err}")))?;
            cache.borrow_mut().insert(self.weights_path.clone(), created.clone());
            Ok(created)
        })
    }

    fn build_quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<FishAudioQuantizerGpuResources> {
        if self.semantic_codebook_size == 0 || self.codebook_size == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        let codebook_dim = self.semantic_quantizer.codebook.cols;
        if codebook_dim == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        if self.semantic_quantizer.codebook.rows != self.semantic_codebook_size {
            return Err(AudioError::Runtime(format!(
                "semantic codebook row mismatch: expected {}, got {}",
                self.semantic_codebook_size, self.semantic_quantizer.codebook.rows
            )));
        }
        if self.semantic_quantizer.out_proj.rows != self.input_dim
            || self.semantic_quantizer.out_proj.cols != codebook_dim
        {
            return Err(AudioError::Runtime("semantic out_proj shape mismatch".to_string()));
        }
        if self.semantic_quantizer.out_bias.len() != self.input_dim {
            return Err(AudioError::Runtime("semantic out_bias shape mismatch".to_string()));
        }

        let residual_quantizers = self.residual_quantizers.len();
        for (index, quantizer) in self.residual_quantizers.iter().enumerate() {
            if quantizer.codebook.rows != self.codebook_size || quantizer.codebook.cols != codebook_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} codebook shape mismatch")));
            }
            if quantizer.out_proj.rows != self.input_dim || quantizer.out_proj.cols != codebook_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} out_proj shape mismatch")));
            }
            if quantizer.out_bias.len() != self.input_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} out_bias shape mismatch")));
            }
        }

        let data_type = self.vocoder_data_type;
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize quantizer decode kernel: {err}")))?;

        let mut semantic_codebook = context.create_array(
            &[self.semantic_codebook_size, codebook_dim],
            data_type,
            "fishaudio_quantizer_semantic_codebook",
        );
        write_f32_slice_to_array(&mut semantic_codebook, &self.semantic_quantizer.codebook.values)?;

        let mut semantic_out_proj =
            context.create_array(&[self.input_dim, codebook_dim], data_type, "fishaudio_quantizer_semantic_out_proj");
        write_f32_slice_to_array(&mut semantic_out_proj, &self.semantic_quantizer.out_proj.values)?;

        let mut semantic_out_bias =
            context.create_array(&[self.input_dim], data_type, "fishaudio_quantizer_semantic_out_bias");
        write_f32_slice_to_array(&mut semantic_out_bias, &self.semantic_quantizer.out_bias)?;

        let residual_count_for_shape = residual_quantizers.max(1);
        let residual_codebook_rows_for_shape = self.codebook_size.max(1);
        let residual_codebook_len =
            checked_product(&[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim])?;
        let residual_proj_len = checked_product(&[residual_count_for_shape, self.input_dim, codebook_dim])?;
        let residual_bias_len = checked_product(&[residual_count_for_shape, self.input_dim])?;
        let mut residual_codebook_host = vec![0.0_f32; residual_codebook_len];
        let mut residual_proj_host = vec![0.0_f32; residual_proj_len];
        let mut residual_bias_host = vec![0.0_f32; residual_bias_len];
        for (index, quantizer) in self.residual_quantizers.iter().enumerate() {
            let codebook_offset = index
                .checked_mul(self.codebook_size)
                .and_then(|value| value.checked_mul(codebook_dim))
                .ok_or(AudioError::Runtime("residual codebook offset overflow".to_string()))?;
            let codebook_end = codebook_offset
                .checked_add(checked_product(&[self.codebook_size, codebook_dim])?)
                .ok_or(AudioError::Runtime("residual codebook offset overflow".to_string()))?;
            residual_codebook_host[codebook_offset..codebook_end].copy_from_slice(&quantizer.codebook.values);

            let proj_offset = index
                .checked_mul(self.input_dim)
                .and_then(|value| value.checked_mul(codebook_dim))
                .ok_or(AudioError::Runtime("residual proj offset overflow".to_string()))?;
            let proj_end = proj_offset
                .checked_add(checked_product(&[self.input_dim, codebook_dim])?)
                .ok_or(AudioError::Runtime("residual proj offset overflow".to_string()))?;
            residual_proj_host[proj_offset..proj_end].copy_from_slice(&quantizer.out_proj.values);

            let bias_offset = index
                .checked_mul(self.input_dim)
                .ok_or(AudioError::Runtime("residual bias offset overflow".to_string()))?;
            let bias_end = bias_offset
                .checked_add(self.input_dim)
                .ok_or(AudioError::Runtime("residual bias offset overflow".to_string()))?;
            residual_bias_host[bias_offset..bias_end].copy_from_slice(&quantizer.out_bias);
        }

        let mut residual_codebooks = context.create_array(
            &[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim],
            data_type,
            "fishaudio_quantizer_residual_codebooks",
        );
        write_f32_slice_to_array(&mut residual_codebooks, &residual_codebook_host)?;

        let mut residual_out_proj = context.create_array(
            &[residual_count_for_shape, self.input_dim, codebook_dim],
            data_type,
            "fishaudio_quantizer_residual_out_proj",
        );
        write_f32_slice_to_array(&mut residual_out_proj, &residual_proj_host)?;

        let mut residual_out_bias = context.create_array(
            &[residual_count_for_shape, self.input_dim],
            data_type,
            "fishaudio_quantizer_residual_out_bias",
        );
        write_f32_slice_to_array(&mut residual_out_bias, &residual_bias_host)?;

        Ok(FishAudioQuantizerGpuResources {
            data_type,
            codebook_dim,
            residual_quantizers,
            semantic_cardinality: self.semantic_codebook_size,
            residual_cardinality: self.codebook_size,
            kernel,
            semantic_codebook,
            semantic_out_proj,
            semantic_out_bias,
            residual_codebooks,
            residual_out_proj,
            residual_out_bias,
        })
    }

    fn quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<FishAudioQuantizerGpuResources>> {
        let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_QUANTIZER_RESOURCES_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let resources = Rc::new(self.build_quantizer_gpu_resources(context)?);
            cache.borrow_mut().insert(key, resources.clone());
            Ok(resources)
        })
    }

    fn build_vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<FishAudioDecoderGpuGraph> {
        let data_type = self.vocoder_data_type;
        let first_conv =
            create_conv1d_gpu_layer(context, data_type, &self.decoder.first_conv, "fishaudio_decoder_first_conv")?;
        let final_snake_alpha = create_alpha_gpu_array(
            context,
            data_type,
            self.decoder.final_conv.cin,
            &self.decoder.final_snake_alpha,
            "fishaudio_decoder_final_snake_alpha",
        )?;
        let final_conv =
            create_conv1d_gpu_layer(context, data_type, &self.decoder.final_conv, "fishaudio_decoder_final_conv")?;

        let mut upsample_blocks = Vec::with_capacity(self.decoder.upsample_blocks.len());
        for (index, (trans_conv, convnext)) in self.decoder.upsample_blocks.iter().enumerate() {
            let trans_conv = create_conv_transpose1d_gpu_layer(
                context,
                data_type,
                trans_conv,
                &format!("fishaudio_upsample_{index}_trans_conv"),
            )?;
            let convnext = create_convnext_gpu_layer(
                context,
                data_type,
                convnext,
                &format!("fishaudio_upsample_{index}_convnext"),
            )?;
            upsample_blocks.push((trans_conv, convnext));
        }

        let mut decoder_blocks = Vec::with_capacity(self.decoder.decoder_blocks.len());
        for (index, block) in self.decoder.decoder_blocks.iter().enumerate() {
            let channels = block.trans_conv.cout;
            let snake_alpha = create_alpha_gpu_array(
                context,
                data_type,
                block.trans_conv.cin,
                &block.snake_alpha,
                &format!("fishaudio_decoder_block_{index}_snake_alpha"),
            )?;
            let trans_conv = create_conv_transpose1d_gpu_layer(
                context,
                data_type,
                &block.trans_conv,
                &format!("fishaudio_decoder_block_{index}_trans_conv"),
            )?;
            let res_unit1 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit1,
                channels,
                &format!("fishaudio_decoder_block_{index}_res1"),
            )?;
            let res_unit2 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit2,
                channels,
                &format!("fishaudio_decoder_block_{index}_res2"),
            )?;
            let res_unit3 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit3,
                channels,
                &format!("fishaudio_decoder_block_{index}_res3"),
            )?;
            decoder_blocks.push(FishAudioDecoderBlockGpuLayer {
                snake_alpha,
                trans_conv,
                res_unit1,
                res_unit2,
                res_unit3,
            });
        }

        Ok(FishAudioDecoderGpuGraph {
            first_conv,
            upsample_blocks,
            decoder_blocks,
            final_snake_alpha,
            final_conv,
        })
    }

    fn vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<FishAudioDecoderGpuGraph>> {
        let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_VOCODER_GRAPH_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let graph = Rc::new(self.build_vocoder_gpu_graph(context)?);
            cache.borrow_mut().insert(key, graph.clone());
            Ok(graph)
        })
    }

    fn encode_post_module_layers(
        runtime: &FishAudioPostModuleRuntime,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &mut <Metal as Backend>::CommandBuffer,
    ) -> AudioResult<()> {
        let encoding_parameters = EncodingParameters::new();
        for layer in runtime.layers.iter() {
            layer
                .encode(state, &encoding_parameters, command_buffer)
                .map_err(|err| AudioError::Runtime(format!("post_module layer encode failed: {err}")))?;
        }
        runtime
            .output_norm
            .encode(state, &encoding_parameters, command_buffer)
            .map_err(|err| AudioError::Runtime(format!("post_module output norm encode failed: {err}")))?;
        Ok(())
    }

    fn apply_post_module_gpu_on_array_single_batch(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if self.post_module_model_dim != self.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }
        if frames == 0 {
            return Ok(latent_nsc.clone());
        }

        let expected_elements = checked_product(&[frames, self.input_dim])?;
        if latent_nsc.num_elements() != expected_elements {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_elements,
                actual_tokens: latent_nsc.num_elements(),
            });
        }

        let runtime = self.post_module_runtime_on_context(context, frames.max(1))?;
        let token_ids = vec![0_u64; frames];
        let token_positions = (0..frames).collect::<Vec<_>>();
        let mut state = ForwardPassState::new_classifier(
            runtime.context.clone(),
            &runtime.model_shape,
            &runtime.scratch_buffers,
            runtime.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            false,
            1,
        );

        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main_output = {
            let main_ref = main.borrow();
            if main_ref.shape() != [frames, self.input_dim] {
                return Err(AudioError::Runtime(format!(
                    "post_module main shape mismatch: expected [{frames}, {}], got {:?}",
                    self.input_dim,
                    main_ref.shape()
                )));
            }
            if main_ref.data_type() != latent_nsc.data_type() {
                return Err(AudioError::Runtime(format!(
                    "post_module dtype mismatch: main={:?}, latent={:?}",
                    main_ref.data_type(),
                    latent_nsc.data_type()
                )));
            }
            main_ref.clone()
        };

        let copy_bytes = latent_nsc
            .num_elements()
            .checked_mul(latent_nsc.data_type().size_in_bytes())
            .ok_or(AudioError::Runtime("post_module copy size overflow".to_string()))?;
        let encode_start = profile.is_some().then(Instant::now);
        let mut command_buffer = runtime
            .context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create post_module command buffer: {err}")))?;
        command_buffer.with_copy_encoder(|copy_encoder| {
            let latent_buffer = latent_nsc.buffer();
            let main_output_buffer = main_output.buffer();
            let latent_buffer = latent_buffer.borrow();
            let main_output_buffer = main_output_buffer.borrow();
            copy_encoder.encode_copy(&latent_buffer, &main_output_buffer, copy_bytes);
        });
        Self::encode_post_module_layers(&runtime, &mut state, &mut command_buffer)?;
        let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        command_buffer.submit();
        let wait_start = profile.is_some().then(Instant::now);
        command_buffer.wait_until_completed().map_err(|err| {
            AudioError::Runtime(format!("failed to wait for post_module command buffer: {err}"))
        })?;
        let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        push_audio_command_buffer_profile(profile, "post_module", &command_buffer, cpu_encode_ms, cpu_wait_ms, None);

        Ok(main_output)
    }

    fn apply_post_module_gpu_on_array(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if self.post_module_model_dim != self.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }
        let expected_elements = checked_product(&[batch_size, frames, self.input_dim])?;
        if latent_nsc.num_elements() != expected_elements {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_elements,
                actual_tokens: latent_nsc.num_elements(),
            });
        }
        if batch_size == 1 && lengths.first().copied() == Some(frames) {
            return self.apply_post_module_gpu_on_array_single_batch(context, latent_nsc, frames, profile);
        }

        let output = context.create_array(
            &[batch_size, frames, self.input_dim],
            latent_nsc.data_type(),
            "fishaudio_post_module_output_nsc",
        );
        let mut batch_indices_by_length = BTreeMap::<usize, Vec<usize>>::new();
        for (batch_index, &active_len) in lengths.iter().enumerate() {
            if active_len == 0 {
                continue;
            }
            if active_len > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_len,
                    frames,
                });
            }
            batch_indices_by_length.entry(active_len).or_default().push(batch_index);
        }

        if batch_indices_by_length.is_empty() {
            let mut output = output;
            output.copy_from_array(latent_nsc);
            return Ok(output);
        }

        let full_copy_bytes = latent_nsc.size();
        let mut copied_output_prefix = false;
        for (active_len, batch_indices) in batch_indices_by_length {
            let runtime = self.post_module_runtime_on_context(context, active_len.max(1))?;
            let token_ids = vec![0_u64; active_len];
            let token_positions = (0..active_len).collect::<Vec<_>>();
            let mut state = ForwardPassState::new_classifier(
                runtime.context.clone(),
                &runtime.model_shape,
                &runtime.scratch_buffers,
                runtime.shared_buffers.clone(),
                &token_ids,
                &token_positions,
                false,
                1,
            );

            let main = state.arrays(&[ArrayId::Main])[0].clone();
            let main_output = {
                let main_ref = main.borrow();
                if main_ref.shape() != [active_len, self.input_dim] {
                    return Err(AudioError::Runtime(format!(
                        "post_module main shape mismatch: expected [{active_len}, {}], got {:?}",
                        self.input_dim,
                        main_ref.shape()
                    )));
                }
                if main_ref.data_type() != latent_nsc.data_type() {
                    return Err(AudioError::Runtime(format!(
                        "post_module dtype mismatch: main={:?}, latent={:?}",
                        main_ref.data_type(),
                        latent_nsc.data_type()
                    )));
                }
                main_ref.clone()
            };

            for &batch_index in &batch_indices {
                let encode_start = profile.is_some().then(Instant::now);
                let mut command_buffer = runtime.context.create_command_buffer().map_err(|err| {
                    AudioError::Runtime(format!("failed to create post_module command buffer: {err}"))
                })?;
                if !copied_output_prefix {
                    command_buffer.with_copy_encoder(|copy_encoder| {
                        let latent_buffer = latent_nsc.buffer();
                        let output_buffer = output.buffer();
                        let latent_buffer = latent_buffer.borrow();
                        let output_buffer = output_buffer.borrow();
                        copy_encoder.encode_copy_ranges(
                            (&latent_buffer, latent_nsc.offset()),
                            (&output_buffer, output.offset()),
                            full_copy_bytes,
                        );
                    });
                    copied_output_prefix = true;
                }
                let source = array_batch_view(latent_nsc, batch_index, frames, self.input_dim, active_len)?;
                command_buffer.with_copy_encoder(|copy_encoder| {
                    let source_buffer = source.buffer();
                    let main_output_buffer = main_output.buffer();
                    let source_buffer = source_buffer.borrow();
                    let main_output_buffer = main_output_buffer.borrow();
                    copy_encoder.encode_copy_ranges(
                        (&source_buffer, source.offset()),
                        (&main_output_buffer, main_output.offset()),
                        source.size(),
                    );
                });
                Self::encode_post_module_layers(&runtime, &mut state, &mut command_buffer)?;
                let destination = array_batch_view(&output, batch_index, frames, self.input_dim, active_len)?;
                command_buffer.with_copy_encoder(|copy_encoder| {
                    let main_output_buffer = main_output.buffer();
                    let destination_buffer = destination.buffer();
                    let main_output_buffer = main_output_buffer.borrow();
                    let destination_buffer = destination_buffer.borrow();
                    copy_encoder.encode_copy_ranges(
                        (&main_output_buffer, main_output.offset()),
                        (&destination_buffer, destination.offset()),
                        destination.size(),
                    );
                });
                let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                command_buffer.submit();
                let wait_start = profile.is_some().then(Instant::now);
                command_buffer.wait_until_completed().map_err(|err| {
                    AudioError::Runtime(format!("failed to wait for post_module command buffer: {err}"))
                })?;
                let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                let label = if batch_size == 1 && active_len == frames {
                    "post_module".to_string()
                } else {
                    format!("post_module_len_{active_len}_batch_{batch_index}")
                };
                push_audio_command_buffer_profile(profile, label, &command_buffer, cpu_encode_ms, cpu_wait_ms, None);
            }
        }

        Ok(output)
    }

    fn run_residual_unit_enqueued(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        command_buffer: &mut <Metal as Backend>::CommandBuffer,
        input: &Array<Metal>,
        unit: &FishAudioResidualUnitGpuLayer,
        lengths: &[i32],
        lengths_array: &Array<Metal>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Array<Metal>> {
        let residual = input.clone();
        let x = snake1d_enqueue(context, command_buffer, input, &unit.snake1_alpha, batch_size, channels, seq_len)?;
        let x = causal_conv1d_grouped_enqueue(
            context,
            command_buffer,
            &x,
            &unit.conv1,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = snake1d_enqueue(context, command_buffer, &x, &unit.snake2_alpha, batch_size, channels, seq_len)?;
        causal_conv1d_grouped_residual_enqueue(
            context,
            command_buffer,
            &x,
            &residual,
            &unit.conv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )
    }

    fn submit_decode_padded(
        &self,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<SubmittedDecodedPaddedAudio> {
        let collect_command_buffer_profile =
            runtime_options.collect_command_buffer_profile || runtime_options.capture_single_decode;
        if batch_size == 0 || frames == 0 {
            let out_lengths = lengths
                .iter()
                .map(|&length| {
                    length
                        .checked_mul(self.upsample_factor)
                        .ok_or(AudioError::Runtime("FishAudio length scaling overflow".to_string()))
                })
                .collect::<AudioResult<Vec<_>>>()?;
            let context = <Metal as Backend>::Context::new()
                .map_err(|err| AudioError::Runtime(format!("failed to create metal audio context: {err}")))?;
            return Ok(SubmittedDecodedPaddedAudio {
                output: context.create_array(&[0], DataType::F32, "fishaudio_empty_decode_output"),
                channels: 1,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
                final_command_buffer: None,
                final_command_label: None,
                final_cpu_encode_ms: 0.0,
                submitted_command_buffers: Vec::new(),
                decode_profile: None,
                capture: None,
            });
        }

        let mut lengths_i32 = lengths
            .iter()
            .map(|&length| {
                i32::try_from(length).map_err(|_| AudioError::Runtime("FishAudio length exceeds i32 range".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let mut decode_profile = collect_command_buffer_profile.then(|| AudioDecodeProfile {
            batch_size,
            frames,
            codebooks,
            ..AudioDecodeProfile::default()
        });
        let capture = if runtime_options.capture_single_decode {
            Some(AudioCaptureGuard::start()?)
        } else {
            None
        };
        let context = if let Some(capture) = capture.as_ref() {
            capture.context()
        } else {
            self.decode_context()?
        };

        let can_use_gpu_latent_path = batch_size == 1 && lengths.first().copied() == Some(frames);
        let mut x;
        let mut x_layout = SequenceLayout::Nsc;
        let quantized_nsc = self.decode_quantizer_to_nsc_array_on_context(
            &context,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
            &mut decode_profile,
        )?;
        x = if can_use_gpu_latent_path {
            self.apply_post_module_gpu_on_array_single_batch(&context, &quantized_nsc, frames, &mut decode_profile)?
        } else {
            self.apply_post_module_gpu_on_array(
                &context,
                &quantized_nsc,
                lengths,
                batch_size,
                frames,
                &mut decode_profile,
            )?
        };

        let vocoder_graph = self.vocoder_gpu_graph(&context)?;
        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create FishAudio decode command buffer: {err}")))?;
        let profile_decoder_micro_stages = runtime_options.profile_decoder_micro_stages;
        let chunked_command_buffers = runtime_options.chunked_command_buffers;
        let micro_flush_min_elements = runtime_options.micro_flush_min_elements;
        let mut submitted_command_buffers = Vec::<PendingAudioCommandBufferProfile>::new();
        let mut command_buffer_encode_start = decode_profile.is_some().then(Instant::now);
        let mut flush_stage = |label: String,
                               estimated_macs: Option<usize>,
                               command_buffer: &mut MetalCommandBuffer|
         -> AudioResult<()> {
            if !(chunked_command_buffers || profile_decoder_micro_stages) {
                return Ok(());
            }
            let cpu_encode_ms =
                command_buffer_encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
            command_buffer.submit();
            if decode_profile.is_some() {
                submitted_command_buffers.push(PendingAudioCommandBufferProfile {
                    label,
                    cpu_encode_ms,
                    cpu_wait_ms: 0.0,
                    command_buffer: command_buffer.clone(),
                    estimated_macs,
                });
            }
            *command_buffer = context.create_command_buffer().map_err(|err| {
                AudioError::Runtime(format!("failed to create FishAudio decode command buffer: {err}"))
            })?;
            command_buffer_encode_start = decode_profile.is_some().then(Instant::now);
            Ok(())
        };
        let mut current_channels = self.input_dim;
        let mut current_frames = frames;
        let mut next_lengths_i32 = vec![0_i32; lengths_i32.len()];
        let mut lengths_array = context.create_array(&[lengths_i32.len()], DataType::I32, "fishaudio_lengths_a");
        write_i32_slice_to_array(&mut lengths_array, &lengths_i32, "fishaudio_lengths_a")?;
        let mut next_lengths_array = context.create_array(&[lengths_i32.len()], DataType::I32, "fishaudio_lengths_b");

        for (block_index, (trans_conv, convnext)) in vocoder_graph.upsample_blocks.iter().enumerate() {
            if trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio upsampler input channel mismatch: expected {}, got {}",
                    trans_conv.cin, current_channels
                )));
            }
            let next_frames = current_frames
                .checked_mul(trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio upsampler frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, trans_conv.stride)?;
            write_i32_slice_to_array(&mut next_lengths_array, &next_lengths_i32, "fishaudio_upsample_lengths")?;

            x = causal_conv_transpose1d_causal_pad_enqueue(
                &context,
                &mut command_buffer,
                &x,
                trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                x_layout,
                &next_lengths_array,
            )
            .map_err(|err| {
                AudioError::Runtime(format!(
                    "FishAudio upsample block {block_index} transpose_conv failed: {err} (x_len={}, batch_size={}, cin={}, seq_len_in={}, seq_len_out={})",
                    x.num_elements(),
                    batch_size,
                    trans_conv.cin,
                    current_frames,
                    next_frames
                ))
            })?;
            x = self
                .apply_convnext_ncs_enqueued(
                    &context,
                    &mut command_buffer,
                    &x,
                    convnext,
                    &next_lengths_i32,
                    &next_lengths_array,
                    batch_size,
                    trans_conv.cout,
                    next_frames,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("FishAudio upsample block {block_index} convnext failed: {err}"))
                })?;
            flush_stage(format!("upsample_block_{block_index}"), None, &mut command_buffer)?;

            x_layout = SequenceLayout::Ncs;
            current_frames = next_frames;
            current_channels = trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);
        }

        if x_layout == SequenceLayout::Nsc {
            x = transpose_nsc_to_ncs_enqueue(
                &context,
                &mut command_buffer,
                &x,
                batch_size,
                current_frames,
                current_channels,
            )?;
            x_layout = SequenceLayout::Ncs;
            flush_stage("upsample_to_decoder_layout".to_string(), None, &mut command_buffer)?;
        }
        debug_assert_eq!(x_layout, SequenceLayout::Ncs);

        if vocoder_graph.first_conv.cin != current_channels {
            return Err(AudioError::Runtime(format!(
                "FishAudio decoder input channels mismatch: expected {}, got {}",
                vocoder_graph.first_conv.cin, current_channels
            )));
        }
        x = causal_conv1d_grouped_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.first_conv,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
        )?;
        current_channels = vocoder_graph.first_conv.cout;
        flush_stage(
            "decoder_first_conv".to_string(),
            Some(conv1d_estimated_macs(batch_size, current_frames, &vocoder_graph.first_conv)?),
            &mut command_buffer,
        )?;

        for (block_index, block) in vocoder_graph.decoder_blocks.iter().enumerate() {
            if block.trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio decoder block input mismatch: expected {}, got {}",
                    block.trans_conv.cin, current_channels
                )));
            }
            x = snake1d_enqueue(
                &context,
                &mut command_buffer,
                &x,
                &block.snake_alpha,
                batch_size,
                current_channels,
                current_frames,
            )?;

            let next_frames = current_frames
                .checked_mul(block.trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio decoder frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, block.trans_conv.stride)?;
            write_i32_slice_to_array(&mut next_lengths_array, &next_lengths_i32, "fishaudio_decoder_block_lengths")?;

            x = causal_conv_transpose1d_causal_pad_enqueue(
                &context,
                &mut command_buffer,
                &x,
                &block.trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                SequenceLayout::Ncs,
                &next_lengths_array,
            )?;
            let trans_conv_estimated_macs =
                convtranspose_estimated_macs(batch_size, current_frames, &block.trans_conv)?;

            current_frames = next_frames;
            current_channels = block.trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);
            let active_elements = batch_size
                .checked_mul(current_channels)
                .and_then(|value| value.checked_mul(current_frames))
                .ok_or(AudioError::Runtime("FishAudio decoder element count overflow".to_string()))?;
            let res1_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit1)?;
            let res2_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit2)?;
            let res3_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit3)?;
            let block_total_estimated_macs = checked_add_usize(
                checked_add_usize(
                    checked_add_usize(trans_conv_estimated_macs, res1_estimated_macs, "decoder block estimated MACs")?,
                    res2_estimated_macs,
                    "decoder block estimated MACs",
                )?,
                res3_estimated_macs,
                "decoder block estimated MACs",
            )?;

            if profile_decoder_micro_stages {
                flush_stage(
                    format!("decoder_block_{block_index}_trans_conv"),
                    Some(trans_conv_estimated_macs),
                    &mut command_buffer,
                )?;
            }

            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit1,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            if profile_decoder_micro_stages {
                flush_stage(
                    format!("decoder_block_{block_index}_res1"),
                    Some(res1_estimated_macs),
                    &mut command_buffer,
                )?;
            } else if chunked_command_buffers && active_elements >= micro_flush_min_elements {
                flush_stage(
                    format!("decoder_block_{block_index}_res1"),
                    Some(checked_add_usize(
                        trans_conv_estimated_macs,
                        res1_estimated_macs,
                        "decoder block res1 estimated MACs",
                    )?),
                    &mut command_buffer,
                )?;
            }
            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit2,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            if profile_decoder_micro_stages || (chunked_command_buffers && active_elements >= micro_flush_min_elements)
            {
                flush_stage(
                    format!("decoder_block_{block_index}_res2"),
                    Some(res2_estimated_macs),
                    &mut command_buffer,
                )?;
            }
            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit3,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            flush_stage(
                if profile_decoder_micro_stages {
                    format!("decoder_block_{block_index}_res3")
                } else {
                    format!("decoder_block_{block_index}")
                },
                Some(if profile_decoder_micro_stages {
                    res3_estimated_macs
                } else {
                    block_total_estimated_macs
                }),
                &mut command_buffer,
            )?;
        }

        x = snake1d_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.final_snake_alpha,
            batch_size,
            current_channels,
            current_frames,
        )?;
        x = causal_conv1d_grouped_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.final_conv,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
        )?;
        x = tanh_enqueue(&context, &mut command_buffer, &x)?;

        let final_cpu_encode_ms =
            command_buffer_encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        let final_command_label = Some("decoder_final".to_string());
        command_buffer.submit();
        let final_command_buffer = Some(command_buffer.clone());
        let out_lengths = lengths_i32
            .into_iter()
            .map(|length| {
                usize::try_from(length)
                    .map_err(|_| AudioError::Runtime("FishAudio decoder produced invalid negative length".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        Ok(SubmittedDecodedPaddedAudio {
            output: x,
            channels: vocoder_graph.final_conv.cout,
            frames: current_frames,
            lengths: out_lengths,
            final_command_buffer,
            final_command_label,
            final_cpu_encode_ms,
            submitted_command_buffers,
            decode_profile,
            capture,
        })
    }

    fn decode_padded(
        &self,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<(super::decoder::DecodedPaddedAudio, Option<AudioDecodeProfile>)> {
        self.submit_decode_padded(runtime_options, tokens, lengths, batch_size, codebooks, frames)?.resolve()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NanoCodecFsqRuntimeConfig {
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Box<[i32]>,
    dim_base_index: Box<[i32]>,
    codebook_dim_per_group: usize,
    channels: usize,
    codec_cardinality: u32,
    eps: f32,
    output_packing: AudioTokenPacking,
    decoder: Option<NanoCodecDecoderGraph>,
    fishaudio_decoder: Option<FishAudioCodecGraph>,
}

impl NanoCodecFsqRuntimeConfig {
    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        let parsed = parse_runtime_config_json(tts_config)?;
        Self::from_runtime_config_json(parsed)
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        if tts_config.get("audio_codec").is_some() {
            return Self::from_tts_config_value(tts_config);
        }

        if tts_config.get("audio_decoder_config").is_some() {
            if let Some(decoder_type) = tts_config
                .get("audio_decoder_config")
                .and_then(|decoder| decoder.get("type"))
                .and_then(serde_json::Value::as_str)
            {
                if decoder_type == "DescriptAudioCodecConfig" {
                    let fishaudio_weights = model_path.join("model.safetensors");
                    if !fishaudio_weights.is_file() {
                        return Err(AudioError::Runtime(format!(
                            "missing exported FishAudio decoder weights '{}'",
                            fishaudio_weights.display()
                        )));
                    }
                    let parsed_fishaudio = parse_fishaudio_tts_config_json(tts_config)?;
                    let cfg = &parsed_fishaudio.audio_decoder_config;
                    let total_codebooks = cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime(
                        "FishAudio codebook count overflow while building runtime config".to_string(),
                    ))?;
                    let codebook_size_i32 = i32::try_from(cfg.codebook_size).map_err(|_| {
                        AudioError::Runtime("FishAudio codebook_size exceeds i32 kernel range".to_string())
                    })?;
                    if codebook_size_i32 <= 1 {
                        return Err(AudioError::InvalidTokenCardinality);
                    }

                    let parsed = RuntimeConfigJson {
                        r#type: Some("nanocodec_fsq".to_string()),
                        sample_rate: cfg.samplerate,
                        num_groups: total_codebooks,
                        num_levels_per_group: vec![codebook_size_i32],
                        eps: default_eps(),
                        output_packing: RuntimePacking::CodebookMajor,
                        decoder: None,
                    };
                    let mut runtime = Self::from_runtime_config_json(parsed)?;
                    runtime.fishaudio_decoder =
                        Some(build_fishaudio_codec_graph(&parsed_fishaudio, &fishaudio_weights)?);
                    return Ok(runtime);
                }
            }
            let parsed_tts = parse_lalamo_tts_config_json(tts_config)?;
            let cfg = &parsed_tts.audio_decoder_config;
            let decoder_json =
                build_decoder_json_from_lalamo_export(&parsed_tts, &model_path.join("model.safetensors"))?;

            let parsed = RuntimeConfigJson {
                r#type: Some("nanocodec_fsq".to_string()),
                sample_rate: cfg.samplerate,
                num_groups: cfg.quantizer_config.num_groups,
                num_levels_per_group: cfg.quantizer_config.quantizer_config.num_levels.clone(),
                eps: cfg.quantizer_config.quantizer_config.eps,
                output_packing: RuntimePacking::CodebookMajor,
                decoder: Some(decoder_json),
            };

            return Self::from_runtime_config_json(parsed);
        }

        Self::from_tts_config_value(tts_config)
    }

    fn from_runtime_config_json(parsed: RuntimeConfigJson) -> AudioResult<Self> {
        let mut config = Self::new(
            parsed.sample_rate,
            parsed.num_groups,
            parsed.num_levels_per_group.into_boxed_slice(),
            parsed.eps,
            parsed.output_packing.into(),
        )?;
        if let Some(decoder_json) = parsed.decoder {
            config.decoder = Some(NanoCodecDecoderGraph::try_from(decoder_json)?);
        }
        Ok(config)
    }

    pub fn new(
        sample_rate: u32,
        num_groups: usize,
        num_levels_per_group: Box<[i32]>,
        eps: f32,
        output_packing: AudioTokenPacking,
    ) -> AudioResult<Self> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidSampleRate);
        }
        if num_groups == 0 || num_levels_per_group.is_empty() {
            return Err(AudioError::InvalidTokenCardinality);
        }
        if !eps.is_finite() || !(0.0..1.0).contains(&eps) {
            return Err(AudioError::Runtime("eps must be finite and satisfy 0.0 <= eps < 1.0".to_string()));
        }
        for &level in num_levels_per_group.iter() {
            if level <= 1 {
                return Err(AudioError::InvalidTokenCardinality);
            }
        }

        let codebook_dim_per_group = num_levels_per_group.len();
        let channels = num_groups
            .checked_mul(codebook_dim_per_group)
            .ok_or(AudioError::Runtime("num_groups * codebook_dim_per_group overflow".to_string()))?;
        let dim_base_index = compute_dim_base_index(&num_levels_per_group)?;

        let mut codec_cardinality_u64 = 1_u64;
        for &level in num_levels_per_group.iter() {
            codec_cardinality_u64 = codec_cardinality_u64
                .checked_mul(level as u64)
                .ok_or(AudioError::Runtime("codec cardinality overflow".to_string()))?;
        }

        if codec_cardinality_u64 > u32::MAX as u64 {
            return Err(AudioError::Runtime("codec cardinality exceeds u32 range".to_string()));
        }

        let codec_cardinality = codec_cardinality_u64 as u32;
        if codec_cardinality_u64 > (i32::MAX as u64 + 1) {
            return Err(AudioError::Runtime("codec cardinality exceeds i32 token kernel range".to_string()));
        }

        Ok(Self {
            sample_rate,
            num_groups,
            num_levels_per_group,
            dim_base_index: dim_base_index.into_boxed_slice(),
            codebook_dim_per_group,
            channels,
            codec_cardinality,
            eps,
            output_packing,
            decoder: None,
            fishaudio_decoder: None,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_levels_per_group(&self) -> &[i32] {
        &self.num_levels_per_group
    }

    pub fn dim_base_index(&self) -> &[i32] {
        &self.dim_base_index
    }

    pub fn codebook_dim_per_group(&self) -> usize {
        self.codebook_dim_per_group
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn codec_cardinality(&self) -> u32 {
        self.codec_cardinality
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn output_packing(&self) -> AudioTokenPacking {
        self.output_packing
    }

    pub fn decoder(&self) -> Option<&NanoCodecDecoderGraph> {
        self.decoder.as_ref()
    }

    fn fishaudio_decoder(&self) -> Option<&FishAudioCodecGraph> {
        self.fishaudio_decoder.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct NanoCodecFsqRuntime {
    config: NanoCodecFsqRuntimeConfig,
    options: NanoCodecFsqRuntimeOptions,
    last_decode_profile: Arc<RwLock<Option<AudioDecodeProfile>>>,
}

impl NanoCodecFsqRuntime {
    pub fn new(config: NanoCodecFsqRuntimeConfig) -> Self {
        Self::new_with_options(config, NanoCodecFsqRuntimeOptions::default())
    }

    pub fn new_with_options(
        config: NanoCodecFsqRuntimeConfig,
        options: NanoCodecFsqRuntimeOptions,
    ) -> Self {
        if options.capture_single_decode {
            <Metal as Backend>::Context::enable_capture();
        }
        Self {
            config,
            options,
            last_decode_profile: Arc::new(RwLock::new(None)),
        }
    }

    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value(tts_config)?))
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(tts_config, model_path)?))
    }

    pub fn from_tts_config_value_and_model_path_with_options(
        tts_config: &serde_json::Value,
        model_path: &Path,
        options: NanoCodecFsqRuntimeOptions,
    ) -> AudioResult<Self> {
        Ok(Self::new_with_options(
            NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(tts_config, model_path)?,
            options,
        ))
    }

    pub fn config(&self) -> &NanoCodecFsqRuntimeConfig {
        &self.config
    }

    pub fn options(&self) -> NanoCodecFsqRuntimeOptions {
        self.options
    }

    pub fn last_decode_profile(&self) -> Option<AudioDecodeProfile> {
        self.last_decode_profile.read().ok().and_then(|profile| profile.as_ref().cloned())
    }

    fn validate_fishaudio_token_delta(
        &self,
        tokens: &AudioTokenGrid,
        fishaudio: &FishAudioCodecGraph,
    ) -> AudioResult<()> {
        let semantic_cardinality = u32::try_from(fishaudio.semantic_codebook_size).map_err(|_| {
            AudioError::Runtime("FishAudio semantic codebook cardinality exceeds u32 range".to_string())
        })?;
        let residual_cardinality = u32::try_from(fishaudio.codebook_size).map_err(|_| {
            AudioError::Runtime("FishAudio residual codebook cardinality exceeds u32 range".to_string())
        })?;
        let codebook_major = tokens.to_packing(AudioTokenPacking::CodebookMajor);
        let frames = codebook_major.frames();
        for batch in 0..codebook_major.batch_size() {
            for codebook in 0..codebook_major.codebooks() {
                let cardinality = if codebook == 0 {
                    semantic_cardinality
                } else {
                    residual_cardinality
                };
                for frame in 0..frames {
                    let token = codebook_major.get(batch, codebook, frame);
                    if token >= cardinality {
                        return Err(AudioError::InvalidCodecToken {
                            token,
                            cardinality,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn decode_fishaudio_stream_delta(
        &self,
        state: &mut AudioDecodeStreamState,
        fishaudio: &FishAudioCodecGraph,
        input_frames: usize,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if state.total_frames() == 0 {
            state.record_last_step_stats(input_frames, 0, 0);
            return Ok(super::decoder::DecodedPaddedAudio {
                samples: Vec::new(),
                channels: 1,
                frames: 0,
                lengths: vec![0usize; state.batch_size],
            });
        }

        let Some(context_frames) = fishaudio.streaming_decode_context_frames()? else {
            let full_grid = state.to_full_grid()?;
            let full_pcm = self.decode(&full_grid)?;
            state.record_last_step_stats(input_frames, 0, state.total_frames());
            return state.extract_delta_padded(&full_pcm);
        };
        let mut window_start = state.total_frames();
        for &emitted in &state.emitted_semantic_lengths {
            window_start = window_start.min(emitted.saturating_sub(context_frames));
        }
        let window_end = state.total_frames();
        let batch_size = state.batch_size;
        let codebooks = state.codebooks;
        let (window_tokens, window_lengths, window_frames) = state.flatten_window(window_start, window_end)?;
        let (decoded_window, decode_profile) = fishaudio.decode_padded(
            self.options,
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
            *last_decode_profile = decode_profile;
        }
        let audio_offset_frames = window_start
            .checked_mul(fishaudio.upsample_factor)
            .ok_or(AudioError::Runtime("stream audio offset overflow".to_string()))?;
        state.record_last_step_stats(input_frames, window_start, window_frames);
        state.extract_delta_from_padded_with_offset(&decoded_window, audio_offset_frames, fishaudio.upsample_factor)
    }

    pub fn begin_decode_stream(
        &self,
        batch_size: usize,
        codebooks: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        self.begin_decode_stream_with_options(batch_size, codebooks, AudioDecodeStreamingMode::IncrementalStateful, 256)
    }

    pub fn begin_decode_stream_with_options(
        &self,
        batch_size: usize,
        codebooks: usize,
        mode: AudioDecodeStreamingMode,
        max_workspace_frames: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        if codebooks != self.config.num_groups() {
            return Err(AudioError::Runtime(format!(
                "stream codebook mismatch: expected {}, got {}",
                self.config.num_groups(),
                codebooks
            )));
        }
        AudioDecodeStreamState::new(batch_size, codebooks, max_workspace_frames, mode)
    }

    pub fn decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        _is_final: bool,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if new_tokens.codebooks() != state.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                state.codebooks,
                new_tokens.codebooks()
            )));
        }
        if new_tokens.batch_size() != state.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                state.batch_size,
                new_tokens.batch_size()
            )));
        }
        if new_tokens.frames() == 0 {
            state.record_last_step_stats(0, state.total_frames(), 0);
            return Ok(super::decoder::DecodedPaddedAudio {
                samples: Vec::new(),
                channels: if let Some(decoder) = self.config.decoder() {
                    decoder.output_channels()
                } else if self.config.fishaudio_decoder().is_some() {
                    1
                } else {
                    self.config.channels()
                },
                frames: 0,
                lengths: vec![0; state.batch_size],
            });
        }

        state.append_delta(new_tokens)?;
        if let Some(fishaudio) = self.config.fishaudio_decoder() {
            self.validate_fishaudio_token_delta(new_tokens, fishaudio)?;
            return match state.mode {
                AudioDecodeStreamingMode::IncrementalStateful => {
                    self.decode_fishaudio_stream_delta(state, fishaudio, new_tokens.frames())
                },
                AudioDecodeStreamingMode::PrefixFallback => {
                    let full_grid = state.to_full_grid()?;
                    let full_pcm = self.decode(&full_grid)?;
                    state.record_last_step_stats(new_tokens.frames(), 0, state.total_frames());
                    state.extract_delta_padded(&full_pcm)
                },
            };
        }

        let full_grid = state.to_full_grid()?;
        let full_pcm = self.decode(&full_grid)?;
        state.record_last_step_stats(new_tokens.frames(), 0, state.total_frames());
        state.extract_delta_padded(&full_pcm)
    }

    pub(crate) fn submit_decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> AudioResult<Option<PendingStreamPcmChunk>> {
        if is_final || !self.options.async_stream_delivery_enabled() {
            return Ok(None);
        }
        if new_tokens.codebooks() != state.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                state.codebooks,
                new_tokens.codebooks()
            )));
        }
        if new_tokens.batch_size() != state.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                state.batch_size,
                new_tokens.batch_size()
            )));
        }
        if new_tokens.frames() == 0 {
            return Ok(None);
        }

        let Some(fishaudio) = self.config.fishaudio_decoder() else {
            return Ok(None);
        };
        if state.mode != AudioDecodeStreamingMode::IncrementalStateful {
            return Ok(None);
        }
        let Some(context_frames) = fishaudio.streaming_decode_context_frames()? else {
            return Ok(None);
        };

        self.validate_fishaudio_token_delta(new_tokens, fishaudio)?;
        state.append_delta(new_tokens)?;

        let mut window_start = state.total_frames();
        for &emitted in &state.emitted_semantic_lengths {
            window_start = window_start.min(emitted.saturating_sub(context_frames));
        }
        let window_end = state.total_frames();
        let batch_size = state.batch_size;
        let codebooks = state.codebooks;
        let previous_audio_lengths = state.emitted_audio_lengths.clone().into_boxed_slice();
        let semantic_lengths = state.semantic_lengths.clone().into_boxed_slice();
        let (window_tokens, window_lengths, window_frames) = state.flatten_window(window_start, window_end)?;
        let submitted = fishaudio.submit_decode_padded(
            self.options,
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        let audio_offset_frames = window_start
            .checked_mul(fishaudio.upsample_factor)
            .ok_or(AudioError::Runtime("stream audio offset overflow".to_string()))?;
        state.record_last_step_stats(new_tokens.frames(), window_start, window_frames);
        state.mark_submitted_audio_window(&semantic_lengths, fishaudio.upsample_factor)?;

        Ok(Some(PendingStreamPcmChunk {
            runtime: self.clone(),
            submitted,
            previous_audio_lengths,
            semantic_lengths,
            audio_offset_frames,
            upsample_factor: fishaudio.upsample_factor,
            step_stats: state.last_step_stats(),
        }))
    }

    pub fn decoded_padded_to_pcm_batch(
        &self,
        decoded: &super::decoder::DecodedPaddedAudio,
    ) -> AudioResult<AudioPcmBatch> {
        let samples = unpack_padded_to_pcm(
            &decoded.samples,
            decoded.lengths.len(),
            decoded.channels,
            decoded.frames,
            &decoded.lengths,
        )?;
        AudioPcmBatch::new(
            samples.into_boxed_slice(),
            self.config.sample_rate(),
            decoded.channels,
            decoded.lengths.clone().into_boxed_slice(),
        )
    }

    pub fn end_decode_stream(
        &self,
        _state: AudioDecodeStreamState,
    ) -> AudioResult<()> {
        Ok(())
    }

    fn create_context() -> AudioResult<Rc<<Metal as Backend>::Context>> {
        <Metal as Backend>::Context::new()
            .map_err(|err| AudioError::Runtime(format!("failed to create metal audio context: {err}")))
    }
}

impl AudioCodecRuntime for NanoCodecFsqRuntime {
    fn encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> AudioResult<AudioTokenGrid> {
        if self.config.decoder().is_some() || self.config.fishaudio_decoder().is_some() {
            return Err(AudioError::Runtime("encode is not supported when a decoder graph is configured".to_string()));
        }

        if pcm.sample_rate() != self.config.sample_rate() {
            return Err(AudioError::Runtime(format!(
                "pcm sample-rate mismatch: expected {}, got {}",
                self.config.sample_rate(),
                pcm.sample_rate()
            )));
        }

        let (padded_input, lengths_usize, lengths_i32, frames) = pack_pcm_to_padded(pcm, self.config.channels())?;
        let batch_size = pcm.batch_size();

        if batch_size == 0 || frames == 0 {
            let empty_tokens = Vec::<u32>::new().into_boxed_slice();
            let grid = AudioTokenGrid::new(
                empty_tokens,
                batch_size,
                self.config.num_groups(),
                frames,
                lengths_usize.into_boxed_slice(),
                self.config.output_packing(),
            )?;
            return Ok(grid);
        }

        let context = Self::create_context()?;
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqEncodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq encode kernel: {err}")))?;

        let mut input = context.create_array(
            &[batch_size, self.config.channels(), frames],
            DataType::F32,
            "nanocodec_fsq_encode_input",
        );
        input.as_slice_mut::<f32>().copy_from_slice(&padded_input);

        let mut lengths = context.create_array(&[batch_size], DataType::I32, "nanocodec_fsq_encode_lengths");
        lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let tokens = context.create_array(
            &[batch_size, self.config.num_groups(), frames],
            DataType::I32,
            "nanocodec_fsq_encode_tokens",
        );

        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;

        command_buffer.with_compute_encoder(|compute_encoder| {
            kernel.encode(
                input.buffer(),
                tokens.buffer(),
                lengths.buffer(),
                num_groups_i32,
                frames_i32,
                codebook_dim_i32,
                self.config.num_levels_per_group(),
                self.config.dim_base_index(),
                self.config.eps(),
                batch_size_i32,
                compute_encoder,
            );
        });

        command_buffer.submit();
        command_buffer.wait_until_completed().map_err(|err| {
            AudioError::Runtime(format!("failed to wait for FSQ encode command buffer: {err}"))
        })?;

        let mut tokens_u32 = vec![0_u32; tokens.num_elements()];
        for (index, &token) in tokens.as_slice::<i32>().iter().enumerate() {
            if token < 0 {
                return Err(AudioError::Runtime(format!(
                    "fsq encode returned negative token at index {index}: {token}"
                )));
            }

            let token_u32 = token as u32;
            if token_u32 >= self.config.codec_cardinality() {
                return Err(AudioError::InvalidCodecToken {
                    token: token_u32,
                    cardinality: self.config.codec_cardinality(),
                });
            }
            tokens_u32[index] = token_u32;
        }

        let codebook_major = AudioTokenGrid::new(
            tokens_u32.into_boxed_slice(),
            batch_size,
            self.config.num_groups(),
            frames,
            lengths_usize.into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )?;

        Ok(codebook_major.to_packing(self.config.output_packing()))
    }

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<AudioPcmBatch> {
        if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
            *last_decode_profile = None;
        }
        if tokens.codebooks() != self.config.num_groups() {
            return Err(AudioError::Runtime(format!(
                "token codebook mismatch: expected {}, got {}",
                self.config.num_groups(),
                tokens.codebooks()
            )));
        }

        let batch_size = tokens.batch_size();
        let frames = tokens.frames();
        let lengths_usize = tokens.lengths().to_vec();
        let lengths_i32 = convert_lengths_to_i32(&lengths_usize, frames)?;

        if batch_size == 0 || frames == 0 {
            let channels = if let Some(decoder) = self.config.decoder() {
                decoder.output_channels()
            } else if self.config.fishaudio_decoder().is_some() {
                1
            } else {
                self.config.channels()
            };
            let out_lengths = if let Some(decoder) = self.config.decoder() {
                lengths_usize
                    .iter()
                    .map(|&length| {
                        length
                            .checked_mul(decoder.upsample_factor())
                            .ok_or(AudioError::Runtime("decoder length scaling overflow".to_string()))
                    })
                    .collect::<AudioResult<Vec<_>>>()?
            } else if let Some(decoder) = self.config.fishaudio_decoder() {
                lengths_usize
                    .iter()
                    .map(|&length| {
                        length
                            .checked_mul(decoder.upsample_factor)
                            .ok_or(AudioError::Runtime("decoder length scaling overflow".to_string()))
                    })
                    .collect::<AudioResult<Vec<_>>>()?
            } else {
                lengths_usize.clone()
            };
            return AudioPcmBatch::new(
                Vec::<f32>::new().into_boxed_slice(),
                self.config.sample_rate(),
                channels,
                out_lengths.into_boxed_slice(),
            );
        }

        let codebook_major = tokens.to_packing(AudioTokenPacking::CodebookMajor);
        if let Some(fishaudio) = self.config.fishaudio_decoder() {
            let semantic_cardinality = u32::try_from(fishaudio.semantic_codebook_size).map_err(|_| {
                AudioError::Runtime("FishAudio semantic codebook cardinality exceeds u32 range".to_string())
            })?;
            let residual_cardinality = u32::try_from(fishaudio.codebook_size).map_err(|_| {
                AudioError::Runtime("FishAudio residual codebook cardinality exceeds u32 range".to_string())
            })?;

            for batch in 0..batch_size {
                for codebook in 0..codebook_major.codebooks() {
                    let cardinality = if codebook == 0 {
                        semantic_cardinality
                    } else {
                        residual_cardinality
                    };
                    for frame in 0..frames {
                        let token = codebook_major.get(batch, codebook, frame);
                        if token >= cardinality {
                            return Err(AudioError::InvalidCodecToken {
                                token,
                                cardinality,
                            });
                        }
                    }
                }
            }
            let (decoded, decode_profile) = fishaudio.decode_padded(
                self.options,
                codebook_major.tokens(),
                &lengths_usize,
                batch_size,
                codebook_major.codebooks(),
                frames,
            )?;
            if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
                *last_decode_profile = decode_profile;
            }
            let samples =
                unpack_padded_to_pcm(&decoded.samples, batch_size, decoded.channels, decoded.frames, &decoded.lengths)?;
            return AudioPcmBatch::new(
                samples.into_boxed_slice(),
                self.config.sample_rate(),
                decoded.channels,
                decoded.lengths.into_boxed_slice(),
            );
        }

        let mut tokens_i32 = vec![0_i32; codebook_major.tokens().len()];
        for (index, &token) in codebook_major.tokens().iter().enumerate() {
            if token >= self.config.codec_cardinality() {
                return Err(AudioError::InvalidCodecToken {
                    token,
                    cardinality: self.config.codec_cardinality(),
                });
            }
            if token > i32::MAX as u32 {
                return Err(AudioError::Runtime(format!("token at index {index} exceeds i32 kernel range: {token}")));
            }
            tokens_i32[index] = token as i32;
        }

        let context = Self::create_context()?;
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqDecodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq decode kernel: {err}")))?;

        let mut tokens_array = context.create_array(
            &[batch_size, self.config.num_groups(), frames],
            DataType::I32,
            "nanocodec_fsq_decode_tokens",
        );
        tokens_array.as_slice_mut::<i32>().copy_from_slice(&tokens_i32);

        let mut lengths_array = context.create_array(&[batch_size], DataType::I32, "nanocodec_fsq_decode_lengths");
        lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let output = context.create_array(
            &[batch_size, self.config.channels(), frames],
            DataType::F32,
            "nanocodec_fsq_decode_output",
        );

        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;

        command_buffer.with_compute_encoder(|compute_encoder| {
            kernel.encode(
                tokens_array.buffer(),
                output.buffer(),
                lengths_array.buffer(),
                num_groups_i32,
                frames_i32,
                codebook_dim_i32,
                self.config.num_levels_per_group(),
                batch_size_i32,
                compute_encoder,
            );
        });

        command_buffer.submit();
        command_buffer.wait_until_completed().map_err(|err| {
            AudioError::Runtime(format!("failed to wait for FSQ decode command buffer: {err}"))
        })?;

        let (padded_output, out_channels, out_frames, out_lengths) = if let Some(decoder) = self.config.decoder() {
            let decoded = decoder.decode_padded(
                output.as_slice::<f32>(),
                &lengths_usize,
                batch_size,
                self.config.channels(),
                frames,
            )?;
            (decoded.samples, decoded.channels, decoded.frames, decoded.lengths)
        } else {
            (output.as_slice::<f32>().to_vec(), self.config.channels(), frames, lengths_usize.clone())
        };

        let samples = unpack_padded_to_pcm(&padded_output, batch_size, out_channels, out_frames, &out_lengths)?;

        AudioPcmBatch::new(
            samples.into_boxed_slice(),
            self.config.sample_rate(),
            out_channels,
            out_lengths.into_boxed_slice(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        AudioDecodeStreamState, AudioDecodeStreamingMode, FishAudioCodecGraph, FishAudioConv1dLayer,
        FishAudioConvNeXtLayer, FishAudioConvTranspose1dLayer, FishAudioDecoderBlockLayer, FishAudioDecoderGraph,
        FishAudioNormLayer, FishAudioResidualUnitLayer, FishAudioVectorQuantizer, MatrixF32, NanoCodecFsqRuntime,
        NanoCodecFsqRuntimeConfig, RuntimePacking, Tensor3Json, convert_lalamo_transpose_weight_oih_to_iog,
        pack_pcm_to_padded, parse_fishaudio_tts_config_json, parse_lalamo_tts_config_json, parse_runtime_config_json,
        read_array_to_f32_vec, resolve_fishaudio_vocoder_data_type, unpack_padded_to_pcm, write_f32_slice_to_array,
    };
    use crate::{
        DataType,
        array::ArrayContextExt,
        audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking},
    };

    #[derive(Debug, Clone)]
    struct CpuPostModuleLayer {
        pre_mixer_norm: FishAudioNormLayer,
        qkv_projection: MatrixF32,
        out_projection: MatrixF32,
        pre_mlp_norm: FishAudioNormLayer,
        up_projection: MatrixF32,
        down_projection: MatrixF32,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        attention_scale: f32,
        sliding_window_size: Option<usize>,
    }

    #[derive(Debug, Clone)]
    struct CpuPostModule {
        rope_cosines: MatrixF32,
        rope_sines: MatrixF32,
        layers: Vec<CpuPostModuleLayer>,
        output_norm: FishAudioNormLayer,
        hidden_dim: usize,
    }

    fn load_optional_fishaudio_model_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("LALAMO_UZU_MODEL_PATH") {
            let path = PathBuf::from(path);
            return if path.join("config.json").exists() && path.join("model.safetensors").exists() {
                Some(path)
            } else {
                None
            };
        }

        let default = PathBuf::from("/private/tmp/lalamo_fishaudio_s1mini_convert");
        if default.join("config.json").exists() && default.join("model.safetensors").exists() {
            Some(default)
        } else {
            None
        }
    }

    fn build_cpu_post_module_for_test(graph: &FishAudioCodecGraph) -> AudioResult<CpuPostModule> {
        let reader = super::SafeTensorReader::open(Path::new(graph.weights_path.as_str()))?;
        let transformer = &graph.post_module_transformer_config;
        let first_layer = transformer
            .layer_configs
            .first()
            .ok_or(AudioError::Runtime("FishAudio post_module has no layers".to_string()))?;
        let crate::config::MixerConfig::Attention(first_attention) = &first_layer.mixer_config else {
            return Err(AudioError::Runtime(
                "FishAudio post_module first layer mixer must be AttentionConfig".to_string(),
            ));
        };
        let rope_head_dim = first_attention
            .head_dim
            .ok_or(AudioError::Runtime("FishAudio post_module head_dim missing".to_string()))?;
        let rope_cosines = super::read_matrix_f32(
            &reader,
            "audio_decoder.quantizer.post_module.global_rope.cosines",
            transformer.context_length,
            rope_head_dim,
        )?;
        let rope_sines = super::read_matrix_f32(
            &reader,
            "audio_decoder.quantizer.post_module.global_rope.sines",
            rope_cosines.rows,
            rope_cosines.cols,
        )?;
        let output_norm = super::read_fishaudio_norm_layer(
            &reader,
            "audio_decoder.quantizer.post_module.output_norm",
            transformer.output_norm_config.epsilon,
            transformer.output_norm_config.subtract_mean,
            false,
        )?;
        if output_norm.scales.len() != transformer.model_dim {
            return Err(AudioError::Runtime(format!(
                "FishAudio output_norm scale mismatch: expected {}, got {}",
                transformer.model_dim,
                output_norm.scales.len()
            )));
        }

        let mut layers = Vec::with_capacity(transformer.layer_configs.len());
        for (index, layer_config) in transformer.layer_configs.iter().enumerate() {
            let Some(pre_mixer_norm_config) = layer_config.pre_attention_norm_config.as_ref() else {
                return Err(AudioError::Runtime(
                    "FishAudio post_module requires pre_attention_norm_config".to_string(),
                ));
            };
            let pre_mixer_norm = super::read_fishaudio_norm_layer(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.pre_mixer_norm"),
                pre_mixer_norm_config.epsilon,
                pre_mixer_norm_config.subtract_mean,
                false,
            )?;
            let pre_mlp_norm = super::read_fishaudio_norm_layer(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.pre_mlp_norm"),
                layer_config.pre_mlp_norm_config.epsilon,
                layer_config.pre_mlp_norm_config.subtract_mean,
                false,
            )?;

            let crate::config::MixerConfig::Attention(attention_config) = &layer_config.mixer_config else {
                return Err(AudioError::Runtime(
                    "FishAudio post_module layer mixer must be AttentionConfig".to_string(),
                ));
            };
            let num_heads = attention_config
                .num_heads
                .ok_or(AudioError::Runtime("FishAudio attention num_heads missing".to_string()))?;
            let num_groups = attention_config
                .num_groups
                .ok_or(AudioError::Runtime("FishAudio attention num_groups missing".to_string()))?;
            let head_dim = attention_config
                .head_dim
                .ok_or(AudioError::Runtime("FishAudio attention head_dim missing".to_string()))?;
            if num_heads == 0 || num_groups == 0 || head_dim == 0 || num_heads % num_groups != 0 {
                return Err(AudioError::InvalidTokenCardinality);
            }

            let attention_dim = num_heads
                .checked_mul(head_dim)
                .ok_or(AudioError::Runtime("FishAudio attention dimension overflow".to_string()))?;
            let qkv_projection = super::read_matrix_f32(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.mixer.qkv_projection.weights"),
                attention_dim * 3,
                transformer.model_dim,
            )?;
            let out_projection = super::read_matrix_f32(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.mixer.out_projection.weights"),
                transformer.model_dim,
                attention_dim,
            )?;
            let up_projection = super::read_matrix_f32(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.mlp.up_projection.weights"),
                transformer
                    .hidden_dim
                    .checked_mul(2)
                    .ok_or(AudioError::Runtime("FishAudio hidden dimension overflow".to_string()))?,
                transformer.model_dim,
            )?;
            let down_projection = super::read_matrix_f32(
                &reader,
                &format!("audio_decoder.quantizer.post_module.layers.{index}.mlp.down_projection.weights"),
                transformer.model_dim,
                transformer.hidden_dim,
            )?;

            layers.push(CpuPostModuleLayer {
                pre_mixer_norm,
                qkv_projection,
                out_projection,
                pre_mlp_norm,
                up_projection,
                down_projection,
                num_heads,
                num_groups,
                head_dim,
                attention_scale: attention_config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
                sliding_window_size: attention_config.sliding_window_size,
            });
        }

        Ok(CpuPostModule {
            rope_cosines,
            rope_sines,
            layers,
            output_norm,
            hidden_dim: transformer.hidden_dim,
        })
    }

    fn apply_post_module_cpu_reference_for_test(
        graph: &FishAudioCodecGraph,
        post_module: &CpuPostModule,
        latent_nsc: &mut [f32],
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
    ) -> AudioResult<()> {
        if graph.post_module_model_dim != graph.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }

        for batch in 0..batch_size {
            let active_len = lengths[batch];
            if active_len == 0 {
                continue;
            }
            if active_len > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_len,
                    frames,
                });
            }

            let batch_base = batch * frames * graph.input_dim;
            let sequence = &mut latent_nsc[batch_base..batch_base + active_len * graph.input_dim];
            let mut x = sequence.to_vec();

            for layer in &post_module.layers {
                apply_post_module_cpu_layer_for_test(graph, post_module, layer, &mut x, active_len)?;
            }

            FishAudioCodecGraph::apply_norm_sequence(&mut x, active_len, graph.input_dim, &post_module.output_norm)?;
            sequence.copy_from_slice(&x);
        }

        Ok(())
    }

    fn apply_post_module_cpu_layer_for_test(
        graph: &FishAudioCodecGraph,
        post_module: &CpuPostModule,
        layer: &CpuPostModuleLayer,
        x: &mut [f32],
        active_len: usize,
    ) -> AudioResult<()> {
        let mut normed = x.to_vec();
        FishAudioCodecGraph::apply_norm_sequence(&mut normed, active_len, graph.input_dim, &layer.pre_mixer_norm)?;
        let qkv =
            FishAudioCodecGraph::linear_sequence(&normed, active_len, graph.input_dim, &layer.qkv_projection, None)?;
        let attention_dim = layer
            .num_heads
            .checked_mul(layer.head_dim)
            .ok_or(AudioError::Runtime("attention dimension overflow".to_string()))?;
        let group_dim = layer
            .num_groups
            .checked_mul(layer.head_dim)
            .ok_or(AudioError::Runtime("group dimension overflow".to_string()))?;
        if attention_dim != group_dim {
            return Err(AudioError::Runtime("post_module CPU reference requires num_heads == num_groups".to_string()));
        }

        let mut q = vec![0.0_f32; active_len * attention_dim];
        let mut k = vec![0.0_f32; active_len * attention_dim];
        let mut v = vec![0.0_f32; active_len * attention_dim];
        for token in 0..active_len {
            let row = &qkv[token * (attention_dim * 3)..(token + 1) * (attention_dim * 3)];
            q[token * attention_dim..(token + 1) * attention_dim].copy_from_slice(&row[..attention_dim]);
            k[token * attention_dim..(token + 1) * attention_dim]
                .copy_from_slice(&row[attention_dim..attention_dim * 2]);
            v[token * attention_dim..(token + 1) * attention_dim]
                .copy_from_slice(&row[attention_dim * 2..attention_dim * 3]);
        }

        let half = layer.head_dim / 2;
        let q_source = q.clone();
        let k_source = k.clone();
        for token in 0..active_len {
            for head in 0..layer.num_heads {
                let rope_row = token.min(post_module.rope_cosines.rows.saturating_sub(1));
                for dim in 0..layer.head_dim {
                    let cos = post_module.rope_cosines.values[rope_row * layer.head_dim + dim];
                    let sin = post_module.rope_sines.values[rope_row * layer.head_dim + dim];
                    let base = token * attention_dim + head * layer.head_dim;
                    let qv = q_source[base + dim];
                    let kv = k_source[base + dim];
                    let q_pair = if dim < half {
                        -q_source[base + dim + half]
                    } else {
                        q_source[base + dim - half]
                    };
                    let k_pair = if dim < half {
                        -k_source[base + dim + half]
                    } else {
                        k_source[base + dim - half]
                    };
                    q[base + dim] = qv * cos + q_pair * sin;
                    k[base + dim] = kv * cos + k_pair * sin;
                }
            }
        }

        let mut attention_output = vec![0.0_f32; active_len * attention_dim];
        for token in 0..active_len {
            let window_start =
                layer.sliding_window_size.map(|window| token.saturating_sub(window.saturating_sub(1))).unwrap_or(0);

            for head in 0..layer.num_heads {
                let query_offset = token * attention_dim + head * layer.head_dim;
                let mut logits = Vec::with_capacity(token + 1 - window_start);
                let mut max_logit = f32::NEG_INFINITY;
                for key_token in window_start..=token {
                    let key_offset = key_token * attention_dim + head * layer.head_dim;
                    let mut score = 0.0_f32;
                    for dim in 0..layer.head_dim {
                        score += q[query_offset + dim] * k[key_offset + dim];
                    }
                    score *= layer.attention_scale;
                    max_logit = max_logit.max(score);
                    logits.push((key_token, score));
                }
                let mut denom = 0.0_f32;
                for (_, score) in logits.iter_mut() {
                    *score = (*score - max_logit).exp();
                    denom += *score;
                }
                if denom <= 0.0 {
                    continue;
                }
                for (key_token, score) in logits {
                    let weight = score / denom;
                    let value_offset = key_token * attention_dim + head * layer.head_dim;
                    for dim in 0..layer.head_dim {
                        attention_output[query_offset + dim] += weight * v[value_offset + dim];
                    }
                }
            }
        }

        let attention_projected = FishAudioCodecGraph::linear_sequence(
            &attention_output,
            active_len,
            attention_dim,
            &layer.out_projection,
            None,
        )?;
        for (dst, value) in x.iter_mut().zip(attention_projected.iter()) {
            *dst += *value;
        }

        let mut mlp_in = x.to_vec();
        FishAudioCodecGraph::apply_norm_sequence(&mut mlp_in, active_len, graph.input_dim, &layer.pre_mlp_norm)?;
        let up =
            FishAudioCodecGraph::linear_sequence(&mlp_in, active_len, graph.input_dim, &layer.up_projection, None)?;
        let mut hidden = vec![0.0_f32; active_len * post_module.hidden_dim];
        for token in 0..active_len {
            let up_row = &up[token * post_module.hidden_dim * 2..(token + 1) * post_module.hidden_dim * 2];
            let hidden_row = &mut hidden[token * post_module.hidden_dim..(token + 1) * post_module.hidden_dim];
            for dim in 0..post_module.hidden_dim {
                let up_val = up_row[dim];
                let gate_val = up_row[post_module.hidden_dim + dim];
                let silu = gate_val / (1.0 + (-gate_val).exp());
                hidden_row[dim] = up_val * silu;
            }
        }
        let mlp_out = FishAudioCodecGraph::linear_sequence(
            &hidden,
            active_len,
            post_module.hidden_dim,
            &layer.down_projection,
            None,
        )?;
        for (dst, value) in x.iter_mut().zip(mlp_out.iter()) {
            *dst += *value;
        }

        Ok(())
    }

    #[test]
    fn config_rejects_invalid_eps() {
        let config = NanoCodecFsqRuntimeConfig::new(
            24_000,
            2,
            vec![8, 6].into_boxed_slice(),
            1.1,
            AudioTokenPacking::FrameMajor,
        );

        assert!(matches!(config, Err(AudioError::Runtime(_))));
    }

    #[test]
    fn pack_and_unpack_round_trip_variable_lengths() {
        let channels = 4usize;
        let lengths = vec![3usize, 1usize];
        let samples: Vec<f32> =
            (0..(lengths.iter().sum::<usize>() * channels)).map(|index| (index as f32 * 0.17 - 1.2).sin()).collect();

        let batch = AudioPcmBatch::new(samples.clone().into_boxed_slice(), 24_000, channels, lengths.clone().into())
            .expect("pcm");

        let (padded, got_lengths, _got_lengths_i32, frames) = pack_pcm_to_padded(&batch, channels).expect("pack");
        assert_eq!(frames, 3);
        assert_eq!(got_lengths, lengths);

        let unpacked = unpack_padded_to_pcm(&padded, 2, channels, frames, &lengths).expect("unpack");
        assert_eq!(samples, unpacked);
    }

    #[test]
    fn parses_runtime_config_from_nested_audio_codec() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "nanocodec_fsq",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6],
                "output_packing": "codebook_major"
            }
        });

        let parsed = parse_runtime_config_json(&tts_config).expect("parse runtime json");
        assert_eq!(parsed.sample_rate, 24_000);
        assert_eq!(parsed.num_groups, 2);
        assert_eq!(parsed.num_levels_per_group, vec![8, 6]);
        assert_eq!(parsed.eps, 1e-3);
        assert_eq!(parsed.output_packing, RuntimePacking::CodebookMajor);
    }

    #[test]
    fn runtime_config_builder_rejects_unsupported_type() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "other_codec",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6]
            }
        });

        let error = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect_err("must fail");
        match error {
            AudioError::Runtime(message) => assert!(message.contains("unsupported audio codec type")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn runtime_config_can_include_decoder_graph() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "nanocodec_fsq",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6],
                "decoder": {
                    "pre_conv": {
                        "weight": {
                            "shape": [2, 2, 1],
                            "values": [1.0, 0.0, 0.0, 1.0]
                        },
                        "bias": [0.0, 0.0]
                    },
                    "stages": [{
                        "activation_alpha": [1.0],
                        "upsample_conv": {
                            "weight": {
                                "shape": [2, 1, 4],
                                "values": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                            },
                            "bias": [0.0, 0.0],
                            "stride": 2,
                            "groups": 2
                        }
                    }],
                    "post_activation_alpha": [1.0],
                    "post_conv": {
                        "weight": {
                            "shape": [1, 2, 1],
                            "values": [1.0, 0.0]
                        },
                        "bias": [0.0]
                    }
                }
            }
        });

        let config = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect("runtime config");
        assert!(config.decoder().is_some());
        let decoder = config.decoder().expect("decoder");
        assert_eq!(decoder.upsample_factor(), 2);
    }

    #[test]
    fn parses_lalamo_tts_config_shape() {
        let tts_config = serde_json::json!({
            "text_decoder_config": {"type": "StubTextDecoderConfig"},
            "audio_decoder_config": {
                "samplerate": 22050,
                "quantizer_config": {
                    "num_groups": 13,
                    "quantizer_config": {
                        "num_levels": [8, 7, 6, 6],
                        "eps": 1e-3
                    }
                },
                "decoder_config": {
                    "activation_config": {
                        "leaky_relu_negative_slope": 0.01
                    }
                },
                "base_channels": 864,
                "up_sample_rates": [7, 7, 6, 3, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilations": [1, 3, 5]
            },
            "vocoder_config": {},
            "activation_precision": "float32"
        });

        let parsed = parse_lalamo_tts_config_json(&tts_config).expect("parse");
        assert_eq!(parsed.audio_decoder_config.samplerate, 22050);
        assert_eq!(parsed.audio_decoder_config.quantizer_config.num_groups, 13);
        assert_eq!(parsed.audio_decoder_config.quantizer_config.quantizer_config.num_levels, vec![8, 7, 6, 6]);
        assert_eq!(parsed.audio_decoder_config.up_sample_rates, vec![7, 7, 6, 3, 2]);
    }

    #[test]
    fn fishaudio_dac_config_requires_model_weights() {
        let tts_config = serde_json::json!({
            "audio_decoder_config": {
                "type": "DescriptAudioCodecConfig",
                "samplerate": 44100,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "semantic_codebook_size": 4096,
                "input_dim": 1024,
                "downsample_factor": [2, 2],
                "decoder_rates": [8, 8, 4, 2]
            }
        });

        let error = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, Path::new("."))
            .expect_err("fishaudio runtime must require exported model.safetensors");
        match error {
            AudioError::Runtime(message) => {
                assert!(message.contains("model.safetensors"), "unexpected error message: {message}");
            },
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn fishaudio_vocoder_dtype_uses_lalamo_export_precision() {
        let tts_config = serde_json::json!({
            "activation_precision": "float16",
            "audio_decoder_config": {
                "type": "DescriptAudioCodecConfig",
                "samplerate": 44100,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "semantic_codebook_size": 4096,
                "input_dim": 1024,
                "downsample_factor": [2, 2],
                "decoder_rates": [8, 8, 4, 2],
                "precision": "float16",
                "quantizer_config": {
                    "precision": "float16",
                    "post_module_config": {
                        "global_rope_config": null,
                        "local_rope_config": null,
                        "layer_configs": [],
                        "output_norm_config": {
                            "scale_precision": "float32",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-5,
                            "scale_offset": null,
                            "upcast_mode": "full_layer",
                            "subtract_mean": true
                        },
                        "model_dim": 1024,
                        "hidden_dim": 1024,
                        "context_length": 1
                    },
                    "upsampler_config": {
                        "block_configs": []
                    }
                }
            }
        });

        let parsed = parse_fishaudio_tts_config_json(&tts_config).expect("parse fishaudio config");
        let dtype = resolve_fishaudio_vocoder_data_type(&parsed).expect("resolve dtype");
        assert_eq!(dtype, DataType::F16);
    }

    #[test]
    fn fishaudio_vocoder_dtype_rejects_conflicting_export_precision() {
        let tts_config = serde_json::json!({
            "activation_precision": "float32",
            "audio_decoder_config": {
                "type": "DescriptAudioCodecConfig",
                "samplerate": 44100,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "semantic_codebook_size": 4096,
                "input_dim": 1024,
                "downsample_factor": [2, 2],
                "decoder_rates": [8, 8, 4, 2],
                "precision": "float16",
                "quantizer_config": {
                    "post_module_config": {
                        "global_rope_config": null,
                        "local_rope_config": null,
                        "layer_configs": [],
                        "output_norm_config": {
                            "scale_precision": "float32",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-5,
                            "scale_offset": null,
                            "upcast_mode": "full_layer",
                            "subtract_mean": true
                        },
                        "model_dim": 1024,
                        "hidden_dim": 1024,
                        "context_length": 1
                    },
                    "upsampler_config": {
                        "block_configs": []
                    }
                }
            }
        });

        let parsed = parse_fishaudio_tts_config_json(&tts_config).expect("parse fishaudio config");
        let error = resolve_fishaudio_vocoder_data_type(&parsed).expect_err("must reject conflicting precision");
        match error {
            AudioError::Runtime(message) => assert!(message.contains("conflicting FishAudio precision")),
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn fishaudio_vocoder_dtype_requires_export_precision() {
        let tts_config = serde_json::json!({
            "audio_decoder_config": {
                "type": "DescriptAudioCodecConfig",
                "samplerate": 44100,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "semantic_codebook_size": 4096,
                "input_dim": 1024,
                "downsample_factor": [2, 2],
                "decoder_rates": [8, 8, 4, 2],
                "quantizer_config": {
                    "post_module_config": {
                        "global_rope_config": null,
                        "local_rope_config": null,
                        "layer_configs": [],
                        "output_norm_config": {
                            "scale_precision": "float32",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-5,
                            "scale_offset": null,
                            "upcast_mode": "full_layer",
                            "subtract_mean": true
                        },
                        "model_dim": 1024,
                        "hidden_dim": 1024,
                        "context_length": 1
                    },
                    "upsampler_config": {
                        "block_configs": []
                    }
                }
            }
        });

        let parsed = parse_fishaudio_tts_config_json(&tts_config).expect("parse fishaudio config");
        let error = resolve_fishaudio_vocoder_data_type(&parsed).expect_err("must require export precision");
        match error {
            AudioError::Runtime(message) => assert!(message.contains("missing FishAudio precision")),
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn fishaudio_runtime_decode_path_handles_empty_frames() {
        let mut config = NanoCodecFsqRuntimeConfig::new(
            44_100,
            2,
            vec![4].into_boxed_slice(),
            1e-3,
            AudioTokenPacking::CodebookMajor,
        )
        .expect("config");

        let one_by_one = FishAudioConv1dLayer {
            weight: vec![1.0],
            bias: vec![0.0],
            cin: 1,
            cout: 1,
            kernel_size: 1,
            dilation: 1,
            groups: 1,
        };

        config.fishaudio_decoder = Some(FishAudioCodecGraph {
            semantic_quantizer: FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 2,
                    cols: 1,
                    values: vec![0.0, 1.0],
                },
                out_proj: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![1.0],
                },
                out_bias: vec![0.0],
            },
            residual_quantizers: vec![FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 2,
                    cols: 1,
                    values: vec![0.0, 1.0],
                },
                out_proj: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![1.0],
                },
                out_bias: vec![0.0],
            }],
            post_module_model_dim: 1,
            post_module_transformer_config: serde_json::from_value(serde_json::json!({
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [],
                "output_norm_config": {
                    "scale_precision": "float32",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-5,
                    "scale_offset": null,
                    "upcast_mode": "full_layer",
                    "subtract_mean": true
                },
                "model_dim": 1,
                "hidden_dim": 1,
                "context_length": 1
            }))
            .expect("post module transformer config"),
            weights_path: String::new(),
            decoder: FishAudioDecoderGraph {
                first_conv: one_by_one.clone(),
                upsample_blocks: Vec::new(),
                decoder_blocks: Vec::new(),
                final_snake_alpha: vec![1.0],
                final_conv: one_by_one,
                upsample_factor: 1,
            },
            codebook_size: 2,
            semantic_codebook_size: 2,
            input_dim: 1,
            total_codebooks: 2,
            upsample_factor: 1,
            vocoder_data_type: DataType::F32,
        });

        let runtime = super::NanoCodecFsqRuntime::new(config);
        let tokens = crate::audio::AudioTokenGrid::new(
            Vec::new().into_boxed_slice(),
            1,
            2,
            0,
            vec![0usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("token grid");

        let pcm = runtime.decode(&tokens).expect("decode");
        assert_eq!(pcm.sample_rate(), 44_100);
        assert_eq!(pcm.channels(), 1);
        assert_eq!(pcm.lengths(), &[0usize]);
        assert!(pcm.samples().is_empty());
    }

    #[test]
    fn fishaudio_quantizer_decode_gpu_matches_cpu_reference_small_graph() {
        let graph = FishAudioCodecGraph {
            semantic_quantizer: FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 4,
                    cols: 3,
                    values: vec![
                        0.0, 0.1, 0.2, //
                        0.3, 0.4, 0.5, //
                        0.6, 0.7, 0.8, //
                        0.9, 1.0, 1.1, //
                    ],
                },
                out_proj: MatrixF32 {
                    rows: 2,
                    cols: 3,
                    values: vec![
                        0.2, -0.1, 0.3, //
                        0.5, 0.4, -0.2, //
                    ],
                },
                out_bias: vec![0.01, -0.02],
            },
            residual_quantizers: vec![FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 5,
                    cols: 3,
                    values: vec![
                        0.2, -0.2, 0.0, //
                        0.1, 0.0, 0.3, //
                        -0.1, 0.4, 0.2, //
                        0.7, -0.3, 0.5, //
                        0.6, 0.8, -0.4, //
                    ],
                },
                out_proj: MatrixF32 {
                    rows: 2,
                    cols: 3,
                    values: vec![
                        0.3, 0.1, -0.2, //
                        -0.4, 0.6, 0.2, //
                    ],
                },
                out_bias: vec![0.03, -0.01],
            }],
            post_module_model_dim: 2,
            post_module_transformer_config: serde_json::from_value(serde_json::json!({
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [],
                "output_norm_config": {
                    "scale_precision": "float32",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-5,
                    "scale_offset": null,
                    "upcast_mode": "full_layer",
                    "subtract_mean": true
                },
                "model_dim": 2,
                "hidden_dim": 2,
                "context_length": 1
            }))
            .expect("post module transformer config"),
            weights_path: String::new(),
            decoder: FishAudioDecoderGraph {
                first_conv: FishAudioConv1dLayer {
                    weight: vec![1.0, 0.0],
                    bias: vec![0.0],
                    cin: 2,
                    cout: 1,
                    kernel_size: 1,
                    dilation: 1,
                    groups: 1,
                },
                upsample_blocks: Vec::new(),
                decoder_blocks: Vec::new(),
                final_snake_alpha: vec![1.0],
                final_conv: FishAudioConv1dLayer {
                    weight: vec![1.0],
                    bias: vec![0.0],
                    cin: 1,
                    cout: 1,
                    kernel_size: 1,
                    dilation: 1,
                    groups: 1,
                },
                upsample_factor: 1,
            },
            codebook_size: 5,
            semantic_codebook_size: 4,
            input_dim: 2,
            total_codebooks: 2,
            upsample_factor: 1,
            vocoder_data_type: DataType::F32,
        };

        let batch_size = 2usize;
        let frames = 4usize;
        let lengths = vec![4usize, 2usize];
        let tokens = vec![
            // batch 0 semantic
            0, 1, 2, 3, // batch 0 residual
            4, 5, 2, 1, // batch 1 semantic
            3, 2, 1, 0, // batch 1 residual
            0, 4, 3, 2,
        ];

        let expected = graph
            .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, graph.total_codebooks, frames)
            .expect("cpu decode");
        let actual = graph
            .decode_quantizer_to_nsc(&tokens, &lengths, batch_size, graph.total_codebooks, frames)
            .expect("gpu decode");

        assert_eq!(actual.len(), expected.len());
        for (index, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
            let delta = (exp - got).abs();
            assert!(delta <= 1e-5, "mismatch at index {index}: expected {exp}, got {got}, delta={delta}");
        }
    }

    #[test]
    fn fishaudio_post_module_gpu_matches_cpu_reference_on_real_export() {
        let Some(model_path) = load_optional_fishaudio_model_path() else {
            println!("Skipping FishAudio post-module parity test: set LALAMO_UZU_MODEL_PATH");
            return;
        };

        let config_bytes = std::fs::read(model_path.join("config.json")).expect("read model config");
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).expect("parse model config");
        let tts_config = config_json
            .get("model_config")
            .and_then(|value| value.get("tts_config"))
            .expect("model_config.tts_config")
            .clone();
        let runtime_config = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, &model_path)
            .expect("runtime config");
        let fishaudio = runtime_config.fishaudio_decoder().expect("fishaudio decoder");
        let cpu_post_module = build_cpu_post_module_for_test(fishaudio).expect("cpu post-module");

        let batch_size = 1usize;
        let frames = 6usize;
        let lengths = vec![6usize];
        let mut tokens = vec![0_u32; batch_size * fishaudio.total_codebooks * frames];
        for frame in 0..frames {
            let semantic = (frame * 17) % fishaudio.semantic_codebook_size;
            tokens[frame] = semantic as u32;
            for residual in 0..(fishaudio.total_codebooks - 1) {
                let index = ((residual + 1) * frames) + frame;
                let value = ((frame + 3) * (residual + 5)) % fishaudio.codebook_size;
                tokens[index] = value as u32;
            }
        }
        let latent_cpu = fishaudio
            .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
            .expect("decode quantizer cpu reference");
        let latent = fishaudio
            .decode_quantizer_to_nsc(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
            .expect("decode quantizer");
        assert_eq!(latent.len(), latent_cpu.len());
        let mut quantizer_max_abs_diff = 0.0_f32;
        for (&cpu_value, &gpu_value) in latent_cpu.iter().zip(latent.iter()) {
            quantizer_max_abs_diff = quantizer_max_abs_diff.max((cpu_value - gpu_value).abs());
        }
        println!("quantizer latent parity: max_abs_diff={quantizer_max_abs_diff}");
        assert!(quantizer_max_abs_diff <= 1e-4, "quantizer decode parity mismatch: {quantizer_max_abs_diff}");

        let mut cpu = latent.clone();
        apply_post_module_cpu_reference_for_test(fishaudio, &cpu_post_module, &mut cpu, &lengths, batch_size, frames)
            .expect("cpu post-module");
        let context = NanoCodecFsqRuntime::create_context().expect("create metal context");
        let mut latent_array = context.create_array(
            &[batch_size, frames, fishaudio.input_dim],
            fishaudio.vocoder_data_type,
            "fishaudio_test_post_module_single_input_nsc",
        );
        write_f32_slice_to_array(&mut latent_array, &latent).expect("write latent to array");
        let mut profile = None;
        let gpu = read_array_to_f32_vec(
            &fishaudio
                .apply_post_module_gpu_on_array(&context, &latent_array, &lengths, batch_size, frames, &mut profile)
                .expect("gpu post-module"),
        )
        .expect("read gpu post-module");

        let mut max_abs_diff = 0.0_f32;
        let mut sum_sq_diff = 0.0_f64;
        for (&cpu_value, &gpu_value) in cpu.iter().zip(gpu.iter()) {
            let diff = (cpu_value - gpu_value).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_sq_diff += f64::from(diff * diff);
        }
        let rmse = (sum_sq_diff / cpu.len() as f64).sqrt() as f32;
        println!("post-module latent parity: max_abs_diff={max_abs_diff}, rmse={rmse}");

        assert!(max_abs_diff <= 1e-4, "post-module max_abs_diff too high: {max_abs_diff}, rmse={rmse}");
        assert!(rmse <= 1e-5, "post-module rmse too high: {rmse}, max_abs_diff={max_abs_diff}");
    }

    #[test]
    fn fishaudio_post_module_gpu_general_path_batches_lengths_in_one_command_buffer() {
        let Some(model_path) = load_optional_fishaudio_model_path() else {
            println!("Skipping FishAudio multi-batch post-module test: set LALAMO_UZU_MODEL_PATH");
            return;
        };

        let config_bytes = std::fs::read(model_path.join("config.json")).expect("read model config");
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).expect("parse model config");
        let tts_config = config_json
            .get("model_config")
            .and_then(|value| value.get("tts_config"))
            .expect("model_config.tts_config")
            .clone();
        let runtime_config = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, &model_path)
            .expect("runtime config");
        let fishaudio = runtime_config.fishaudio_decoder().expect("fishaudio decoder");
        let cpu_post_module = build_cpu_post_module_for_test(fishaudio).expect("cpu post-module");

        let batch_size = 3usize;
        let frames = 6usize;
        let lengths = vec![6usize, 4usize, 4usize];
        let mut tokens = vec![0_u32; batch_size * fishaudio.total_codebooks * frames];
        for batch in 0..batch_size {
            for frame in 0..frames {
                let semantic = ((batch + 1) * 17 + frame * 13) % fishaudio.semantic_codebook_size;
                let semantic_index = ((batch * fishaudio.total_codebooks) * frames) + frame;
                tokens[semantic_index] = semantic as u32;
                for residual in 0..(fishaudio.total_codebooks - 1) {
                    let index = ((batch * fishaudio.total_codebooks) + residual + 1) * frames + frame;
                    let value = ((batch + 3) * (frame + 5) * (residual + 7)) % fishaudio.codebook_size;
                    tokens[index] = value as u32;
                }
            }
        }

        let latent_cpu = fishaudio
            .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
            .expect("decode quantizer cpu reference");
        let mut cpu = latent_cpu.clone();
        apply_post_module_cpu_reference_for_test(fishaudio, &cpu_post_module, &mut cpu, &lengths, batch_size, frames)
            .expect("cpu post-module");

        let context = NanoCodecFsqRuntime::create_context().expect("create metal context");
        let mut latent = context.create_array(
            &[batch_size, frames, fishaudio.input_dim],
            fishaudio.vocoder_data_type,
            "fishaudio_test_post_module_input_nsc",
        );
        write_f32_slice_to_array(&mut latent, &latent_cpu).expect("write latent to array");
        let mut profile = Some(super::AudioDecodeProfile::default());
        let output = fishaudio
            .apply_post_module_gpu_on_array(&context, &latent, &lengths, batch_size, frames, &mut profile)
            .expect("gpu post-module");
        let gpu = read_array_to_f32_vec(&output).expect("read gpu post-module");

        let mut max_abs_diff = 0.0_f32;
        let mut sum_sq_diff = 0.0_f64;
        for (&cpu_value, &gpu_value) in cpu.iter().zip(gpu.iter()) {
            let diff = (cpu_value - gpu_value).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_sq_diff += f64::from(diff * diff);
        }
        let rmse = (sum_sq_diff / cpu.len() as f64).sqrt() as f32;
        println!("multi-batch post-module latent parity: max_abs_diff={max_abs_diff}, rmse={rmse}");

        assert!(max_abs_diff <= 2e-4, "post-module max_abs_diff too high: {max_abs_diff}, rmse={rmse}");
        assert!(rmse <= 2e-5, "post-module rmse too high: {rmse}, max_abs_diff={max_abs_diff}");

        let profile = profile.expect("profile");
        assert_eq!(profile.command_buffers.len(), batch_size, "expected one post-module command buffer per batch item");
        assert_eq!(profile.command_buffers[0].label, "post_module_len_4_batch_1");
        assert_eq!(profile.command_buffers[1].label, "post_module_len_4_batch_2");
        assert_eq!(profile.command_buffers[2].label, "post_module_len_6_batch_0");
    }

    #[test]
    fn transpose_weight_conversion_matches_expected_layout() {
        let weight_oih = Tensor3Json {
            shape: [2, 2, 2],
            values: vec![
                // out=0
                1.0, 2.0, // in_group=0
                3.0, 4.0, // in_group=1
                // out=1
                5.0, 6.0, // in_group=0
                7.0, 8.0, // in_group=1
            ],
        };

        let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, 4, 2, 2).expect("convert");
        assert_eq!(converted.shape, [4, 1, 2]);
        assert_eq!(
            converted.values,
            vec![
                1.0, 2.0, // in=0, out_group=0
                3.0, 4.0, // in=1, out_group=0
                5.0, 6.0, // in=2, out_group=0
                7.0, 8.0, // in=3, out_group=0
            ]
        );
    }

    #[test]
    fn fishaudio_streaming_context_matches_expected_value() {
        let convnext = FishAudioConvNeXtLayer {
            depthwise_conv: FishAudioConv1dLayer {
                weight: vec![1.0; 6],
                bias: vec![0.0; 2],
                cin: 2,
                cout: 2,
                kernel_size: 3,
                dilation: 1,
                groups: 2,
            },
            norm: FishAudioNormLayer {
                scales: vec![1.0, 1.0],
                biases: Some(vec![0.0, 0.0]),
                epsilon: 1e-5,
                subtract_mean: true,
            },
            pwconv1: MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
            pwconv1_bias: vec![0.0, 0.0],
            pwconv2: MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
            pwconv2_bias: vec![0.0, 0.0],
        };
        let residual_unit = FishAudioResidualUnitLayer {
            snake1_alpha: vec![1.0, 1.0],
            conv1: FishAudioConv1dLayer {
                weight: vec![1.0; 12],
                bias: vec![0.0, 0.0],
                cin: 2,
                cout: 2,
                kernel_size: 3,
                dilation: 1,
                groups: 1,
            },
            snake2_alpha: vec![1.0, 1.0],
            conv2: FishAudioConv1dLayer {
                weight: vec![1.0; 12],
                bias: vec![0.0, 0.0],
                cin: 2,
                cout: 2,
                kernel_size: 3,
                dilation: 1,
                groups: 1,
            },
        };

        let graph = FishAudioCodecGraph {
            semantic_quantizer: FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![0.0, 0.0, 0.0, 0.0],
                },
                out_proj: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                },
                out_bias: vec![0.0, 0.0],
            },
            residual_quantizers: vec![],
            post_module_model_dim: 2,
            post_module_transformer_config: serde_json::from_value(serde_json::json!({
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [],
                "output_norm_config": {
                    "scale_precision": "float32",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-5,
                    "scale_offset": null,
                    "upcast_mode": "full_layer",
                    "subtract_mean": true
                },
                "model_dim": 2,
                "hidden_dim": 2,
                "context_length": 1
            }))
            .expect("post module transformer config"),
            weights_path: String::new(),
            decoder: FishAudioDecoderGraph {
                first_conv: FishAudioConv1dLayer {
                    weight: vec![1.0; 8],
                    bias: vec![0.0, 0.0],
                    cin: 2,
                    cout: 2,
                    kernel_size: 3,
                    dilation: 1,
                    groups: 1,
                },
                upsample_blocks: vec![(
                    FishAudioConvTranspose1dLayer {
                        weight: vec![1.0; 16],
                        bias: vec![0.0, 0.0],
                        cin: 2,
                        cout: 2,
                        kernel_size: 4,
                        stride: 2,
                        groups: 1,
                    },
                    convnext,
                )],
                decoder_blocks: vec![FishAudioDecoderBlockLayer {
                    snake_alpha: vec![1.0, 1.0],
                    trans_conv: FishAudioConvTranspose1dLayer {
                        weight: vec![1.0; 16],
                        bias: vec![0.0, 0.0],
                        cin: 2,
                        cout: 2,
                        kernel_size: 4,
                        stride: 2,
                        groups: 1,
                    },
                    res_unit1: residual_unit.clone(),
                    res_unit2: residual_unit.clone(),
                    res_unit3: residual_unit,
                }],
                final_snake_alpha: vec![1.0, 1.0],
                final_conv: FishAudioConv1dLayer {
                    weight: vec![1.0; 10],
                    bias: vec![0.0],
                    cin: 2,
                    cout: 1,
                    kernel_size: 5,
                    dilation: 1,
                    groups: 1,
                },
                upsample_factor: 4,
            },
            codebook_size: 2,
            semantic_codebook_size: 2,
            input_dim: 2,
            total_codebooks: 1,
            upsample_factor: 4,
            vocoder_data_type: DataType::F32,
        };

        // Manual derivation:
        // final_conv (k=5): 4
        // decoder block residual stack: +12 => 16, then trans_conv (k=4,s=2): ceil((16+3)/2)=10
        // first_conv (k=3): +2 => 12
        // upsample block convnext dwconv (k=3): +2 => 14, then trans_conv (k=4,s=2): ceil((14+3)/2)=9
        assert_eq!(graph.streaming_vocoder_context_frames().expect("stream context"), 9);
    }

    #[test]
    fn stream_delta_extraction_with_window_offset_matches_expected_slice() {
        let mut state =
            AudioDecodeStreamState::new(1, 2, 16, AudioDecodeStreamingMode::IncrementalStateful).expect("state");
        let first_delta = AudioTokenGrid::new(
            vec![1_u32, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice(),
            1,
            2,
            4,
            vec![4usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("first token grid");
        state.append_delta(&first_delta).expect("append first delta");
        let first_decoded = crate::audio::nanocodec::decoder::DecodedPaddedAudio {
            samples: (0..8).map(|value| value as f32).collect(),
            channels: 1,
            frames: 8,
            lengths: vec![8],
        };
        let first_out = state.extract_delta_from_padded_with_offset(&first_decoded, 0, 2).expect("first extract");
        assert_eq!(first_out.lengths, vec![8]);
        assert_eq!(first_out.samples, (0..8).map(|value| value as f32).collect::<Vec<_>>());

        let second_delta = AudioTokenGrid::new(
            vec![9_u32, 10, 11, 12].into_boxed_slice(),
            1,
            2,
            2,
            vec![2usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("second token grid");
        state.append_delta(&second_delta).expect("append second delta");
        let window_decoded = crate::audio::nanocodec::decoder::DecodedPaddedAudio {
            // Global output range is [4, 12), local window starts at global sample 4.
            samples: (4..12).map(|value| value as f32).collect(),
            channels: 1,
            frames: 8,
            lengths: vec![8],
        };
        let second_out = state.extract_delta_from_padded_with_offset(&window_decoded, 4, 2).expect("second extract");
        assert_eq!(second_out.lengths, vec![4]);
        assert_eq!(second_out.frames, 4);
        assert_eq!(second_out.samples, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn incremental_stream_state_evicts_old_frames_with_bounded_workspace() {
        let mut state =
            AudioDecodeStreamState::new(1, 2, 4, AudioDecodeStreamingMode::IncrementalStateful).expect("state");
        let first_delta = AudioTokenGrid::new(
            vec![10_u32, 11, 12, 20, 21, 22].into_boxed_slice(),
            1,
            2,
            3,
            vec![3usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("first token grid");
        state.append_delta(&first_delta).expect("append first delta");

        let second_delta = AudioTokenGrid::new(
            vec![13_u32, 14, 15, 23, 24, 25].into_boxed_slice(),
            1,
            2,
            3,
            vec![3usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("second token grid");
        state.append_delta(&second_delta).expect("append second delta");

        assert_eq!(state.total_frames(), 6);
        assert_eq!(state.stored_frame_start, 2);
        assert_eq!(state.stored_frames(), 4);

        let (tokens, lengths, frames) = state.flatten_window(2, 6).expect("flatten retained window");
        assert_eq!(frames, 4);
        assert_eq!(lengths, &[4usize]);
        assert_eq!(tokens, &[12, 13, 14, 15, 22, 23, 24, 25]);
        assert!(state.to_full_grid().is_err(), "full-grid decode should fail after eviction");
    }

    #[test]
    fn prefix_fallback_stream_state_rejects_workspace_overflow() {
        let mut state = AudioDecodeStreamState::new(1, 2, 4, AudioDecodeStreamingMode::PrefixFallback).expect("state");
        let first_delta = AudioTokenGrid::new(
            vec![1_u32, 2, 3, 4, 5, 6].into_boxed_slice(),
            1,
            2,
            3,
            vec![3usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("first token grid");
        state.append_delta(&first_delta).expect("append first delta");

        let overflow_delta = AudioTokenGrid::new(
            vec![7_u32, 8, 9, 10].into_boxed_slice(),
            1,
            2,
            2,
            vec![2usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("overflow token grid");
        let err = state.append_delta(&overflow_delta).expect_err("prefix fallback must reject overflow");
        assert!(err.to_string().contains("workspace exceeded"), "unexpected prefix overflow error: {err}");
    }
}
