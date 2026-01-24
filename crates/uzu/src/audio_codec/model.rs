use std::{fs::File, path::Path, rc::Rc};

use serde::Deserialize;

use crate::{
    Array,
    DataType,
    DeviceContext,
    backends::metal::{KernelDataType, MTLContext},
    backends::metal::kernel::{FsqEncodeArguments, FsqEncodeKernel},
};

use super::{NanoCodecDecoder, NanoCodecEncoder, NanoCodecError};

#[derive(Debug, Deserialize)]
struct ExportedVectorQuantizerConfig {
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct ExportedAudioEncoderConfig {
    down_sample_rates: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ExportedNanoCodecConfig {
    sample_rate: usize,
    samples_per_frame: usize,
    vector_quantizer: ExportedVectorQuantizerConfig,
    audio_encoder: ExportedAudioEncoderConfig,
}

pub struct NanoCodecModel {
    pub sample_rate: usize,
    pub samples_per_frame: usize,

    encoder: NanoCodecEncoder,
    decoder: NanoCodecDecoder,
    fsq_encode: FsqEncodeKernel,

    num_codebooks: usize,
    codebook_dim_per_group: usize,
    num_levels_per_group: Box<[i32]>,
    dim_base_index: Box<[i32]>,
    fsq_eps: f32,
}

impl NanoCodecModel {
    pub fn load_from_export_dir(
        context: Rc<MTLContext>,
        export_dir: &Path,
    ) -> Result<Self, NanoCodecError> {
        let config_path = export_dir.join("nanocodec_config.json");
        let f = File::open(&config_path)?;
        let cfg: ExportedNanoCodecConfig = serde_json::from_reader(f)
            .map_err(|e| NanoCodecError::InvalidWeights(format!("Invalid nanocodec_config.json: {e}")))?;

        let num_codebooks = cfg.vector_quantizer.num_groups;
        let num_levels_per_group = cfg.vector_quantizer.num_levels_per_group;
        let codebook_dim_per_group = num_levels_per_group.len();

        let dim_base_index = {
            let mut base = 1i32;
            let mut out = Vec::with_capacity(codebook_dim_per_group);
            for &levels in num_levels_per_group.iter() {
                out.push(base);
                base = base.saturating_mul(levels);
            }
            out.into_boxed_slice()
        };

        let encoder = NanoCodecEncoder::load_from_export_dir(
            context.clone(),
            export_dir,
            cfg.audio_encoder.down_sample_rates.into_boxed_slice(),
        )?;
        let decoder = NanoCodecDecoder::load(
            context.clone(),
            &export_dir.join("audio_decoder.safetensors"),
            num_codebooks,
            codebook_dim_per_group,
            num_levels_per_group.clone().into_boxed_slice(),
        )?;

        let fsq_encode =
            FsqEncodeKernel::new(&context, KernelDataType::Float32)?;

        Ok(Self {
            sample_rate: cfg.sample_rate,
            samples_per_frame: cfg.samples_per_frame,
            encoder,
            decoder,
            fsq_encode,
            num_codebooks,
            codebook_dim_per_group,
            num_levels_per_group: num_levels_per_group.into_boxed_slice(),
            dim_base_index,
            fsq_eps: 1e-3,
        })
    }

    pub fn load_from_nemo(
        context: Rc<MTLContext>,
        nemo_path: &Path,
    ) -> Result<Self, NanoCodecError> {
        let tmp = tempfile::TempDir::new()?;
        let _ = super::nemo::export_nanocodec_from_nemo(nemo_path, tmp.path())
            .map_err(|e| NanoCodecError::InvalidWeights(format!("Failed to export from .nemo: {e}")))?;
        Self::load_from_export_dir(context, tmp.path())
    }

    /// Encode waveform to tokens.
    ///
    /// Input `audio` is expected as a batch-major flat buffer `[B, T]` with `seq_len=T`.
    pub fn encode(
        &self,
        audio: &[f32],
        audio_len: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<i32>, Vec<i32>, usize), NanoCodecError> {
        let (padded_audio, padded_len, padded_seq_len) = pad_audio_batch(
            audio,
            audio_len,
            batch_size,
            seq_len,
            self.samples_per_frame,
        )?;

        let (encoded, encoded_len) = self.encoder.encode_latents(
            &padded_audio,
            &padded_len,
            batch_size,
            padded_seq_len,
        )?;

        let encoded_seq_len = *encoded
            .shape()
            .last()
            .ok_or_else(|| NanoCodecError::InvalidWeights("encoded must be 3D".into()))?;

        // lengths buffer
        let mut lengths_arr = self.encoder_context().array(
            &[batch_size],
            DataType::I32,
            "nanocodec_tokens_len".into(),
        );
        lengths_arr
            .as_slice_mut::<i32>()
            .expect("dtype")
            .copy_from_slice(&encoded_len);

        // output tokens buffer
        let mut tokens_arr = self.encoder_context().array(
            &[batch_size, self.num_codebooks, encoded_seq_len],
            DataType::I32,
            "nanocodec_tokens".into(),
        );

        let command_buffer = self.encoder_context().command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.fsq_encode.encode(
            &encoder,
            FsqEncodeArguments {
                input: encoded.backend_buffer(),
                tokens: tokens_arr.backend_buffer(),
                lengths: lengths_arr.backend_buffer(),
                batch_size,
                num_groups: self.num_codebooks,
                seq_len: encoded_seq_len,
                codebook_dim_per_group: self.codebook_dim_per_group,
                num_levels_per_group: self.num_levels_per_group.clone(),
                dim_base_index: self.dim_base_index.clone(),
                eps: self.fsq_eps,
            },
        )?;

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let tokens = tokens_arr
            .as_slice::<i32>()
            .expect("dtype")
            .to_vec();

        Ok((tokens, encoded_len, encoded_seq_len))
    }

    pub fn decode(
        &self,
        tokens: &[i32],
        tokens_len: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), NanoCodecError> {
        self.decoder.decode(tokens, tokens_len, batch_size, seq_len)
    }

    pub fn roundtrip(
        &self,
        audio: &[f32],
        audio_len: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), NanoCodecError> {
        let (tokens, tokens_len, tok_seq_len) =
            self.encode(audio, audio_len, batch_size, seq_len)?;
        self.decode(&tokens, &tokens_len, batch_size, tok_seq_len)
    }

    fn encoder_context(&self) -> &Rc<MTLContext> {
        // Both encoder and decoder share the same Rc<MTLContext>.
        // Encoder stores it internally but does not expose it; use decoder's.
        &self.decoder.context
    }
}

fn pad_audio_batch(
    audio: &[f32],
    audio_len: &[i32],
    batch_size: usize,
    seq_len: usize,
    samples_per_frame: usize,
) -> Result<(Vec<f32>, Vec<i32>, usize), NanoCodecError> {
    if audio_len.len() != batch_size {
        return Err(NanoCodecError::InvalidWeights(
            "audio_len len must match batch_size".into(),
        ));
    }
    if audio.len() != batch_size * seq_len {
        return Err(NanoCodecError::InvalidWeights(
            "audio slice length must be batch_size * seq_len".into(),
        ));
    }
    if samples_per_frame == 0 {
        return Err(NanoCodecError::InvalidWeights(
            "samples_per_frame must be > 0".into(),
        ));
    }

    let padded_len: Vec<i32> = audio_len
        .iter()
        .map(|&l| {
            let l_usize: usize = l.max(0) as usize;
            let rem = l_usize % samples_per_frame;
            let padded = if rem == 0 {
                l_usize
            } else {
                l_usize + (samples_per_frame - rem)
            };
            padded as i32
        })
        .collect();

    let max_len = padded_len
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(0) as usize;

    let mut out = vec![0.0f32; batch_size * max_len];
    for b in 0..batch_size {
        let len_b = audio_len[b].max(0) as usize;
        let src = &audio[b * seq_len..b * seq_len + len_b];
        let dst = &mut out[b * max_len..b * max_len + len_b];
        dst.copy_from_slice(src);
    }

    Ok((out, padded_len, max_len))
}

