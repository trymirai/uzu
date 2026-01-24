use std::{path::PathBuf, rc::Rc};

use audioadapter_buffers::direct::SequentialSliceOfVecs;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use metal::Device;
use rubato::{FixedSync, Fft, Resampler};
use uzu::{audio_codec::NanoCodecModel, backends::metal::MTLContext};

pub fn handle_nanocodec_roundtrip(
    nemo_path: String,
    input_wav: String,
    output_wav: String,
) -> Result<(), String> {
    let (mut audio, sr_in) =
        read_wav_mono(PathBuf::from(input_wav))?;

    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    let context = MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))?;
    let context = Rc::new(context);

    let model = NanoCodecModel::load_from_nemo(
        context,
        &PathBuf::from(nemo_path),
    )
    .map_err(|e| format!("Failed to load NanoCodecModel: {e}"))?;

    let sr_out = model.sample_rate as u32;
    if sr_in != sr_out {
        audio = resample_mono(&audio, sr_in, sr_out)?;
    }

    let audio_len = vec![audio.len() as i32];
    let (recon, recon_len) =
        model
            .roundtrip(&audio, &audio_len, 1, audio.len())
            .map_err(|e| format!("Roundtrip failed: {e}"))?;

    let expected = recon_len[0] as usize;
    let recon = recon.into_iter().take(expected).collect::<Vec<_>>();

    // Compare reconstruction with input to assess correctness.
    print_roundtrip_comparison(&audio, &recon, sr_out as usize, model.samples_per_frame);

    write_wav_mono(
        PathBuf::from(output_wav),
        sr_out,
        &recon,
    )?;

    println!(
        "Encoded tokens_len={} frames, wrote {} samples at {} Hz",
        recon_len[0] / model.samples_per_frame as i32,
        expected,
        sr_out
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    len: usize,
    rms_in: f64,
    rms_out: f64,
    rms_err: f64,
    mse: f64,
    mae: f64,
    max_abs_err: f64,
    corr: f64,
    snr_db: f64,
}

fn compute_metrics(input: &[f32], output: &[f32]) -> Metrics {
    let n = input.len().min(output.len());
    if n == 0 {
        return Metrics {
            len: 0,
            rms_in: 0.0,
            rms_out: 0.0,
            rms_err: 0.0,
            mse: 0.0,
            mae: 0.0,
            max_abs_err: 0.0,
            corr: 0.0,
            snr_db: f64::NEG_INFINITY,
        };
    }

    let mut sum_in2 = 0.0f64;
    let mut sum_out2 = 0.0f64;
    let mut sum_err2 = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sum_xy = 0.0f64;

    for (&x, &y) in input[..n].iter().zip(output[..n].iter()) {
        let x = x as f64;
        let y = y as f64;
        let e = x - y;
        sum_in2 += x * x;
        sum_out2 += y * y;
        sum_err2 += e * e;
        sum_abs += e.abs();
        max_abs = max_abs.max(e.abs());
        sum_xy += x * y;
    }

    let inv_n = 1.0f64 / (n as f64);
    let rms_in = (sum_in2 * inv_n).sqrt();
    let rms_out = (sum_out2 * inv_n).sqrt();
    let rms_err = (sum_err2 * inv_n).sqrt();
    let mse = sum_err2 * inv_n;
    let mae = sum_abs * inv_n;
    let denom = (sum_in2 * sum_out2).sqrt();
    let corr = if denom > 0.0 { sum_xy / denom } else { 0.0 };
    let snr_db = if rms_err > 0.0 && rms_in > 0.0 {
        20.0 * (rms_in / rms_err).log10()
    } else if rms_err == 0.0 && rms_in > 0.0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };

    Metrics {
        len: n,
        rms_in,
        rms_out,
        rms_err,
        mse,
        mae,
        max_abs_err: max_abs,
        corr,
        snr_db,
    }
}

fn aligned_slices<'a>(
    input: &'a [f32],
    output: &'a [f32],
    offset: isize,
) -> (&'a [f32], &'a [f32]) {
    if offset >= 0 {
        let off = offset as usize;
        if off >= output.len() {
            return (&[], &[]);
        }
        let n = input.len().min(output.len() - off);
        (&input[..n], &output[off..off + n])
    } else {
        let off = (-offset) as usize;
        if off >= input.len() {
            return (&[], &[]);
        }
        let n = (input.len() - off).min(output.len());
        (&input[off..off + n], &output[..n])
    }
}

fn best_alignment_offset(
    input: &[f32],
    output: &[f32],
    sample_rate: usize,
) -> Option<(isize, f64)> {
    let max_shift = (sample_rate / 20).max(64).min(4096); // ~50ms, cap
    let base = input.len().min(output.len());
    let window_len = base.saturating_sub(max_shift).min(2 * sample_rate).max(0);
    if window_len == 0 {
        return None;
    }

    let mut best_off = 0isize;
    let mut best_corr = f64::NEG_INFINITY;

    for off in -(max_shift as isize)..=(max_shift as isize) {
        // Fixed-size window to avoid bias from varying overlap length.
        let (x, y) = if off >= 0 {
            let o = off as usize;
            (&input[..window_len], &output[o..o + window_len])
        } else {
            let o = (-off) as usize;
            (&input[o..o + window_len], &output[..window_len])
        };
        let m = compute_metrics(x, y);
        if m.corr > best_corr {
            best_corr = m.corr;
            best_off = off;
        }
    }
    Some((best_off, best_corr))
}

fn print_roundtrip_comparison(
    input: &[f32],
    output: &[f32],
    sample_rate: usize,
    samples_per_frame: usize,
) {
    let padded_len = output.len();
    let mut input_padded = input.to_vec();
    input_padded.resize(padded_len, 0.0);

    let m_valid = compute_metrics(input, output);
    let m_pad = compute_metrics(&input_padded, output);

    println!(
        "Compare: sr={}Hz, input_len={} (padded to {}={}, {} frames), output_len={}",
        sample_rate,
        input.len(),
        padded_len,
        if samples_per_frame > 0 {
            format!("{}*{}", padded_len / samples_per_frame, samples_per_frame)
        } else {
            "unknown".into()
        },
        if samples_per_frame > 0 {
            padded_len / samples_per_frame
        } else {
            0
        },
        output.len(),
    );
    println!(
        "No-align (valid {} samples): SNR={:.2} dB, corr={:.5}, rms_err={:.6}, MAE={:.6}, MSE={:.6}, max_abs_err={:.6}",
        m_valid.len,
        m_valid.snr_db,
        m_valid.corr,
        m_valid.rms_err,
        m_valid.mae,
        m_valid.mse,
        m_valid.max_abs_err
    );
    println!(
        "No-align (padded {} samples): SNR={:.2} dB, corr={:.5}, rms_err={:.6}",
        m_pad.len,
        m_pad.snr_db,
        m_pad.corr,
        m_pad.rms_err
    );

    if let Some((off, corr)) = best_alignment_offset(input, output, sample_rate) {
        if off != 0 {
            let (x, y) = aligned_slices(input, output, off);
            let m = compute_metrics(x, y);
            println!(
                "Aligned (best offset {off} samples, corr={:.5}, overlap {}): SNR={:.2} dB, rms_err={:.6}, MAE={:.6}, max_abs_err={:.6}",
                corr,
                m.len,
                m.snr_db,
                m.rms_err,
                m.mae,
                m.max_abs_err
            );
        }
    }
}

fn read_wav_mono(path: PathBuf) -> Result<(Vec<f32>, u32), String> {
    let mut reader =
        WavReader::open(&path).map_err(|e| format!("Failed to open wav: {e}"))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| format!("Failed to read sample: {e}")))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, bits) if bits <= 32 => {
            let denom = (1u64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|v| (v as f32) / denom)
                        .map_err(|e| format!("Failed to read sample: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        },
        other => {
            return Err(format!(
                "Unsupported wav format {:?} bits_per_sample={}",
                other.0, other.1
            ));
        },
    };

    if channels == 1 {
        return Ok((samples, sr));
    }

    if samples.len() % channels != 0 {
        return Err("Invalid wav: interleaved samples not divisible by channels".into());
    }

    let frames = samples.len() / channels;
    let mut mono = vec![0.0f32; frames];
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += samples[f * channels + c];
        }
        mono[f] = acc / (channels as f32);
    }
    Ok((mono, sr))
}

fn resample_mono(input: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>, String> {
    if sr_in == sr_out {
        return Ok(input.to_vec());
    }

    let mut resampler = Fft::<f32>::new(
        sr_in as usize,
        sr_out as usize,
        1024,
        1,
        1,
        FixedSync::Input,
    )
    .map_err(|e| format!("Failed to create resampler: {e}"))?;

    let input_len = input.len();
    let input_data = vec![input.to_vec()];
    let input_adapter = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| format!("Adapter error: {e}"))?;

    let output_len = resampler.process_all_needed_output_len(input_len);
    let mut output_data = vec![vec![0.0f32; output_len]];
    let mut output_adapter = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| format!("Adapter error: {e}"))?;

    let (_in_frames, out_frames) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, input_len, None)
        .map_err(|e| format!("Resample failed: {e}"))?;

    Ok(output_data[0][..out_frames].to_vec())
}

fn write_wav_mono(path: PathBuf, sample_rate: u32, audio: &[f32]) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(&path, spec).map_err(|e| format!("Failed to create wav: {e}"))?;

    for &x in audio.iter() {
        let x = if x.is_finite() { x } else { 0.0 };
        let x = x.clamp(-1.0, 1.0);
        let s = (x * i16::MAX as f32) as i16;
        writer
            .write_sample(s)
            .map_err(|e| format!("Failed to write sample: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("Finalize failed: {e}"))?;
    Ok(())
}

