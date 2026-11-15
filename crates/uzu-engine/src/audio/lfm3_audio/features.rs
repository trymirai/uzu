//! Log-mel front-end, conv downsampler and projector of the LFM3-Audio speech
//! encoder (CPU f32, bf16 boundary rounding as the reference runs bf16).

use anyhow::{ensure, Context, Result};

use crate::audio::liquid_codec::{bf16_round, round_bf16};
use crate::vision::tower::{gelu_erf, matmul_nt, LinearWeight, SafetensorLoader};

// ---------------------------------------------------------------------------
// Log-mel
// ---------------------------------------------------------------------------

/// Slaney mel filter bank, `transformers.audio_utils.mel_filter_bank` /
/// librosa semantics, computed in f64 like `mlx_audio.mel_filters(precise=True)`.
pub struct MelFilterBank {
    /// `[num_mel, num_freq_bins]` row-major.
    pub filters: Vec<f32>,
    pub num_mel: usize,
    pub num_freq_bins: usize,
}

impl MelFilterBank {
    pub fn slaney(sample_rate: usize, n_fft: usize, num_mel: usize) -> Self {
        let num_freq_bins = n_fft / 2 + 1;
        let f_min = 0.0f64;
        let f_max = sample_rate as f64 / 2.0;
        let f_sp = 200.0 / 3.0;

        let hz_to_mel = |hz: f64| -> f64 {
            let min_log_hz = 1000.0;
            if hz >= min_log_hz {
                (min_log_hz / f_sp) + (hz / min_log_hz).ln() / (6.4f64.ln() / 27.0)
            } else {
                hz / f_sp
            }
        };
        let mel_to_hz = |mel: f64| -> f64 {
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            if mel >= min_log_mel {
                min_log_hz * ((6.4f64.ln() / 27.0) * (mel - min_log_mel)).exp()
            } else {
                f_sp * mel
            }
        };

        let mel_low = hz_to_mel(f_min);
        let mel_high = hz_to_mel(f_max);
        let mel_points: Vec<f64> = (0..num_mel + 2)
            .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (num_mel + 1) as f64)
            .collect();
        let hz_points: Vec<f64> = mel_points.iter().copied().map(mel_to_hz).collect();

        let mut filters = vec![0.0f32; num_mel * num_freq_bins];
        for i in 0..num_mel {
            let (left, center, right) = (hz_points[i], hz_points[i + 1], hz_points[i + 2]);
            let fdiff_center = center - left;
            let fdiff_right = right - center;
            for (bin, freq) in (0..num_freq_bins)
                .map(|b| b as f64 * f_max / (num_freq_bins - 1) as f64)
                .enumerate()
            {
                let weight = if freq <= left || freq >= right {
                    0.0
                } else if freq <= center {
                    (freq - left) / fdiff_center
                } else {
                    (right - freq) / fdiff_right
                };
                filters[i * num_freq_bins + bin] = weight as f32;
            }
            // Slaney norm.
            let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
            for bin in 0..num_freq_bins {
                filters[i * num_freq_bins + bin] *= enorm as f32;
            }
        }
        Self {
            filters,
            num_mel,
            num_freq_bins,
        }
    }

    /// Mono 16 kHz clip -> normalized log-mel `(num_mel, num_frames)` row-major.
    pub fn log_mel(
        &self,
        waveform: &[f32],
        n_fft: usize,
        hop_length: usize,
        num_frames: usize,
        log_floor: f32,
        norm_offset: f32,
        norm_divisor: f32,
    ) -> Vec<f32> {
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n_fft as f64).cos())
            .map(|w| w as f32)
            .collect();

        let mut mel = vec![0.0f32; self.num_mel * num_frames];
        let mut frame = vec![0.0f32; n_fft];
        for frame_index in 0..num_frames {
            let offset = frame_index * hop_length;
            for k in 0..n_fft {
                let sample = waveform.get(offset + k).copied().unwrap_or(0.0);
                frame[k] = sample * window[k];
            }
            let (re, im) = crate::audio::fft::rfft(&frame);
            let mut power = vec![0.0f32; self.num_freq_bins];
            for bin in 0..self.num_freq_bins {
                power[bin] = re[bin] * re[bin] + im[bin] * im[bin];
            }
            for m in 0..self.num_mel {
                let mut sum = 0.0f32;
                for bin in 0..self.num_freq_bins {
                    sum += self.filters[m * self.num_freq_bins + bin] * power[bin];
                }
                let logged = sum.max(1e-30).log10().max(log_floor);
                mel[m * num_frames + frame_index] = (logged + norm_offset) / norm_divisor;
            }
        }
        mel
    }
}

// ---------------------------------------------------------------------------
// Downsampler
// ---------------------------------------------------------------------------

/// Conv2d weight stored `[out, kh, kw, in]` (MLX layout, as shipped).
struct Conv2dWeights {
    kernel: Vec<f32>,
    bias: Vec<f32>,
    out_ch: usize,
    kh: usize,
    kw: usize,
    depthwise: bool,
}

impl Conv2dWeights {
    #[allow(dead_code)]
    fn load(loader: &mut SafetensorLoader, base: &str, depthwise: bool) -> Result<Self> {
        let (kernel, shape) = loader
            .read_tensor_f32(&format!("{base}.weights"))
            .with_context(|| format!("Reading {base}.weights"))?;
        ensure!(shape.len() == 4, "expected 4D conv weight for {base}, got {shape:?}");
        let (bias, _) = loader.read_tensor_f32(&format!("{base}.biases"))?;
        Ok(Self {
            kernel,
            bias,
            out_ch: shape[0],
            kh: shape[1],
            kw: shape[2],
            depthwise,
        })
    }
}

/// Three stride-2 3x3 convs (two depthwise, gated by pointwise convs) over the
/// (mel, time) plane: 8 frames of 128 bins become one 768-vector.
pub struct Downsampler {
    conv_in: Conv2dWeights,
    conv_mid: Conv2dWeights,
    lin_mid: Conv2dWeights,
    conv_end: Conv2dWeights,
    lin_end: Conv2dWeights,
    lin_out: LinearWeight,
    channels: usize,
    hidden: usize,
}

impl Downsampler {
    #[allow(dead_code)]
    pub(crate) fn load(loader: &mut SafetensorLoader, num_mel_bins: usize, channels: usize, hidden: usize) -> Result<Self> {
        let conv_in = Conv2dWeights::load(loader, "audio.downsampler.conv_in", false)?;
        let conv_mid = Conv2dWeights::load(loader, "audio.downsampler.conv_mid", true)?;
        let lin_mid = Conv2dWeights::load(loader, "audio.downsampler.lin_mid", false)?;
        let conv_end = Conv2dWeights::load(loader, "audio.downsampler.conv_end", true)?;
        let lin_end = Conv2dWeights::load(loader, "audio.downsampler.lin_end", false)?;
        let (lin_out_w, shape) = loader.read_tensor_f32("audio.downsampler.lin_out.weights")?;
        ensure!(shape.len() == 2, "lin_out weight must be 2D, got {shape:?}");
        let (lin_out_b, _) = loader.read_tensor_f32("audio.downsampler.lin_out.biases")?;
        let lin_out = LinearWeight {
            weight: lin_out_w,
            bias: Some(lin_out_b),
            in_features: shape[1],
            out_features: shape[0],
        };
        ensure!(lin_out.in_features == num_mel_bins * channels / 8);
        Ok(Self {
            conv_in,
            conv_mid,
            lin_mid,
            conv_end,
            lin_end,
            lin_out,
            channels,
            hidden,
        })
    }

    /// `(num_mel, num_frames)` features and valid frame count -> downsampled
    /// rows `[tokens, hidden]` (bf16 values in f32), one per 8 frames.
    pub fn forward(&self, features: &[f32], num_mel: usize, num_frames: usize) -> Result<Vec<f32>> {
        let _ = self.hidden;
        let channels = self.channels;

        // Stage 0: (1, mel, frames, 1).
        let mut x = vec![0.0f32; num_mel * num_frames];
        x.copy_from_slice(&features[..num_mel * num_frames]);
        let mut lengths = num_frames;

        // conv_in: stride 2, pad 1, full in-channels.
        let (out_t1, y1) = self.conv2d(&x, num_mel, num_frames, 1, &self.conv_in, None, lengths)?;
        lengths = lengths.div_ceil(2);
        let (out_f2, y2) = self.conv2d(&y1, out_t1.0, out_t1.1, channels, &self.conv_mid, Some(&self.lin_mid), lengths)?;
        lengths = lengths.div_ceil(2);
        let (out_f3, y3) = self.conv2d(&y2, out_f2.0, out_f2.1, channels, &self.conv_end, Some(&self.lin_end), lengths)?;
        lengths = lengths.div_ceil(2);

        // Flatten (channel, mel) channel-major and project.
        let (out_f, out_t, out_c) = (out_f3.0, out_f3.1, channels);
        let tokens = lengths;
        let mut flat = vec![0.0f32; tokens * (out_c * out_f)];
        for t in 0..tokens {
            for c in 0..out_c {
                for f in 0..out_f {
                    flat[t * out_c * out_f + c * out_f + f] = y3[(f * out_t + t) * out_c + c];
                }
            }
        }
        let mut out = vec![0.0f32; tokens * self.lin_out.out_features];
        matmul_nt(
            &flat,
            tokens,
            out_c * out_f,
            &self.lin_out.weight,
            self.lin_out.out_features,
            self.lin_out.bias.as_deref(),
            &mut out,
        );
        round_bf16(&mut out);
        Ok(out)
    }

    /// NHWC (F, T, C) -> (F', T', C) stride-2 pad-1 3x3 (or 1x1) conv with
    /// valid-length masking, SiLU, and an optional pointwise follow-up.
    fn conv2d(
        &self,
        x: &[f32],
        in_f: usize,
        in_t: usize,
        in_c: usize,
        w: &Conv2dWeights,
        pointwise: Option<&Conv2dWeights>,
        valid_length: usize,
    ) -> Result<((usize, usize), Vec<f32>)> {
        let out_f = in_f.div_ceil(2);
        let out_t = in_t.div_ceil(2);
        let out_c = w.out_ch;
        let kernel_len = w.kh * w.kw * in_c;
        let mut y = vec![0.0f32; out_f * out_t * out_c];
        for f in 0..out_f {
            for t in 0..out_t {
                for oc in 0..out_c {
                    let mut sum = w.bias[oc];
                    if w.depthwise {
                        // in_ch == 1: groups == out_ch, channel = oc.
                        for kh in 0..w.kh {
                            let Some(sf) = (f * 2 + kh).checked_sub(1) else {
                                continue;
                            };
                            if sf >= in_f {
                                continue;
                            }
                            for kw in 0..w.kw {
                                let Some(st) = (t * 2 + kw).checked_sub(1) else {
                                    continue;
                                };
                                if st >= in_t {
                                    continue;
                                }
                                let value = x[(sf * in_t + st) * in_c + oc];
                                let weight = w.kernel[(oc * w.kh + kh) * w.kw + kw];
                                sum += value * weight;
                            }
                        }
                    } else {
                        let row = &w.kernel[oc * kernel_len..(oc + 1) * kernel_len];
                        for kh in 0..w.kh {
                            let Some(sf) = (f * 2 + kh).checked_sub(1) else {
                                continue;
                            };
                            if sf >= in_f {
                                continue;
                            }
                            for kw in 0..w.kw {
                                let Some(st) = (t * 2 + kw).checked_sub(1) else {
                                    continue;
                                };
                                if st >= in_t {
                                    continue;
                                }
                                let base = (sf * in_t + st) * in_c;
                                let weight_row = &row[(kh * w.kw + kw) * in_c..(kh * w.kw + kw + 1) * in_c];
                                for ic in 0..in_c {
                                    sum += x[base + ic] * weight_row[ic];
                                }
                            }
                        }
                    }
                    y[(f * out_t + t) * out_c + oc] = bf16_round(sum);
                }
            }
        }
        let mut out = if let Some(pointwise) = pointwise {
            // Pointwise 1x1 conv = per-token matmul over channels.
            let pw_c = pointwise.out_ch;
            let mut z = vec![0.0f32; out_f * out_t * pw_c];
            for f in 0..out_f {
                for t in 0..out_t {
                    let input = &y[(f * out_t + t) * out_c..(f * out_t + t + 1) * out_c];
                    let output = &mut z[(f * out_t + t) * pw_c..(f * out_t + t + 1) * pw_c];
                    matmul_nt(input, 1, out_c, &pointwise.kernel, pw_c, Some(&pointwise.bias), output);
                }
            }
            z
        } else {
            y
        };
        let final_c = pointwise.map_or(out_c, |p| p.out_ch);
        let out_valid = valid_length.div_ceil(2);
        // SiLU with bf16 rounding, then valid-length masking.
        for (index, value) in out.iter_mut().enumerate() {
            let t = (index / final_c) % out_t;
            *value = if t < out_valid { bf16_round(*value / (1.0 + (-*value).exp())) } else { 0.0 };
        }
        Ok(((out_f, out_t), out))
    }
}

// ---------------------------------------------------------------------------
// Projector
// ---------------------------------------------------------------------------

/// LayerNorm -> Linear -> erf-GELU -> Linear (768 -> 2048 -> 2048).
pub struct Projector {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    linear_1: LinearWeight,
    linear_2: LinearWeight,
    hidden: usize,
    epsilon: f32,
}

impl Projector {
    #[allow(dead_code)]
    pub(crate) fn load(loader: &mut SafetensorLoader, hidden: usize) -> Result<Self> {
        let (gamma, _) = loader.read_tensor_f32("audio.projector.layer_norm.scales")?;
        let (beta, _) = loader.read_tensor_f32("audio.projector.layer_norm.biases")?;
        let (w1, shape1) = loader.read_tensor_f32("audio.projector.linear_1.weights")?;
        let (b1, _) = loader.read_tensor_f32("audio.projector.linear_1.biases")?;
        let (w2, shape2) = loader.read_tensor_f32("audio.projector.linear_2.weights")?;
        let (b2, _) = loader.read_tensor_f32("audio.projector.linear_2.biases")?;
        ensure!(shape1 == [hidden * 8 / 3, hidden] || shape1[1] == hidden, "linear_1 {shape1:?}");
        Ok(Self {
            gamma,
            beta,
            linear_1: LinearWeight { weight: w1, bias: Some(b1), in_features: shape1[1], out_features: shape1[0] },
            linear_2: LinearWeight { weight: w2, bias: Some(b2), in_features: shape2[1], out_features: shape2[0] },
            hidden,
            epsilon: 1e-5,
        })
    }

    pub fn forward(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        let hidden = self.hidden;
        let mut out = vec![0.0f32; tokens * hidden];
        for token in 0..tokens {
            let row = &x[token * hidden..(token + 1) * hidden];
            let mean = row.iter().sum::<f32>() / hidden as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (var + self.epsilon).sqrt();
            for (d, value) in row.iter().enumerate() {
                out[token * hidden + d] = bf16_round((value - mean) * inv_std * self.gamma[d] + self.beta[d]);
            }
        }
        let mut mid = vec![0.0f32; tokens * self.linear_1.out_features];
        matmul_nt(&out, tokens, hidden, &self.linear_1.weight, self.linear_1.out_features, self.linear_1.bias.as_deref(), &mut mid);
        for value in mid.iter_mut() {
            *value = bf16_round(gelu_erf(*value));
        }
        let mut final_out = vec![0.0f32; tokens * self.linear_2.out_features];
        matmul_nt(
            &mid,
            tokens,
            self.linear_2.in_features,
            &self.linear_2.weight,
            self.linear_2.out_features,
            self.linear_2.bias.as_deref(),
            &mut final_out,
        );
        round_bf16(&mut final_out);
        final_out
    }
}
