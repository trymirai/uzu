use serde::Deserialize;

use super::ops::{
    CausalConv1dSpec, CausalConvTranspose1dSpec, HalfSnakeSpec, causal_conv_transpose1d_lalamo_reference,
    causal_conv1d_reference, half_snake_reference,
};
use crate::audio::{AudioError, AudioResult};

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor3Json {
    pub shape: [usize; 3],
    pub values: Vec<f32>,
}

impl Tensor3Json {
    fn validate_len(&self) -> AudioResult<()> {
        let expected = checked_product(&self.shape)?;
        if self.values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: self.values.len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalConv1dJson {
    pub weight: Tensor3Json,
    pub bias: Vec<f32>,
    #[serde(default = "default_dilation")]
    pub dilation: usize,
}

fn default_dilation() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalConvTranspose1dJson {
    pub weight: Tensor3Json,
    pub bias: Vec<f32>,
    pub stride: usize,
    pub groups: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodecUpsampleStageJson {
    pub activation_alpha: Vec<f32>,
    pub upsample_conv: CausalConvTranspose1dJson,
    #[serde(default)]
    pub res_layer: Option<NanoCodecHiFiGanResLayerJson>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodecResidualBlockJson {
    pub input_activation_alpha: Vec<f32>,
    pub input_conv: CausalConv1dJson,
    pub skip_activation_alpha: Vec<f32>,
    pub skip_conv: CausalConv1dJson,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodecHiFiGanResBlockJson {
    pub res_blocks: Vec<NanoCodecResidualBlockJson>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodecHiFiGanResLayerJson {
    pub res_blocks: Vec<NanoCodecHiFiGanResBlockJson>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodecDecoderJson {
    pub pre_conv: CausalConv1dJson,
    pub stages: Vec<NanoCodecUpsampleStageJson>,
    pub post_activation_alpha: Vec<f32>,
    pub post_conv: CausalConv1dJson,
    #[serde(default = "default_negative_slope")]
    pub negative_slope: f32,
    #[serde(default = "default_eps")]
    pub eps: f32,
}

fn default_negative_slope() -> f32 {
    0.01
}

fn default_eps() -> f32 {
    1e-9
}

#[derive(Debug, Clone, PartialEq)]
struct CausalConv1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct CausalConvTranspose1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct HalfSnakeLayer {
    alpha: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct UpsampleStage {
    activation: HalfSnakeLayer,
    upsample_conv: CausalConvTranspose1dLayer,
    res_layer: Option<HiFiGanResLayer>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResidualBlockLayer {
    input_activation: HalfSnakeLayer,
    input_conv: CausalConv1dLayer,
    skip_activation: HalfSnakeLayer,
    skip_conv: CausalConv1dLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct HiFiGanResBlock {
    res_blocks: Vec<ResidualBlockLayer>,
}

#[derive(Debug, Clone, PartialEq)]
struct HiFiGanResLayer {
    res_blocks: Vec<HiFiGanResBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedPaddedAudio {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
    pub lengths: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NanoCodecDecoderGraph {
    pre_conv: CausalConv1dLayer,
    stages: Vec<UpsampleStage>,
    post_activation: HalfSnakeLayer,
    post_conv: CausalConv1dLayer,
    negative_slope: f32,
    eps: f32,
    upsample_factor: usize,
}

impl NanoCodecDecoderGraph {
    pub fn upsample_factor(&self) -> usize {
        self.upsample_factor
    }

    pub fn output_channels(&self) -> usize {
        self.post_conv.cout
    }

    fn run_residual_block(
        &self,
        input: &[f32],
        block: &ResidualBlockLayer,
        lengths: &[i32],
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Vec<f32>> {
        let conv_input = half_snake_reference(HalfSnakeSpec {
            input,
            alpha: &block.input_activation.alpha,
            batch_size,
            channels,
            seq_len,
            snake_channels: block.input_activation.alpha.len(),
            negative_slope: self.negative_slope,
            eps: self.eps,
        })?;
        let skip_input = causal_conv1d_reference(CausalConv1dSpec {
            input: &conv_input,
            weight: &block.input_conv.weight,
            bias: &block.input_conv.bias,
            lengths,
            batch_size,
            cin: block.input_conv.cin,
            cout: block.input_conv.cout,
            seq_len,
            kernel_size: block.input_conv.kernel_size,
            dilation: block.input_conv.dilation,
        })?;
        let skip_input = half_snake_reference(HalfSnakeSpec {
            input: &skip_input,
            alpha: &block.skip_activation.alpha,
            batch_size,
            channels,
            seq_len,
            snake_channels: block.skip_activation.alpha.len(),
            negative_slope: self.negative_slope,
            eps: self.eps,
        })?;
        let residual = causal_conv1d_reference(CausalConv1dSpec {
            input: &skip_input,
            weight: &block.skip_conv.weight,
            bias: &block.skip_conv.bias,
            lengths,
            batch_size,
            cin: block.skip_conv.cin,
            cout: block.skip_conv.cout,
            seq_len,
            kernel_size: block.skip_conv.kernel_size,
            dilation: block.skip_conv.dilation,
        })?;
        if residual.len() != input.len() {
            return Err(AudioError::Runtime("residual block output shape mismatch".to_string()));
        }

        let mut output = input.to_vec();
        for (dst, res) in output.iter_mut().zip(residual.iter()) {
            *dst += *res;
        }
        Ok(output)
    }

    fn run_res_layer(
        &self,
        input: &[f32],
        layer: &HiFiGanResLayer,
        lengths: &[i32],
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Vec<f32>> {
        let mut accumulated = vec![0.0_f32; input.len()];
        for block in &layer.res_blocks {
            let mut block_output = input.to_vec();
            for residual_block in &block.res_blocks {
                block_output =
                    self.run_residual_block(&block_output, residual_block, lengths, batch_size, channels, seq_len)?;
            }
            if block_output.len() != accumulated.len() {
                return Err(AudioError::Runtime("residual layer output shape mismatch".to_string()));
            }
            for (dst, value) in accumulated.iter_mut().zip(block_output.iter()) {
                *dst += *value;
            }
        }

        let scale = 1.0 / layer.res_blocks.len() as f32;
        for value in &mut accumulated {
            *value *= scale;
        }
        Ok(accumulated)
    }

    pub fn decode_padded(
        &self,
        latent: &[f32],
        lengths: &[usize],
        batch_size: usize,
        channels: usize,
        frames: usize,
    ) -> AudioResult<DecodedPaddedAudio> {
        if self.pre_conv.cin != channels {
            return Err(AudioError::Runtime(format!(
                "decoder pre_conv expects {expected} channels, got {actual}",
                expected = self.pre_conv.cin,
                actual = channels
            )));
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }

        let mut lengths_i32 = Vec::with_capacity(lengths.len());
        for &length in lengths {
            if length > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length,
                    frames,
                });
            }
            lengths_i32
                .push(i32::try_from(length).map_err(|_| AudioError::Runtime("length exceeds i32 range".to_string()))?);
        }

        let mut x = causal_conv1d_reference(CausalConv1dSpec {
            input: latent,
            weight: &self.pre_conv.weight,
            bias: &self.pre_conv.bias,
            lengths: &lengths_i32,
            batch_size,
            cin: self.pre_conv.cin,
            cout: self.pre_conv.cout,
            seq_len: frames,
            kernel_size: self.pre_conv.kernel_size,
            dilation: self.pre_conv.dilation,
        })?;
        let mut current_channels = self.pre_conv.cout;
        let mut current_frames = frames;

        for stage in &self.stages {
            x = half_snake_reference(HalfSnakeSpec {
                input: &x,
                alpha: &stage.activation.alpha,
                batch_size,
                channels: current_channels,
                seq_len: current_frames,
                snake_channels: stage.activation.alpha.len(),
                negative_slope: self.negative_slope,
                eps: self.eps,
            })?;

            let next_frames = current_frames
                .checked_mul(stage.upsample_conv.stride)
                .ok_or(AudioError::Runtime("upsampled frame count overflow".to_string()))?;
            let next_lengths_i32: Vec<i32> = lengths_i32
                .iter()
                .copied()
                .map(|length| checked_mul_i32(length, stage.upsample_conv.stride))
                .collect::<AudioResult<Vec<_>>>()?;

            x = causal_conv_transpose1d_lalamo_reference(CausalConvTranspose1dSpec {
                input: &x,
                weight: &stage.upsample_conv.weight,
                bias: &stage.upsample_conv.bias,
                lengths: &next_lengths_i32,
                batch_size,
                cin: stage.upsample_conv.cin,
                cout: stage.upsample_conv.cout,
                seq_len_in: current_frames,
                seq_len_out: next_frames,
                stride: stage.upsample_conv.stride,
                groups: stage.upsample_conv.groups,
            })?;

            current_channels = stage.upsample_conv.cout;
            current_frames = next_frames;
            lengths_i32 = next_lengths_i32;

            if let Some(res_layer) = &stage.res_layer {
                x = self.run_res_layer(&x, res_layer, &lengths_i32, batch_size, current_channels, current_frames)?;
            }
        }

        x = half_snake_reference(HalfSnakeSpec {
            input: &x,
            alpha: &self.post_activation.alpha,
            batch_size,
            channels: current_channels,
            seq_len: current_frames,
            snake_channels: self.post_activation.alpha.len(),
            negative_slope: self.negative_slope,
            eps: self.eps,
        })?;

        x = causal_conv1d_reference(CausalConv1dSpec {
            input: &x,
            weight: &self.post_conv.weight,
            bias: &self.post_conv.bias,
            lengths: &lengths_i32,
            batch_size,
            cin: self.post_conv.cin,
            cout: self.post_conv.cout,
            seq_len: current_frames,
            kernel_size: self.post_conv.kernel_size,
            dilation: self.post_conv.dilation,
        })?;
        for value in &mut x {
            *value = value.tanh();
        }

        let mut output_lengths = Vec::with_capacity(lengths_i32.len());
        for &length in &lengths_i32 {
            if length < 0 {
                return Err(AudioError::Runtime("decoder produced negative length".to_string()));
            }
            output_lengths.push(length as usize);
        }

        Ok(DecodedPaddedAudio {
            samples: x,
            channels: self.post_conv.cout,
            frames: current_frames,
            lengths: output_lengths,
        })
    }
}

fn parse_causal_conv1d_layer(layer: CausalConv1dJson) -> AudioResult<CausalConv1dLayer> {
    layer.weight.validate_len()?;
    if layer.dilation == 0 {
        return Err(AudioError::Runtime("causal conv dilation must be > 0".to_string()));
    }

    let [cout, cin, kernel_size] = layer.weight.shape;
    if cin == 0 || cout == 0 || kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if layer.bias.len() != cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout,
            actual_tokens: layer.bias.len(),
        });
    }

    Ok(CausalConv1dLayer {
        weight: layer.weight.values,
        bias: layer.bias,
        cin,
        cout,
        kernel_size,
        dilation: layer.dilation,
    })
}

fn parse_causal_conv_transpose1d_layer(layer: CausalConvTranspose1dJson) -> AudioResult<CausalConvTranspose1dLayer> {
    layer.weight.validate_len()?;
    if layer.stride == 0 || layer.groups == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let [cin, cout_per_group, kernel_size] = layer.weight.shape;
    if cin == 0 || cout_per_group == 0 || kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let cout = layer.bias.len();
    if cout == 0 || cout % layer.groups != 0 || cin % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cout_per_group != cout / layer.groups {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout / layer.groups,
            actual_tokens: cout_per_group,
        });
    }

    Ok(CausalConvTranspose1dLayer {
        weight: layer.weight.values,
        bias: layer.bias,
        cin,
        cout,
        kernel_size,
        stride: layer.stride,
        groups: layer.groups,
    })
}

fn parse_half_snake_layer(
    alpha: Vec<f32>,
    channels: usize,
    context: &str,
) -> AudioResult<HalfSnakeLayer> {
    if channels == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let snake_channels = channels / 2;
    if alpha.len() != snake_channels {
        return Err(AudioError::Runtime(format!(
            "{context} alpha length mismatch: expected {snake_channels}, got {}",
            alpha.len()
        )));
    }
    Ok(HalfSnakeLayer {
        alpha,
    })
}

fn parse_residual_block_layer(
    block: NanoCodecResidualBlockJson,
    channels: usize,
) -> AudioResult<ResidualBlockLayer> {
    let input_activation = parse_half_snake_layer(block.input_activation_alpha, channels, "residual input activation")?;
    let input_conv = parse_causal_conv1d_layer(block.input_conv)?;
    if input_conv.cin != channels || input_conv.cout != channels {
        return Err(AudioError::Runtime(format!(
            "residual input_conv channel mismatch: expected {channels}->{channels}, got {}->{}",
            input_conv.cin, input_conv.cout
        )));
    }

    let skip_activation = parse_half_snake_layer(block.skip_activation_alpha, channels, "residual skip activation")?;
    let skip_conv = parse_causal_conv1d_layer(block.skip_conv)?;
    if skip_conv.cin != channels || skip_conv.cout != channels {
        return Err(AudioError::Runtime(format!(
            "residual skip_conv channel mismatch: expected {channels}->{channels}, got {}->{}",
            skip_conv.cin, skip_conv.cout
        )));
    }

    Ok(ResidualBlockLayer {
        input_activation,
        input_conv,
        skip_activation,
        skip_conv,
    })
}

fn parse_res_block(
    block: NanoCodecHiFiGanResBlockJson,
    channels: usize,
) -> AudioResult<HiFiGanResBlock> {
    if block.res_blocks.is_empty() {
        return Err(AudioError::Runtime("residual block list must be non-empty".to_string()));
    }

    let mut parsed = Vec::with_capacity(block.res_blocks.len());
    for residual in block.res_blocks {
        parsed.push(parse_residual_block_layer(residual, channels)?);
    }
    Ok(HiFiGanResBlock {
        res_blocks: parsed,
    })
}

fn parse_res_layer(
    layer: NanoCodecHiFiGanResLayerJson,
    channels: usize,
) -> AudioResult<HiFiGanResLayer> {
    if layer.res_blocks.is_empty() {
        return Err(AudioError::Runtime("residual layer must contain at least one block".to_string()));
    }

    let mut parsed = Vec::with_capacity(layer.res_blocks.len());
    for block in layer.res_blocks {
        parsed.push(parse_res_block(block, channels)?);
    }

    Ok(HiFiGanResLayer {
        res_blocks: parsed,
    })
}

impl TryFrom<NanoCodecDecoderJson> for NanoCodecDecoderGraph {
    type Error = AudioError;

    fn try_from(json: NanoCodecDecoderJson) -> Result<Self, Self::Error> {
        if !json.negative_slope.is_finite() || !json.eps.is_finite() || json.eps <= 0.0 {
            return Err(AudioError::Runtime(
                "decoder negative_slope/eps must be finite and eps must be > 0".to_string(),
            ));
        }

        let pre_conv = parse_causal_conv1d_layer(json.pre_conv)?;
        let mut current_channels = pre_conv.cout;
        let mut upsample_factor = 1usize;

        let mut stages = Vec::with_capacity(json.stages.len());
        for stage in json.stages {
            let activation = parse_half_snake_layer(stage.activation_alpha, current_channels, "stage activation")?;
            let upsample_conv = parse_causal_conv_transpose1d_layer(stage.upsample_conv)?;
            if upsample_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "stage upsample input channels mismatch: expected {current_channels}, got {}",
                    upsample_conv.cin
                )));
            }
            current_channels = upsample_conv.cout;
            upsample_factor = upsample_factor
                .checked_mul(upsample_conv.stride)
                .ok_or(AudioError::Runtime("decoder upsample factor overflow".to_string()))?;
            let res_layer = stage.res_layer.map(|layer| parse_res_layer(layer, current_channels)).transpose()?;

            stages.push(UpsampleStage {
                activation,
                upsample_conv,
                res_layer,
            });
        }

        let post_activation = parse_half_snake_layer(json.post_activation_alpha, current_channels, "post activation")?;
        let post_conv = parse_causal_conv1d_layer(json.post_conv)?;
        if post_conv.cin != current_channels {
            return Err(AudioError::Runtime(format!(
                "post_conv input channels mismatch: expected {current_channels}, got {}",
                post_conv.cin
            )));
        }

        Ok(Self {
            pre_conv,
            stages,
            post_activation,
            post_conv,
            negative_slope: json.negative_slope,
            eps: json.eps,
            upsample_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{NanoCodecDecoderGraph, NanoCodecDecoderJson};
    use crate::audio::AudioError;

    fn decoder_json() -> serde_json::Value {
        serde_json::json!({
            "negative_slope": 0.01,
            "eps": 1e-9,
            "pre_conv": {
                "weight": {
                    "shape": [2, 2, 1],
                    "values": [1.0, 0.0, 0.0, 1.0]
                },
                "bias": [0.0, 0.0]
            },
            "stages": [
                {
                    "activation_alpha": [1.0],
                    "upsample_conv": {
                        "weight": {
                            "shape": [2, 1, 4],
                            "values": [
                                1.0, 0.0, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0
                            ]
                        },
                        "bias": [0.0, 0.0],
                        "stride": 2,
                        "groups": 2
                    }
                }
            ],
            "post_activation_alpha": [1.0],
            "post_conv": {
                "weight": {
                    "shape": [1, 2, 1],
                    "values": [1.0, 0.0]
                },
                "bias": [0.0]
            }
        })
    }

    fn parse_graph(value: serde_json::Value) -> NanoCodecDecoderGraph {
        let parsed: NanoCodecDecoderJson = serde_json::from_value(value).expect("json");
        NanoCodecDecoderGraph::try_from(parsed).expect("graph")
    }

    #[test]
    fn decoder_graph_parses_and_reports_upsample_factor() {
        let graph = parse_graph(decoder_json());
        assert_eq!(graph.upsample_factor(), 2);
        assert_eq!(graph.output_channels(), 1);
    }

    #[test]
    fn decoder_graph_decode_scales_lengths() {
        let graph = parse_graph(decoder_json());

        let batch_size = 1usize;
        let channels = 2usize;
        let frames = 3usize;
        let latent = vec![
            0.2, 0.1, -0.2, // ch0
            0.0, 0.0, 0.0, // ch1
        ];
        let result = graph.decode_padded(&latent, &[2usize], batch_size, channels, frames).expect("decode");

        assert_eq!(result.channels, 1);
        assert_eq!(result.frames, 6);
        assert_eq!(result.lengths, vec![4usize]);
        assert_eq!(result.samples.len(), batch_size * result.channels * result.frames);
    }

    #[test]
    fn decoder_graph_rejects_residual_channel_mismatch() {
        let mut json = decoder_json();
        json["stages"][0]["res_layer"] = serde_json::json!({
            "res_blocks": [{
                "res_blocks": [{
                    "input_activation_alpha": [1.0],
                    "input_conv": {
                        "weight": {
                            "shape": [2, 1, 1],
                            "values": [1.0, 0.0]
                        },
                        "bias": [0.0, 0.0]
                    },
                    "skip_activation_alpha": [1.0],
                    "skip_conv": {
                        "weight": {
                            "shape": [2, 2, 1],
                            "values": [1.0, 0.0, 0.0, 1.0]
                        },
                        "bias": [0.0, 0.0]
                    }
                }]
            }]
        });

        let parsed: NanoCodecDecoderJson = serde_json::from_value(json).expect("json");
        let error = NanoCodecDecoderGraph::try_from(parsed).expect_err("must reject mismatch");
        match error {
            AudioError::Runtime(message) => assert!(message.contains("residual input_conv channel mismatch")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decoder_graph_residual_layer_changes_output() {
        let without_residual = parse_graph(decoder_json());
        let mut with_residual_json = decoder_json();
        with_residual_json["stages"][0]["res_layer"] = serde_json::json!({
            "res_blocks": [{
                "res_blocks": [{
                    "input_activation_alpha": [1.0],
                    "input_conv": {
                        "weight": {
                            "shape": [2, 2, 1],
                            "values": [0.0, 0.0, 0.0, 0.0]
                        },
                        "bias": [0.0, 0.0]
                    },
                    "skip_activation_alpha": [1.0],
                    "skip_conv": {
                        "weight": {
                            "shape": [2, 2, 1],
                            "values": [0.0, 0.0, 0.0, 0.0]
                        },
                        "bias": [0.5, 0.0]
                    }
                }]
            }]
        });
        let with_residual = parse_graph(with_residual_json);

        let batch_size = 1usize;
        let channels = 2usize;
        let frames = 3usize;
        let lengths = [2usize];
        let latent = vec![
            0.2, 0.1, -0.2, // ch0
            0.0, 0.0, 0.0, // ch1
        ];

        let baseline = without_residual.decode_padded(&latent, &lengths, batch_size, channels, frames).expect("decode");
        let with_residual =
            with_residual.decode_padded(&latent, &lengths, batch_size, channels, frames).expect("decode");

        assert_eq!(with_residual.channels, baseline.channels);
        assert_eq!(with_residual.frames, baseline.frames);
        assert_eq!(with_residual.lengths, baseline.lengths);

        let max_delta = with_residual
            .samples
            .iter()
            .zip(baseline.samples.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_delta > 1e-5, "residual layer should alter decoded waveform");
    }
}
