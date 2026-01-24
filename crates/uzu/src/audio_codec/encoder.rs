use std::{fs::File, path::Path, rc::Rc};

use crate::{
    Array,
    DataType,
    DeviceContext,
    backends::metal::MTLContext,
    backends::metal::KernelDataType,
    backends::metal::kernel::{
        AddKernel, Conv1dArguments, Conv1dKernel, LeakyReluKernel, ScaleKernel,
    },
    parameters::ParameterLoader,
};

use super::{Conv1dWeights, NanoCodecError, max_index_with_prefix};

#[derive(Debug, Clone)]
struct EncoderResidualBlockWeights {
    input_conv: Conv1dWeights,
    skip_conv: Conv1dWeights,
    dilation: usize,
}

#[derive(Debug, Clone)]
struct EncoderResBranchWeights {
    blocks: Box<[EncoderResidualBlockWeights]>,
}

#[derive(Debug, Clone)]
struct EncoderResLayerWeights {
    branches: Box<[EncoderResBranchWeights]>,
}

#[derive(Debug, Clone)]
struct DownsampleLayerWeights {
    res_layer: EncoderResLayerWeights,
    down_conv: Conv1dWeights,
    stride: usize,
}

#[derive(Debug, Clone)]
struct EncoderWeights {
    pre_conv: Conv1dWeights,
    down_layers: Box<[DownsampleLayerWeights]>,
    post_conv: Conv1dWeights,
    base_channels: usize,
    encoded_dim: usize,
}

impl EncoderWeights {
    fn load(
        context: &Rc<MTLContext>,
        encoder_weights_path: &Path,
    ) -> Result<Self, NanoCodecError> {
        let file = File::open(encoder_weights_path)?;
        let loader = ParameterLoader::new(&file, context).map_err(|e| {
            NanoCodecError::InvalidWeights(format!(
                "Failed to read safetensors header for encoder weights: {e}"
            ))
        })?;
        let keys: Vec<String> = loader.keys().cloned().collect();
        let tree = loader.tree();

        let pre_conv = Conv1dWeights::load(
            context,
            &tree,
            "pre_conv.conv",
            "nanocodec_enc_pre_conv",
        )?;
        let post_conv = Conv1dWeights::load(
            context,
            &tree,
            "post_conv.conv",
            "nanocodec_enc_post_conv",
        )?;

        let base_channels = pre_conv.cout;
        let encoded_dim = post_conv.cout;

        // Discover number of downsample layers.
        let max_down = max_index_with_prefix(
            keys.clone().into_iter(),
            "down_sample_conv_layers.",
        )
        .ok_or_else(|| NanoCodecError::InvalidWeights("No down_sample_conv_layers.* found".into()))?;
        let num_down_layers = max_down + 1;

        // Discover res layer structure from layer 0.
        let max_branch = max_index_with_prefix(
            keys.clone().into_iter(),
            "res_layers.0.res_blocks.",
        )
        .ok_or_else(|| NanoCodecError::InvalidWeights("No res_layers.0.res_blocks.* found".into()))?;
        let num_branches = max_branch + 1;

        let max_dilation = max_index_with_prefix(
            keys.clone().into_iter(),
            "res_layers.0.res_blocks.0.res_blocks.",
        )
        .ok_or_else(|| NanoCodecError::InvalidWeights("No res_layers.0.res_blocks.0.res_blocks.* found".into()))?;
        let num_dilations = max_dilation + 1;

        // NeMo default for HiFiGAN is (1, 3, 5). We require this for now.
        let dilation_sizes: Box<[usize]> = match num_dilations {
            3 => Box::new([1, 3, 5]),
            other => {
                return Err(NanoCodecError::InvalidWeights(format!(
                    "Unsupported number of dilation blocks {other} (expected 3)"
                )));
            },
        };

        let mut down_layers = Vec::with_capacity(num_down_layers);
        for layer_idx in 0..num_down_layers {
            let down_conv = Conv1dWeights::load(
                context,
                &tree,
                &format!("down_sample_conv_layers.{layer_idx}.conv"),
                &format!("nanocodec_enc_down_conv_{layer_idx}"),
            )?;

            // In HiFiGANEncoder, downsample conv uses kernel_size = 2 * stride.
            if down_conv.kernel_size % 2 != 0 {
                return Err(NanoCodecError::InvalidWeights(format!(
                    "down_sample_conv_layers.{layer_idx}: expected even kernel size, got {}",
                    down_conv.kernel_size
                )));
            }
            let stride = down_conv.kernel_size / 2;
            if stride == 0 {
                return Err(NanoCodecError::InvalidWeights(format!(
                    "down_sample_conv_layers.{layer_idx}: inferred stride=0"
                )));
            }

            // Load res_layer weights for this stage.
            let mut branches = Vec::with_capacity(num_branches);
            for branch_idx in 0..num_branches {
                let mut blocks = Vec::with_capacity(num_dilations);
                for dilation_idx in 0..num_dilations {
                    let dilation = dilation_sizes[dilation_idx];
                    let base_prefix = format!(
                        "res_layers.{layer_idx}.res_blocks.{branch_idx}.res_blocks.{dilation_idx}"
                    );
                    let input_conv = Conv1dWeights::load(
                        context,
                        &tree,
                        &format!("{base_prefix}.input_conv.conv"),
                        &format!(
                            "nanocodec_enc_res_in_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;
                    let skip_conv = Conv1dWeights::load(
                        context,
                        &tree,
                        &format!("{base_prefix}.skip_conv.conv"),
                        &format!(
                            "nanocodec_enc_res_skip_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;
                    blocks.push(EncoderResidualBlockWeights {
                        input_conv,
                        skip_conv,
                        dilation,
                    });
                }
                branches.push(EncoderResBranchWeights {
                    blocks: blocks.into_boxed_slice(),
                });
            }

            down_layers.push(DownsampleLayerWeights {
                res_layer: EncoderResLayerWeights {
                    branches: branches.into_boxed_slice(),
                },
                down_conv,
                stride,
            });
        }

        Ok(Self {
            pre_conv,
            down_layers: down_layers.into_boxed_slice(),
            post_conv,
            base_channels,
            encoded_dim,
        })
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    // NeMo get_padding(kernel_size, dilation) = (kernel_size*dilation - dilation) // 2
    (kernel_size * dilation - dilation) / 2
}

fn get_down_sample_padding(kernel_size: usize, stride: usize) -> usize {
    // NeMo get_down_sample_padding(kernel_size, stride) = (kernel_size - stride + 1) // 2
    (kernel_size - stride + 1) / 2
}

fn conv1d_out_len(
    seq_len_in: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> usize {
    // torch.nn.Conv1d output length:
    // floor((L + 2*padding - dilation*(K-1) - 1)/stride + 1)
    ((seq_len_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)
        + 1
}

pub struct NanoCodecEncoder {
    context: Rc<MTLContext>,
    weights: EncoderWeights,
    down_sample_rates: Box<[usize]>,

    // kernels
    conv1d: Conv1dKernel,
    lrelu: LeakyReluKernel,
    add: AddKernel,
    scale: ScaleKernel,

    negative_slope: f32,
    pad_mode: usize,
}

impl NanoCodecEncoder {
    pub fn load_from_export_dir(
        context: Rc<MTLContext>,
        export_dir: &Path,
        down_sample_rates: Box<[usize]>,
    ) -> Result<Self, NanoCodecError> {
        let encoder_weights_path = export_dir.join("audio_encoder.safetensors");
        let weights = EncoderWeights::load(&context, &encoder_weights_path)?;

        if down_sample_rates.len() != weights.down_layers.len() {
            return Err(NanoCodecError::InvalidWeights(format!(
                "Config down_sample_rates len {} does not match weights down_sample_conv_layers len {}",
                down_sample_rates.len(),
                weights.down_layers.len()
            )));
        }

        let conv1d =
            Conv1dKernel::new(&context, KernelDataType::Float32)?;
        let lrelu =
            LeakyReluKernel::new(&context, KernelDataType::Float32)?;
        let add = AddKernel::new(&context, KernelDataType::Float32)?;
        let scale = ScaleKernel::new(&context, KernelDataType::Float32)?;

        Ok(Self {
            context,
            weights,
            down_sample_rates,
            conv1d,
            lrelu,
            add,
            scale,
            negative_slope: 0.01,
            pad_mode: 1, // replicate
        })
    }

    pub fn encode_latents(
        &self,
        audio: &[f32],
        audio_len: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(crate::backends::metal::MetalArray, Vec<i32>), NanoCodecError> {
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

        // Stage lengths (in samples): divide by stride per downsample stage.
        let mut stage_lengths: Vec<Vec<i32>> = Vec::with_capacity(self.down_sample_rates.len() + 1);
        stage_lengths.push(audio_len.to_vec());
        for &rate in self.down_sample_rates.iter() {
            let prev = stage_lengths.last().expect("exists");
            stage_lengths.push(prev.iter().map(|l| l / (rate as i32)).collect());
        }

        // Allocate lengths arrays on device.
        let mut length_arrs: Vec<crate::backends::metal::MetalArray> =
            Vec::with_capacity(stage_lengths.len());
        for (i, lens) in stage_lengths.iter().enumerate() {
            let mut arr = self.context.array(
                &[batch_size],
                DataType::I32,
                format!("nanocodec_enc_lengths_{i}"),
            );
            arr.as_slice_mut::<i32>().expect("dtype").copy_from_slice(lens);
            length_arrs.push(arr);
        }

        // Copy audio to device as [B, 1, T]
        let mut x_in = self.context.array(
            &[batch_size, 1, seq_len],
            DataType::F32,
            "nanocodec_enc_audio".into(),
        );
        x_in.as_slice_mut::<f32>().expect("dtype").copy_from_slice(audio);

        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // pre_conv: [B, 1, T] -> [B, C, T]
        let mut x = self.context.array(
            &[batch_size, self.weights.base_channels, seq_len],
            DataType::F32,
            "nanocodec_enc_pre".into(),
        );
        let pre_padding = get_padding(self.weights.pre_conv.kernel_size, 1);
        self.conv1d.encode(
            &encoder,
            Conv1dArguments {
                input: x_in.backend_buffer(),
                weight: self.weights.pre_conv.weight.backend_buffer(),
                bias: self.weights.pre_conv.bias.backend_buffer(),
                output: x.backend_buffer(),
                lengths: length_arrs[0].backend_buffer(),
                batch_size,
                cin: self.weights.pre_conv.cin,
                cout: self.weights.pre_conv.cout,
                seq_len_in: seq_len,
                seq_len_out: seq_len,
                kernel_size: self.weights.pre_conv.kernel_size,
                stride: 1,
                dilation: 1,
                padding: pre_padding,
                pad_mode: self.pad_mode,
            },
        )?;

        let mut curr_seq_len = seq_len;
        let mut curr_channels = self.weights.base_channels;

        for (layer_idx, layer) in self.weights.down_layers.iter().enumerate() {
            // res_layer average across branches.
            let n_stage = batch_size * curr_channels * curr_seq_len;

            let mut acc = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_acc"),
            );

            // Scratch buffers reused across all branches in this stage.
            let mut sig0 = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_sig0"),
            );
            let mut sig1 = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_sig1"),
            );
            let mut tmp = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_tmp"),
            );

            for branch in layer.res_layer.branches.iter() {
                // Ping-pong signal buffers for residual blocks.
                let mut in_buf = x.backend_buffer().to_owned();
                let mut out_arr: &mut crate::backends::metal::MetalArray = &mut sig0;
                let mut other_arr: &mut crate::backends::metal::MetalArray = &mut sig1;

                for block in branch.blocks.iter() {
                    // conv_input = lrelu(inputs)
                    self.lrelu.encode(
                        &encoder,
                        &in_buf,
                        tmp.backend_buffer(),
                        n_stage,
                        self.negative_slope,
                    )?;

                    // input_conv (dilated)
                    let in_pad = get_padding(block.input_conv.kernel_size, block.dilation);
                    self.conv1d.encode(
                        &encoder,
                        Conv1dArguments {
                            input: tmp.backend_buffer(),
                            weight: block.input_conv.weight.backend_buffer(),
                            bias: block.input_conv.bias.backend_buffer(),
                            output: out_arr.backend_buffer(),
                            lengths: length_arrs[layer_idx].backend_buffer(),
                            batch_size,
                            cin: block.input_conv.cin,
                            cout: block.input_conv.cout,
                            seq_len_in: curr_seq_len,
                            seq_len_out: curr_seq_len,
                            kernel_size: block.input_conv.kernel_size,
                            stride: 1,
                            dilation: block.dilation,
                            padding: in_pad,
                            pad_mode: self.pad_mode,
                        },
                    )?;

                    // skip_activation = lrelu
                    self.lrelu.encode(
                        &encoder,
                        out_arr.backend_buffer(),
                        tmp.backend_buffer(),
                        n_stage,
                        self.negative_slope,
                    )?;

                    // skip_conv (dilation=1)
                    let skip_pad = get_padding(block.skip_conv.kernel_size, 1);
                    self.conv1d.encode(
                        &encoder,
                        Conv1dArguments {
                            input: tmp.backend_buffer(),
                            weight: block.skip_conv.weight.backend_buffer(),
                            bias: block.skip_conv.bias.backend_buffer(),
                            output: out_arr.backend_buffer(),
                            lengths: length_arrs[layer_idx].backend_buffer(),
                            batch_size,
                            cin: block.skip_conv.cin,
                            cout: block.skip_conv.cout,
                            seq_len_in: curr_seq_len,
                            seq_len_out: curr_seq_len,
                            kernel_size: block.skip_conv.kernel_size,
                            stride: 1,
                            dilation: 1,
                            padding: skip_pad,
                            pad_mode: self.pad_mode,
                        },
                    )?;

                    // out = inputs + res (in-place into out_arr)
                    self.add.encode(
                        &encoder,
                        &in_buf,
                        out_arr.backend_buffer(),
                        out_arr.backend_buffer(),
                        n_stage,
                    )?;

                    // Next block: swap output buffers.
                    in_buf = out_arr.backend_buffer().to_owned();
                    std::mem::swap(&mut out_arr, &mut other_arr);
                }

                // Accumulate branch output into acc.
                self.add.encode(
                    &encoder,
                    acc.backend_buffer(),
                    &in_buf,
                    acc.backend_buffer(),
                    n_stage,
                )?;
            }

            // Average across branches.
            let inv_branches =
                1.0f32 / (layer.res_layer.branches.len() as f32);
            self.scale.encode(
                &encoder,
                acc.backend_buffer(),
                acc.backend_buffer(),
                n_stage,
                inv_branches,
            )?;

            // activation before downsample conv
            let mut act = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_act"),
            );
            self.lrelu.encode(
                &encoder,
                acc.backend_buffer(),
                act.backend_buffer(),
                n_stage,
                self.negative_slope,
            )?;

            // Downsample conv
            let stride = layer.stride;
            let next_seq_len = conv1d_out_len(
                curr_seq_len,
                layer.down_conv.kernel_size,
                stride,
                1,
                get_down_sample_padding(layer.down_conv.kernel_size, stride),
            );
            let next_channels = layer.down_conv.cout;

            let mut down = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_enc_stage_{layer_idx}_down"),
            );

            let down_pad =
                get_down_sample_padding(layer.down_conv.kernel_size, stride);
            self.conv1d.encode(
                &encoder,
                Conv1dArguments {
                    input: act.backend_buffer(),
                    weight: layer.down_conv.weight.backend_buffer(),
                    bias: layer.down_conv.bias.backend_buffer(),
                    output: down.backend_buffer(),
                    lengths: length_arrs[layer_idx + 1].backend_buffer(),
                    batch_size,
                    cin: layer.down_conv.cin,
                    cout: layer.down_conv.cout,
                    seq_len_in: curr_seq_len,
                    seq_len_out: next_seq_len,
                    kernel_size: layer.down_conv.kernel_size,
                    stride,
                    dilation: 1,
                    padding: down_pad,
                    pad_mode: self.pad_mode,
                },
            )?;

            x = down;
            curr_seq_len = next_seq_len;
            curr_channels = next_channels;
        }

        // post_activation + post_conv -> [B, encoded_dim, T_encoded]
        let n_post = batch_size * curr_channels * curr_seq_len;
        let mut post_act = self.context.array(
            &[batch_size, curr_channels, curr_seq_len],
            DataType::F32,
            "nanocodec_enc_post_act".into(),
        );
        self.lrelu.encode(
            &encoder,
            x.backend_buffer(),
            post_act.backend_buffer(),
            n_post,
            self.negative_slope,
        )?;

        let mut encoded = self.context.array(
            &[batch_size, self.weights.encoded_dim, curr_seq_len],
            DataType::F32,
            "nanocodec_enc_out".into(),
        );
        let post_pad = get_padding(self.weights.post_conv.kernel_size, 1);
        self.conv1d.encode(
            &encoder,
            Conv1dArguments {
                input: post_act.backend_buffer(),
                weight: self.weights.post_conv.weight.backend_buffer(),
                bias: self.weights.post_conv.bias.backend_buffer(),
                output: encoded.backend_buffer(),
                lengths: length_arrs.last().unwrap().backend_buffer(),
                batch_size,
                cin: self.weights.post_conv.cin,
                cout: self.weights.post_conv.cout,
                seq_len_in: curr_seq_len,
                seq_len_out: curr_seq_len,
                kernel_size: self.weights.post_conv.kernel_size,
                stride: 1,
                dilation: 1,
                padding: post_pad,
                pad_mode: self.pad_mode,
            },
        )?;

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((encoded, stage_lengths.last().unwrap().clone()))
    }
}

