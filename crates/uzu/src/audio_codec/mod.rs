pub mod nemo;
mod encoder;
mod model;
mod torch_checkpoint;

use std::{fs::File, path::Path, rc::Rc};

use half::{bf16, f16};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    Array,
    DataType,
    DeviceContext,
    backends::metal::{MTLContext, MetalArray},
    backends::metal::kernel::{
        AddKernel, AudioCodecKernelError, CausalConv1dArguments, CausalConv1dKernel,
        CausalConvTranspose1dArguments, CausalConvTranspose1dKernel, ClampKernel,
        FsqDecodeArguments, FsqDecodeKernel, HalfSnakeKernel, ScaleKernel,
    },
    parameters::{ParameterLoader, ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum NanoCodecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parameter loader error: {0}")]
    Parameters(#[from] ParameterLoaderError),
    #[error("Audio codec kernel error: {0}")]
    Kernel(#[from] AudioCodecKernelError),
    #[error("Unsupported dtype for weight folding: {0:?}")]
    UnsupportedWeightDType(DataType),
    #[error("Invalid weights: {0}")]
    InvalidWeights(String),
}

pub use encoder::NanoCodecEncoder;
pub use model::NanoCodecModel;

fn read_array_as_f32(array: &MetalArray) -> Result<Vec<f32>, NanoCodecError> {
    match array.data_type() {
        DataType::F32 => Ok(array
            .as_slice::<f32>()
            .expect("dtype checked")
            .to_vec()),
        DataType::F16 => Ok(array
            .as_slice::<f16>()
            .expect("dtype checked")
            .iter()
            .map(|x| x.to_f32())
            .collect()),
        DataType::BF16 => Ok(array
            .as_slice::<bf16>()
            .expect("dtype checked")
            .iter()
            .map(|x| x.to_f32())
            .collect()),
        other => Err(NanoCodecError::UnsupportedWeightDType(other)),
    }
}

fn fold_weight_norm_inplace_f32(
    g: &[f32],
    v: &mut [f32],
    dim0: usize,
    rest: usize,
) {
    for i in 0..dim0 {
        let gi = g[i];
        let row = &v[i * rest..(i + 1) * rest];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { gi / norm } else { 0.0 };
        let row_mut = &mut v[i * rest..(i + 1) * rest];
        for x in row_mut.iter_mut() {
            *x *= scale;
        }
    }
}

#[derive(Debug, Clone)]
struct Conv1dWeights {
    // [Cout, Cin, K]
    weight: MetalArray,
    // [Cout]
    bias: MetalArray,
    cin: usize,
    cout: usize,
    kernel_size: usize,
}

#[derive(Debug, Clone)]
struct ConvTranspose1dWeights {
    // [Cin, Cout/groups, K] where K=2*stride
    weight: MetalArray,
    // [Cout]
    bias: MetalArray,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct HalfSnakeAlpha {
    /// Alpha parameter for Snake part. Stored as shape [1, snake_channels, 1] (NeMo default),
    /// but treated as a contiguous 1D array indexed by channel.
    alpha: MetalArray,
    channels: usize,
    snake_channels: usize,
}

#[derive(Debug, Clone)]
struct ResidualBlockWeights {
    input_alpha: HalfSnakeAlpha,
    input_conv: Conv1dWeights,
    skip_alpha: HalfSnakeAlpha,
    skip_conv: Conv1dWeights,
    dilation: usize,
}

#[derive(Debug, Clone)]
struct ResBranchWeights {
    blocks: Box<[ResidualBlockWeights]>,
    kernel_size: usize,
}

#[derive(Debug, Clone)]
struct ResLayerWeights {
    branches: Box<[ResBranchWeights]>,
}

#[derive(Debug, Clone)]
struct UpsampleLayerWeights {
    up_conv: ConvTranspose1dWeights,
    res_layer: ResLayerWeights,
}

#[derive(Debug, Clone)]
struct DecoderWeights {
    pre_conv: Conv1dWeights,
    activations: Box<[HalfSnakeAlpha]>,
    up_layers: Box<[UpsampleLayerWeights]>,
    post_activation: HalfSnakeAlpha,
    post_conv: Conv1dWeights,
    input_dim: usize,
    base_channels: usize,
    post_kernel_size: usize,
}

fn max_index_with_prefix(
    keys: impl Iterator<Item = String>,
    prefix: &str,
) -> Option<usize> {
    keys.filter_map(|k| {
        k.strip_prefix(prefix)
            .and_then(|rest| rest.split('.').next())
            .and_then(|idx| idx.parse::<usize>().ok())
    })
    .max()
}

fn load_weight_norm_conv(
    context: &Rc<MTLContext>,
    tree: &ParameterTree<'_, Rc<MTLContext>>,
    prefix: &str,
    label_prefix: &str,
) -> Result<(MetalArray, MetalArray), NanoCodecError> {
    let conv_tree = tree.subtree(prefix)?;
    let g_arr = conv_tree.leaf("parametrizations.weight.original0")?;
    let v_arr = conv_tree.leaf("parametrizations.weight.original1")?;
    let bias_arr = conv_tree.leaf("bias")?;

    let v_shape = v_arr.shape().to_vec();
    if v_shape.is_empty() {
        return Err(NanoCodecError::InvalidWeights(format!(
            "{label_prefix}: expected weight to be 3D, got scalar"
        )));
    }
    if v_shape.len() != 3 {
        return Err(NanoCodecError::InvalidWeights(format!(
            "{label_prefix}: expected weight to be 3D, got shape {v_shape:?}"
        )));
    }

    let dim0 = v_shape[0];
    let rest = v_shape[1] * v_shape[2];

    // Try fast path: fold in-place when V is f32.
    let weight_fused = if v_arr.data_type() == DataType::F32 {
        let g = read_array_as_f32(&g_arr)?;
        let mut v = v_arr.clone();
        {
            let v_slice = v
                .as_slice_mut::<f32>()
                .expect("dtype checked");
            fold_weight_norm_inplace_f32(&g, v_slice, dim0, rest);
        }
        v
    } else {
        // Convert to f32 buffer then fold.
        let g = read_array_as_f32(&g_arr)?;
        let mut v_f32 = read_array_as_f32(&v_arr)?;
        fold_weight_norm_inplace_f32(&g, &mut v_f32, dim0, rest);
        let mut out = context.array(
            &v_shape,
            DataType::F32,
            format!("{label_prefix}_weight_fused"),
        );
        out.as_slice_mut::<f32>()
            .expect("dtype")
            .copy_from_slice(&v_f32);
        out
    };

    let bias_f32 = if bias_arr.data_type() == DataType::F32 {
        bias_arr
    } else {
        let bias = read_array_as_f32(&bias_arr)?;
        let mut out = context.array(
            bias_arr.shape(),
            DataType::F32,
            format!("{label_prefix}_bias_f32"),
        );
        out.as_slice_mut::<f32>()
            .expect("dtype")
            .copy_from_slice(&bias);
        out
    };

    Ok((weight_fused, bias_f32))
}

fn load_half_snake_alpha(
    context: &Rc<MTLContext>,
    tree: &ParameterTree<'_, Rc<MTLContext>>,
    alpha_key: &str,
    channels: usize,
    label_prefix: &str,
) -> Result<HalfSnakeAlpha, NanoCodecError> {
    let alpha_arr = tree.leaf(alpha_key)?;
    let shape = alpha_arr.shape().to_vec();
    let snake_channels = match shape.as_slice() {
        [sc] => *sc,
        [1, sc, 1] => *sc,
        [1, sc] => *sc,
        other => {
            return Err(NanoCodecError::InvalidWeights(format!(
                "{label_prefix}: expected alpha shape [snake_channels] or [1,snake_channels,1], got {other:?}"
            )));
        },
    };

    if snake_channels > channels {
        return Err(NanoCodecError::InvalidWeights(format!(
            "{label_prefix}: snake_channels {snake_channels} must be <= channels {channels}"
        )));
    }

    let alpha_f32 = if alpha_arr.data_type() == DataType::F32 {
        alpha_arr
    } else {
        let alpha = read_array_as_f32(&alpha_arr)?;
        let mut out = context.array(
            alpha_arr.shape(),
            DataType::F32,
            format!("{label_prefix}_alpha_f32"),
        );
        out.as_slice_mut::<f32>()
            .expect("dtype")
            .copy_from_slice(&alpha);
        out
    };

    Ok(HalfSnakeAlpha {
        alpha: alpha_f32,
        channels,
        snake_channels,
    })
}

impl Conv1dWeights {
    fn load(
        context: &Rc<MTLContext>,
        tree: &ParameterTree<'_, Rc<MTLContext>>,
        prefix: &str,
        label_prefix: &str,
    ) -> Result<Self, NanoCodecError> {
        let (weight, bias) =
            load_weight_norm_conv(context, tree, prefix, label_prefix)?;
        let shape = weight.shape();
        let cout = shape[0];
        let cin = shape[1];
        let kernel_size = shape[2];
        Ok(Self {
            weight,
            bias,
            cin,
            cout,
            kernel_size,
        })
    }
}

impl ConvTranspose1dWeights {
    fn load(
        context: &Rc<MTLContext>,
        tree: &ParameterTree<'_, Rc<MTLContext>>,
        prefix: &str,
        label_prefix: &str,
    ) -> Result<Self, NanoCodecError> {
        let (weight, bias) =
            load_weight_norm_conv(context, tree, prefix, label_prefix)?;
        let shape = weight.shape();
        let cin = shape[0];
        let cout_per_group = shape[1];
        let kernel_size = shape[2];
        let cout = bias.num_elements();
        if kernel_size % 2 != 0 {
            return Err(NanoCodecError::InvalidWeights(format!(
                "{label_prefix}: expected kernel_size to be even, got {kernel_size}"
            )));
        }
        let stride = kernel_size / 2;
        let groups = cout / cout_per_group;
        if groups == 0 || cin % groups != 0 || cout % groups != 0 {
            return Err(NanoCodecError::InvalidWeights(format!(
                "{label_prefix}: invalid inferred groups={groups} for cin={cin}, cout={cout}, cout_per_group={cout_per_group}"
            )));
        }
        Ok(Self {
            weight,
            bias,
            cin,
            cout,
            kernel_size,
            stride,
            groups,
        })
    }
}

impl DecoderWeights {
    fn load(
        context: &Rc<MTLContext>,
        decoder_weights_path: &Path,
    ) -> Result<Self, NanoCodecError> {
        let file = File::open(decoder_weights_path)?;
        let loader = ParameterLoader::new(&file, context).map_err(|e| {
            NanoCodecError::InvalidWeights(format!(
                "Failed to read safetensors header for decoder weights: {e}"
            ))
        })?;
        let keys: Vec<String> =
            loader.keys().cloned().collect();
        let tree = loader.tree();

        let pre_conv = Conv1dWeights::load(
            context,
            &tree,
            "pre_conv.conv",
            "nanocodec_pre_conv",
        )?;
        let post_conv = Conv1dWeights::load(
            context,
            &tree,
            "post_conv.conv",
            "nanocodec_post_conv",
        )?;

        let input_dim = pre_conv.cin;
        let base_channels = pre_conv.cout;
        let post_kernel_size = post_conv.kernel_size;

        // Discover number of upsample layers from keys.
        let max_up = max_index_with_prefix(
            keys.clone().into_iter(),
            "up_sample_conv_layers.",
        )
        .ok_or_else(|| NanoCodecError::InvalidWeights("No up_sample_conv_layers.* found".into()))?;
        let num_up_layers = max_up + 1;

        // Load per-layer activations (NanoCodec uses half_snake).
        let mut activations = Vec::with_capacity(num_up_layers);
        let mut act_channels = base_channels;
        for layer_idx in 0..num_up_layers {
            let alpha_key = format!(
                "activations.{layer_idx}.activation.snake_act.alpha"
            );
            let alpha = load_half_snake_alpha(
                context,
                &tree,
                &alpha_key,
                act_channels,
                &format!("nanocodec_activation_{layer_idx}"),
            )?;
            activations.push(alpha);
            act_channels /= 2;
        }

        let post_activation = load_half_snake_alpha(
            context,
            &tree,
            "post_activation.activation.snake_act.alpha",
            act_channels,
            "nanocodec_post_activation",
        )?;

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

        let mut up_layers = Vec::with_capacity(num_up_layers);
        for layer_idx in 0..num_up_layers {
            let up_conv = ConvTranspose1dWeights::load(
                context,
                &tree,
                &format!("up_sample_conv_layers.{layer_idx}.conv"),
                &format!("nanocodec_up_conv_{layer_idx}"),
            )?;

            // Load res_layer weights for this stage.
            let mut branches = Vec::with_capacity(num_branches);
            for branch_idx in 0..num_branches {
                let mut blocks = Vec::with_capacity(num_dilations);
                for dilation_idx in 0..num_dilations {
                    let dilation = dilation_sizes[dilation_idx];
                    let base_prefix = format!(
                        "res_layers.{layer_idx}.res_blocks.{branch_idx}.res_blocks.{dilation_idx}"
                    );

                    // Activations are HalfSnake in NanoCodec.
                    let input_alpha = load_half_snake_alpha(
                        context,
                        &tree,
                        &format!(
                            "{base_prefix}.input_activation.activation.snake_act.alpha"
                        ),
                        up_conv.cout, // NOTE: after upsample, residual blocks operate on next_channels (== up_conv.cout)
                        &format!(
                            "nanocodec_res_in_alpha_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;

                    let skip_alpha = load_half_snake_alpha(
                        context,
                        &tree,
                        &format!(
                            "{base_prefix}.skip_activation.activation.snake_act.alpha"
                        ),
                        up_conv.cout,
                        &format!(
                            "nanocodec_res_skip_alpha_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;

                    let input_conv = Conv1dWeights::load(
                        context,
                        &tree,
                        &format!("{base_prefix}.input_conv.conv"),
                        &format!(
                            "nanocodec_res_in_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;
                    let skip_conv = Conv1dWeights::load(
                        context,
                        &tree,
                        &format!("{base_prefix}.skip_conv.conv"),
                        &format!(
                            "nanocodec_res_skip_{layer_idx}_{branch_idx}_{dilation_idx}"
                        ),
                    )?;

                    blocks.push(ResidualBlockWeights {
                        input_alpha,
                        input_conv,
                        skip_alpha,
                        skip_conv,
                        dilation,
                    });
                }

                let kernel_size = blocks
                    .first()
                    .map(|b| b.input_conv.kernel_size)
                    .unwrap_or(0);

                branches.push(ResBranchWeights {
                    blocks: blocks.into_boxed_slice(),
                    kernel_size,
                });
            }

            up_layers.push(UpsampleLayerWeights {
                up_conv,
                res_layer: ResLayerWeights {
                    branches: branches.into_boxed_slice(),
                },
            });
        }

        Ok(Self {
            pre_conv,
            activations: activations.into_boxed_slice(),
            up_layers: up_layers.into_boxed_slice(),
            post_activation,
            post_conv,
            input_dim,
            base_channels,
            post_kernel_size,
        })
    }
}

pub struct NanoCodecDecoder {
    context: Rc<MTLContext>,
    // Quantizer config (GroupFSQ)
    num_codebooks: usize,
    codebook_dim_per_group: usize,
    num_levels_per_group: Box<[i32]>,

    // Decoder weights
    weights: DecoderWeights,

    // Kernels (float32 for now)
    fsq: FsqDecodeKernel,
    half_snake: HalfSnakeKernel,
    clamp: ClampKernel,
    add: AddKernel,
    scale: ScaleKernel,
    conv1d: CausalConv1dKernel,
    convtr: CausalConvTranspose1dKernel,

    negative_slope: f32,
    snake_eps: f32,
}

#[derive(Debug, Deserialize)]
struct ExportedVectorQuantizerConfig {
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct ExportedNanoCodecConfig {
    vector_quantizer: ExportedVectorQuantizerConfig,
}

impl NanoCodecDecoder {
    /// Create a decoder for NeMo NanoCodec-style models:
    /// - GroupFiniteScalarQuantizer with num_codebooks groups and identical num_levels_per_group
    /// - CausalHiFiGANDecoder generator with weight-norm folded at load time
    pub fn load(
        context: Rc<MTLContext>,
        decoder_weights_path: &Path,
        num_codebooks: usize,
        codebook_dim_per_group: usize,
        num_levels_per_group: Box<[i32]>,
    ) -> Result<Self, NanoCodecError> {
        if num_codebooks == 0 {
            return Err(NanoCodecError::InvalidWeights(
                "num_codebooks must be > 0".into(),
            ));
        }
        if codebook_dim_per_group == 0 {
            return Err(NanoCodecError::InvalidWeights(
                "codebook_dim_per_group must be > 0".into(),
            ));
        }
        if num_levels_per_group.len() != codebook_dim_per_group {
            return Err(NanoCodecError::InvalidWeights(format!(
                "num_levels_per_group length {} must equal codebook_dim_per_group {codebook_dim_per_group}",
                num_levels_per_group.len()
            )));
        }

        let weights = DecoderWeights::load(&context, decoder_weights_path)?;

        // Sanity: quantizer output dim must match decoder input dim.
        let input_dim = num_codebooks * codebook_dim_per_group;
        if weights.input_dim != input_dim {
            return Err(NanoCodecError::InvalidWeights(format!(
                "Decoder pre_conv expects input_dim={}, but FSQ config would produce {} (num_codebooks={} * codebook_dim_per_group={})",
                weights.input_dim, input_dim, num_codebooks, codebook_dim_per_group
            )));
        }

        // Build kernels (float32 for now).
        let fsq = FsqDecodeKernel::new(&context, crate::backends::metal::KernelDataType::Float32)?;
        let half_snake =
            HalfSnakeKernel::new(&context, crate::backends::metal::KernelDataType::Float32)?;
        let clamp =
            ClampKernel::new(&context, crate::backends::metal::KernelDataType::Float32)?;
        let add =
            AddKernel::new(&context, crate::backends::metal::KernelDataType::Float32)?;
        let scale =
            ScaleKernel::new(&context, crate::backends::metal::KernelDataType::Float32)?;
        let conv1d = CausalConv1dKernel::new(
            &context,
            crate::backends::metal::KernelDataType::Float32,
        )?;
        let convtr = CausalConvTranspose1dKernel::new(
            &context,
            crate::backends::metal::KernelDataType::Float32,
        )?;

        Ok(Self {
            context,
            num_codebooks,
            codebook_dim_per_group,
            num_levels_per_group,
            weights,
            fsq,
            half_snake,
            clamp,
            add,
            scale,
            conv1d,
            convtr,
            negative_slope: 0.01,
            snake_eps: 1e-9,
        })
    }

    /// Load a NanoCodec decoder from an export directory produced by the Rust `.nemo` exporter
    /// (`uzu::audio_codec::nemo::export_nanocodec_from_nemo`).
    ///
    /// Expects:
    /// - `nanocodec_config.json`
    /// - `audio_decoder.safetensors`
    pub fn load_from_export_dir(
        context: Rc<MTLContext>,
        export_dir: &Path,
    ) -> Result<Self, NanoCodecError> {
        let config_path = export_dir.join("nanocodec_config.json");
        let decoder_path = export_dir.join("audio_decoder.safetensors");
        let file = File::open(&config_path)?;
        let cfg: ExportedNanoCodecConfig = serde_json::from_reader(file)
            .map_err(|e| NanoCodecError::InvalidWeights(format!("Invalid nanocodec_config.json: {e}")))?;

        let num_codebooks = cfg.vector_quantizer.num_groups;
        let levels = cfg.vector_quantizer.num_levels_per_group;
        let codebook_dim_per_group = levels.len();
        Self::load(
            context,
            &decoder_path,
            num_codebooks,
            codebook_dim_per_group,
            levels.into_boxed_slice(),
        )
    }

    /// Decode token indices to waveform.
    ///
    /// Tokens are expected in NeMo public API shape: [B, num_codebooks, T_encoded]
    pub fn decode(
        &self,
        tokens: &[i32],
        token_lengths: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), NanoCodecError> {
        if token_lengths.len() != batch_size {
            return Err(NanoCodecError::InvalidWeights(
                "token_lengths len must match batch_size".into(),
            ));
        }
        let expected_tokens =
            batch_size * self.num_codebooks * seq_len;
        if tokens.len() != expected_tokens {
            return Err(NanoCodecError::InvalidWeights(format!(
                "tokens len {} must equal batch_size*num_codebooks*seq_len = {expected_tokens}",
                tokens.len()
            )));
        }

        // Precompute stage lengths and stage sequence lengths.
        let strides: Vec<usize> = self
            .weights
            .up_layers
            .iter()
            .map(|l| l.up_conv.stride)
            .collect();

        let mut stage_lengths: Vec<Vec<i32>> = Vec::with_capacity(strides.len() + 1);
        stage_lengths.push(token_lengths.to_vec());
        for &s in &strides {
            let prev = stage_lengths.last().unwrap();
            stage_lengths.push(prev.iter().map(|&l| l * (s as i32)).collect());
        }
        let mut stage_seq_lens: Vec<usize> = Vec::with_capacity(strides.len() + 1);
        stage_seq_lens.push(seq_len);
        for &s in &strides {
            stage_seq_lens.push(stage_seq_lens.last().unwrap() * s);
        }

        // Upload tokens + lengths (one lengths buffer per stage; immutable during GPU execution).
        let mut tokens_arr = self.context.array(
            &[batch_size, self.num_codebooks, seq_len],
            DataType::I32,
            "nanocodec_tokens".into(),
        );
        tokens_arr
            .as_slice_mut::<i32>()
            .expect("dtype")
            .copy_from_slice(tokens);

        let mut length_arrs: Vec<MetalArray> = Vec::with_capacity(stage_lengths.len());
        for (i, lens) in stage_lengths.iter().enumerate() {
            let mut a = self.context.array(
                &[batch_size],
                DataType::I32,
                format!("nanocodec_lengths_stage_{i}"),
            );
            a.as_slice_mut::<i32>()
                .expect("dtype")
                .copy_from_slice(lens);
            length_arrs.push(a);
        }

        let input_dim = self.num_codebooks * self.codebook_dim_per_group;
        let mut codes = self.context.array(
            &[batch_size, input_dim, seq_len],
            DataType::F32,
            "nanocodec_codes".into(),
        );

        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // FSQ decode: tokens -> continuous codes (latents).
        self.fsq.encode(
            &encoder,
            FsqDecodeArguments {
                tokens: tokens_arr.backend_buffer(),
                lengths: length_arrs[0].backend_buffer(),
                out: codes.backend_buffer(),
                batch_size,
                num_groups: self.num_codebooks,
                seq_len,
                codebook_dim_per_group: self.codebook_dim_per_group,
                num_levels_per_group: self.num_levels_per_group.clone(),
            },
        )?;

        // pre_conv: [B, input_dim, T] -> [B, base_channels, T]
        let mut x = self.context.array(
            &[batch_size, self.weights.base_channels, seq_len],
            DataType::F32,
            "nanocodec_pre_out".into(),
        );
        self.conv1d.encode(
            &encoder,
            CausalConv1dArguments {
                input: codes.backend_buffer(),
                weight: self.weights.pre_conv.weight.backend_buffer(),
                bias: self.weights.pre_conv.bias.backend_buffer(),
                output: x.backend_buffer(),
                lengths: length_arrs[0].backend_buffer(),
                batch_size,
                cin: self.weights.pre_conv.cin,
                cout: self.weights.pre_conv.cout,
                seq_len,
                kernel_size: self.weights.pre_conv.kernel_size,
                dilation: 1,
            },
        )?;

        // Upsample stack
        let mut curr_seq_len = seq_len;
        let mut curr_channels = self.weights.base_channels;

        for (layer_idx, layer) in self.weights.up_layers.iter().enumerate() {
            let stride = layer.up_conv.stride;
            let next_seq_len = curr_seq_len * stride;
            let next_channels = curr_channels / 2;

            // activation before upsample conv
            let mut act = self.context.array(
                &[batch_size, curr_channels, curr_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_act"),
            );
            let stage_act = &self.weights.activations[layer_idx];
            self.half_snake.encode(
                &encoder,
                x.backend_buffer(),
                stage_act.alpha.backend_buffer(),
                act.backend_buffer(),
                batch_size,
                stage_act.channels,
                curr_seq_len,
                stage_act.snake_channels,
                self.negative_slope,
                self.snake_eps,
            )?;

            // upsample convtranspose
            let mut up = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_up"),
            );
            self.convtr.encode(
                &encoder,
                CausalConvTranspose1dArguments {
                    input: act.backend_buffer(),
                    weight: layer.up_conv.weight.backend_buffer(),
                    bias: layer.up_conv.bias.backend_buffer(),
                    output: up.backend_buffer(),
                    lengths: length_arrs[layer_idx + 1].backend_buffer(),
                    batch_size,
                    cin: layer.up_conv.cin,
                    cout: layer.up_conv.cout,
                    seq_len_in: curr_seq_len,
                    seq_len_out: next_seq_len,
                    stride,
                    groups: layer.up_conv.groups,
                },
            )?;

            // Res layer: average across branches.
            let mut acc = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_acc"),
            );

            // Scratch buffers reused across all branches in this stage.
            let mut sig0 = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_sig0"),
            );
            let mut sig1 = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_sig1"),
            );
            let mut tmp = self.context.array(
                &[batch_size, next_channels, next_seq_len],
                DataType::F32,
                format!("nanocodec_stage_{layer_idx}_tmp"),
            );

            let n_stage = batch_size * next_channels * next_seq_len;
            for branch in layer.res_layer.branches.iter() {
                // Ping-pong signal buffers for residual blocks.
                let mut in_buf = up.backend_buffer().to_owned();
                let mut out_arr: &mut MetalArray = &mut sig0;
                let mut other_arr: &mut MetalArray = &mut sig1;

                for block in branch.blocks.iter() {
                    // conv_input = HalfSnake(inputs)
                    self.half_snake.encode(
                        &encoder,
                        &in_buf,
                        block.input_alpha.alpha.backend_buffer(),
                        tmp.backend_buffer(),
                        batch_size,
                        block.input_alpha.channels,
                        next_seq_len,
                        block.input_alpha.snake_channels,
                        self.negative_slope,
                        self.snake_eps,
                    )?;
                    // input_conv (dilated)
                    self.conv1d.encode(
                        &encoder,
                        CausalConv1dArguments {
                            input: tmp.backend_buffer(),
                            weight: block.input_conv.weight.backend_buffer(),
                            bias: block.input_conv.bias.backend_buffer(),
                            output: out_arr.backend_buffer(),
                            lengths: length_arrs[layer_idx + 1].backend_buffer(),
                            batch_size,
                            cin: block.input_conv.cin,
                            cout: block.input_conv.cout,
                            seq_len: next_seq_len,
                            kernel_size: block.input_conv.kernel_size,
                            dilation: block.dilation,
                        },
                    )?;
                    // skip_activation = HalfSnake
                    self.half_snake.encode(
                        &encoder,
                        out_arr.backend_buffer(),
                        block.skip_alpha.alpha.backend_buffer(),
                        tmp.backend_buffer(),
                        batch_size,
                        block.skip_alpha.channels,
                        next_seq_len,
                        block.skip_alpha.snake_channels,
                        self.negative_slope,
                        self.snake_eps,
                    )?;
                    // skip_conv (dilation=1)
                    self.conv1d.encode(
                        &encoder,
                        CausalConv1dArguments {
                            input: tmp.backend_buffer(),
                            weight: block.skip_conv.weight.backend_buffer(),
                            bias: block.skip_conv.bias.backend_buffer(),
                            output: out_arr.backend_buffer(),
                            lengths: length_arrs[layer_idx + 1].backend_buffer(),
                            batch_size,
                            cin: block.skip_conv.cin,
                            cout: block.skip_conv.cout,
                            seq_len: next_seq_len,
                            kernel_size: block.skip_conv.kernel_size,
                            dilation: 1,
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
            let inv_branches = 1.0f32 / (layer.res_layer.branches.len() as f32);
            self.scale.encode(
                &encoder,
                acc.backend_buffer(),
                acc.backend_buffer(),
                n_stage,
                inv_branches,
            )?;

            // Next stage input
            x = acc;
            curr_seq_len = next_seq_len;
            curr_channels = next_channels;
        }

        // post_activation + post_conv -> [B, 1, T]
        let mut post_act = self.context.array(
            &[batch_size, curr_channels, curr_seq_len],
            DataType::F32,
            "nanocodec_post_act".into(),
        );
        self.half_snake.encode(
            &encoder,
            x.backend_buffer(),
            self.weights.post_activation.alpha.backend_buffer(),
            post_act.backend_buffer(),
            batch_size,
            self.weights.post_activation.channels,
            curr_seq_len,
            self.weights.post_activation.snake_channels,
            self.negative_slope,
            self.snake_eps,
        )?;

        let mut post = self.context.array(
            &[batch_size, 1, curr_seq_len],
            DataType::F32,
            "nanocodec_post".into(),
        );
        self.conv1d.encode(
            &encoder,
            CausalConv1dArguments {
                input: post_act.backend_buffer(),
                weight: self.weights.post_conv.weight.backend_buffer(),
                bias: self.weights.post_conv.bias.backend_buffer(),
                output: post.backend_buffer(),
                lengths: length_arrs.last().unwrap().backend_buffer(),
                batch_size,
                cin: self.weights.post_conv.cin,
                cout: self.weights.post_conv.cout,
                seq_len: curr_seq_len,
                kernel_size: self.weights.post_kernel_size,
                dilation: 1,
            },
        )?;

        // clamp to [-1, 1]
        self.clamp.encode(
            &encoder,
            post.backend_buffer(),
            post.backend_buffer(),
            batch_size * curr_seq_len,
            -1.0,
            1.0,
        )?;

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back: [B, 1, T] -> Vec<f32> [B, T]
        let out_slice = post.as_slice::<f32>().expect("dtype");
        let audio: Vec<f32> = out_slice
            .chunks_exact(curr_seq_len)
            .flat_map(|chunk| chunk.iter().copied())
            .collect();

        Ok((audio, stage_lengths.last().unwrap().clone()))
    }
}

