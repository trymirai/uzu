use std::{any::Any, range::Range};

use half::bf16;
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels,
        gpu_types::quantization::QuantizationMode,
        kernel::{
            MarkovEmbedKernel, MarkovLogitsKernel, MarkovStartKernel, MarkovStepKernel, TapMixNormKernel,
            matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulGather, MatmulKernel, MatmulOutput, QuantParams,
                QuantParamsLayout, QuantizedB, QuantizedCorrection, WeightsLayout,
            },
        },
    },
    config::{
        residual::{AnyResidualConfig, additive::AdditiveResidualConfig},
        speculator::dspark::{DSparkSpeculatorConfig, TapMixConfig},
        token_mixer::AnyTokenMixerConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        block_attention::upload,
        decoder::Decoder,
        dflash::DFlashOutput,
        embedding::EmbeddingError,
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::{
            MixerState,
            attention::{ATTENTION_SUFFIX_CAPACITY, Attention, AttentionNewError, AttentionState},
        },
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        sampling::{Sampling, SamplingMethod},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_mut::MaybeMut, size_for_shape::size_for_shape},
};

/// The tap mixes run in one kernel that reads four taps.
const TAP_COUNT: usize = 4;
/// The folded Markov head is stored in MLX affine 4-bit with groups of this many ranks: every proposal position reads
/// it whole, and it only shapes proposals.
const FOLD_GROUP_SIZE: usize = 64;

/// A matrix in MLX affine 4-bit: codes [rows][columns / 8] words, low nibble first, and bfloat16 scales and biases
/// [group][row], with value = scale * code + bias.
struct AffineMatrix<B: Backend> {
    codes: B::GlobalGpuBuffer,
    scales: B::GlobalGpuBuffer,
    biases: B::GlobalGpuBuffer,
    rows: u32,
    groups: u32,
}

fn quantize_affine<B: Backend>(
    values: &[f32],
    rows: usize,
    columns: usize,
    context: &B::Context,
) -> Result<AffineMatrix<B>, B::Error> {
    assert!(columns.is_multiple_of(FOLD_GROUP_SIZE) && values.len() == rows * columns);
    let groups = columns / FOLD_GROUP_SIZE;
    let mut codes = vec![0u32; rows * columns / 8];
    let mut scales = vec![bf16::ZERO; groups * rows];
    let mut biases = vec![bf16::ZERO; groups * rows];
    for row in 0..rows {
        for group in 0..groups {
            let start = row * columns + group * FOLD_GROUP_SIZE;
            let group_values = &values[start..start + FOLD_GROUP_SIZE];
            let minimum = group_values.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = group_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = bf16::from_f32(((maximum - minimum) / 15.0).max(f32::MIN_POSITIVE));
            let bias = bf16::from_f32(minimum);
            scales[group * rows + row] = scale;
            biases[group * rows + row] = bias;
            for (offset, value) in group_values.iter().enumerate() {
                let code = ((value - bias.to_f32()) / scale.to_f32()).round().clamp(0.0, 15.0) as u32;
                let index = start + offset;
                codes[index / 8] |= code << (4 * (index % 8));
            }
        }
    }
    Ok(AffineMatrix {
        codes: upload::<B, u32>(&codes, context)?,
        scales: upload::<B, bf16>(&scales, context)?,
        biases: upload::<B, bf16>(&biases, context)?,
        rows: rows as u32,
        groups: groups as u32,
    })
}

type MatmulError<B> = <<<B as Backend>::Kernels as Kernels>::MatmulKernel as MatmulKernel>::Error;

#[derive(Debug, Error)]
pub enum DSparkNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Transformer layer error: {0}")]
    TransformerLayer(#[from] TransformerLayerError<B>),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("Matmul error: {0}")]
    Matmul(#[source] MatmulError<B>),
    #[error("unsupported DSpark draft: {0}")]
    Unsupported(String),
}

#[derive(Debug, Error)]
pub enum DSparkEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("Transformer layer error: {0}")]
    TransformerLayer(#[from] TransformerLayerError<B>),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("Matmul error: {0}")]
    Matmul(#[source] MatmulError<B>),
}

pub struct DSparkState<B: Backend> {
    layer_states: Box<[Box<dyn MixerState<B>>]>,
    /// The Markov head's mix of the taps of the newest committed row.
    markov_context: B::GlobalGpuBuffer,
    context_length: u32,
}

impl<B: Backend> DSparkState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }
}

struct DraftLayer<B: Backend> {
    layer: TransformerLayer<B>,
    context_kv_projection: Box<dyn Linear<B>>,
    context_key_norm_scales: B::GlobalGpuBuffer,
}

struct MarkovHead<B: Backend> {
    rank: u32,
    token_embeddings: B::GlobalGpuBuffer,
    recurrence: Box<dyn Linear<B>>,
    /// The target's LM head applied to the head's output projection: [target vocabulary][rank], and to its bias.
    fold: AffineMatrix<B>,
    fold_bias: B::GlobalGpuBuffer,
    soft_cap: Option<f32>,
    start: <B::Kernels as Kernels>::MarkovStartKernel,
    /// Without, then with a next step.
    step: [<B::Kernels as Kernels>::MarkovStepKernel; 2],
    logits: <B::Kernels as Kernels>::MarkovLogitsKernel,
    embed: <B::Kernels as Kernels>::MarkovEmbedKernel,
    fold_matmul: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
}

/// The DSpark block drafter, see `DSparkSpeculatorConfig`.
pub struct DSpark<B: Backend> {
    layers: Box<[DraftLayer<B>]>,
    tap_mix: <B::Kernels as Kernels>::TapMixNormKernel,
    /// softmax(mix logits) of each layer, then of the Markov head: [mix][tap].
    mix_weights: B::GlobalGpuBuffer,
    mix_norm_scales: B::GlobalGpuBuffer,
    mix_epsilon: f32,
    output_norm: Normalization<B>,
    /// The mask embedding, once per proposal position.
    mask_rows: B::GlobalGpuBuffer,
    markov: MarkovHead<B>,
    model_dim: u32,
    block_size: u32,
    max_context_length: u32,
    normalizes_anchor_embedding: bool,
    vocab_size: u32,
    data_type: DataType,
}

async fn read_tap_mix<B: Backend>(
    config: &TapMixConfig,
    model_dim: u32,
    tree: &ParameterTree<'_, B>,
) -> Result<(Box<[f32]>, Box<[f32]>), DSparkNewError<B>> {
    if config.num_taps as usize != TAP_COUNT {
        return Err(DSparkNewError::Unsupported(format!("{} taps, the tap mix kernel reads {TAP_COUNT}", config.num_taps)));
    }
    let logits = tree.read_slice::<f32>("mix_logits", &[config.num_taps]).await?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = logits.iter().map(|logit| (logit - maximum).exp()).collect::<Box<[f32]>>();
    let total = exponentials.iter().sum::<f32>();
    // The reference rounds the mix weights to bfloat16.
    let weights = exponentials.iter().map(|value| bf16::from_f32(value / total).to_f32()).collect();
    let norm_scales = tree.subtree("norm").read_slice::<f32>("scales", &[model_dim]).await?;
    Ok((weights, norm_scales.into()))
}

impl<B: Backend> DSpark<B> {
    pub async fn new(
        config: &DSparkSpeculatorConfig,
        target: &Decoder<B>,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, DSparkNewError<B>> {
        let data_type = DataType::BF16;
        let model_dim = config.model_dim;
        let target_embedding = target.embedding();
        if target_embedding.model_dim() != model_dim {
            return Err(DSparkNewError::Unsupported(format!(
                "draft model_dim {model_dim} differs from the target's {}",
                target_embedding.model_dim()
            )));
        }
        if target_embedding.readout_input_hadamard_factors().is_some() {
            return Err(DSparkNewError::Unsupported("a target readout with rotated inputs".into()));
        }
        if config.normalizes_anchor_embedding && target.embedding_norm().is_none() {
            return Err(DSparkNewError::Unsupported("anchor normalization without a target embedding norm".into()));
        }
        if config.block_size > ATTENTION_SUFFIX_CAPACITY {
            return Err(DSparkNewError::Unsupported(format!("block size {} exceeds the suffix", config.block_size)));
        }

        let residual_config = AnyResidualConfig::AdditiveResidualConfig(AdditiveResidualConfig::default());
        let layers_tree = tree.subtree("layers");
        let num_layers = config.layer_configs.len() as u32;
        let mut layers = Vec::with_capacity(num_layers as usize);
        let mut mix_weights = Vec::new();
        let mut mix_norm_scales = Vec::new();
        for (layer_config, index) in config.layer_configs.iter().zip(0u32..) {
            let layer_tree = layers_tree.subtree(&index.to_string());
            let mut transformer_layer_config = layer_config.transformer_layer_config.clone();
            let AnyTokenMixerConfig::AttentionConfig(attention_config) =
                &mut transformer_layer_config.mixer.body_config
            else {
                return Err(DSparkNewError::Unsupported("draft layers without attention".into()));
            };
            // The context keeps only the newest rows, whatever the trained window of the layer.
            attention_config.sliding_window_size = Some(config.context_window + 1);
            let kv_dim = attention_config.num_groups * attention_config.head_dim;
            let head_dim = attention_config.head_dim;
            let layer = TransformerLayer::new(
                model_dim,
                config.hidden_dim,
                num_layers,
                index,
                data_type,
                &transformer_layer_config,
                &residual_config,
                None,
                &layer_tree.subtree("transformer_layer"),
                context,
            )
            .await?;
            let context_kv_projection = <dyn Linear<B>>::new(
                model_dim,
                2 * kv_dim,
                data_type,
                false,
                &layer_tree.subtree("context_kv_projection"),
                context,
            )
            .await?;
            let context_key_norm_scales =
                layer_tree.subtree("context_key_norm").read_buffer("scales", &[head_dim], DataType::F32).await?;
            let (weights, norm_scales) =
                read_tap_mix(&layer_config.context_mix_config, model_dim, &layer_tree.subtree("context_mix")).await?;
            mix_weights.extend_from_slice(&weights);
            mix_norm_scales.extend_from_slice(&norm_scales);
            layers.push(DraftLayer {
                layer,
                context_kv_projection,
                context_key_norm_scales,
            });
        }

        let head_config = &config.markov_head_config;
        let head_tree = tree.subtree("markov_head");
        let rank = head_config.rank;
        let (weights, norm_scales) =
            read_tap_mix(&head_config.context_mix_config, model_dim, &head_tree.subtree("context_mix")).await?;
        mix_weights.extend_from_slice(&weights);
        mix_norm_scales.extend_from_slice(&norm_scales);
        let mix_count = num_layers + 1;
        let mix_epsilon = config.layer_configs[0].context_mix_config.norm_config.epsilon;

        let recurrence = <dyn Linear<B>>::new(
            2 * rank + 2 * model_dim,
            3 * rank,
            data_type,
            true,
            &head_tree.subtree("recurrence"),
            context,
        )
        .await?;
        let token_embeddings =
            head_tree.read_buffer("token_embeddings", &[head_config.vocab_size, rank], DataType::BF16).await?;
        let (fold, fold_bias) = Self::fold_markov_output(target, &head_tree.subtree("output"), rank, context).await?;

        let output_norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.output_norm_config,
            &tree.subtree("output_norm"),
            context,
        )
        .await?;
        let mask_embedding = tree.read_slice::<bf16>("mask_embedding", &[model_dim]).await?;
        let mask_rows = upload::<B, bf16>(&mask_embedding.repeat(config.block_size as usize - 1), context)
            .map_err(DSparkNewError::Backend)?;

        let markov = MarkovHead {
            rank,
            token_embeddings,
            recurrence,
            fold,
            fold_bias,
            soft_cap: head_config.logit_soft_cap,
            start: <B::Kernels as Kernels>::MarkovStartKernel::new(context, data_type, rank, model_dim)
                .map_err(DSparkNewError::Backend)?,
            step: [false, true].map(|has_next| {
                <B::Kernels as Kernels>::MarkovStepKernel::new(context, data_type, rank, has_next)
                    .expect("Failed to create the Markov step kernel")
            }),
            logits: <B::Kernels as Kernels>::MarkovLogitsKernel::new(
                context,
                data_type,
                head_config.logit_soft_cap.is_some(),
            )
            .map_err(DSparkNewError::Backend)?,
            embed: <B::Kernels as Kernels>::MarkovEmbedKernel::new(context, data_type, rank)
                .map_err(DSparkNewError::Backend)?,
            fold_matmul: Mutex::new(
                <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                    .map_err(DSparkNewError::Matmul)?,
            ),
        };

        Ok(Self {
            layers: layers.into_boxed_slice(),
            tap_mix: <B::Kernels as Kernels>::TapMixNormKernel::new(context, data_type, model_dim, mix_count)
                .map_err(DSparkNewError::Backend)?,
            mix_weights: upload::<B, f32>(&mix_weights, context).map_err(DSparkNewError::Backend)?,
            mix_norm_scales: upload::<B, f32>(&mix_norm_scales, context).map_err(DSparkNewError::Backend)?,
            mix_epsilon,
            output_norm,
            mask_rows,
            markov,
            model_dim,
            block_size: config.block_size,
            max_context_length: *config.rope_config.max_sequence_length(),
            normalizes_anchor_embedding: config.normalizes_anchor_embedding,
            vocab_size: target_embedding.vocab_size(),
            data_type,
        })
    }

    /// Folds the target's LM head into the Markov head's output projection: the head's bias over the vocabulary is
    /// lm_head(output(latent)) = fold · latent + fold_bias. Runs the target's own readout on the projection's columns.
    async fn fold_markov_output(
        target: &Decoder<B>,
        output_tree: &ParameterTree<'_, B>,
        rank: u32,
        context: &B::Context,
    ) -> Result<(AffineMatrix<B>, B::GlobalGpuBuffer), DSparkNewError<B>> {
        let embedding = target.embedding();
        let (model_dim, vocab_size) = (embedding.model_dim() as usize, embedding.vocab_size() as usize);
        let rank_size = rank as usize;
        // [model_dim][rank] in the export: one readout row per rank column, then one for the bias.
        let weights = output_tree.subtree("weights").read_slice::<bf16>("weights", &[model_dim as u32, rank]).await?;
        let biases = output_tree.read_slice::<bf16>("biases", &[model_dim as u32]).await?;
        let mut rows = vec![bf16::ZERO; (rank_size + 1) * model_dim];
        for channel in 0..model_dim {
            for column in 0..rank_size {
                rows[column * model_dim + channel] = weights[channel * rank_size + column];
            }
            rows[rank_size * model_dim + channel] = biases[channel];
        }
        let rows = upload::<B, bf16>(&rows, context).map_err(DSparkNewError::Backend)?;

        let logit_bytes = (rank_size + 1) * vocab_size * DataType::BF16.size_in_bytes();
        let mut folded = context.create_buffer_gpu(logit_bytes).map_err(DSparkNewError::Backend)?;
        let mut command_buffer =
            context.create_command_buffer(Some("fold DSpark Markov head"), None).map_err(DSparkNewError::Backend)?;
        let logits =
            embedding.encode_readout(&rows, rank + 1, vocab_size as u32, None::<&B::ScratchBuffer>, false, &mut command_buffer)?;
        assert_eq!(logits.size(), logit_bytes, "the readout writes bfloat16 logits");
        command_buffer.encode_copy(&logits, &mut folded);
        command_buffer.end_encoding().submit().wait_until_completed().await.map_err(DSparkNewError::Backend)?;
        let folded = context.copyout::<bf16>(&folded).await.map_err(DSparkNewError::Backend)?;

        // The GEMV reads [vocab][rank] rows.
        let mut fold = vec![0.0f32; vocab_size * rank_size];
        for column in 0..rank_size {
            for (token, value) in folded[column * vocab_size..(column + 1) * vocab_size].iter().enumerate() {
                fold[token * rank_size + column] = value.to_f32();
            }
        }
        let fold_bias = &folded[rank_size * vocab_size..];
        Ok((
            quantize_affine::<B>(&fold, vocab_size, rank_size, context).map_err(DSparkNewError::Backend)?,
            upload::<B, bf16>(fold_bias, context).map_err(DSparkNewError::Backend)?,
        ))
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn empty_state(
        &self,
        context_capacity: u32,
        context: &B::Context,
    ) -> Result<DSparkState<B>, B::Error> {
        assert!(context_capacity <= self.max_context_length, "DSpark state capacity exceeds its RoPE capacity");
        let layer_states = self
            .layers
            .iter()
            .map(|layer| layer.layer.mixer.create_empty_state(Some(context_capacity), context))
            .collect::<Result<Box<[_]>, B::Error>>()?;
        Ok(DSparkState {
            layer_states,
            markov_context: context.create_buffer_gpu(size_for_shape(&[self.model_dim], self.data_type))?,
            context_length: 0,
        })
    }

    /// Projects the taps of the accepted rows into every layer's context and keeps the newest one's Markov mix.
    pub fn encode_accept(
        &self,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        accepted_indices: &[u32],
        state: &mut DSparkState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DSparkEncodeError<B>> {
        if accepted_indices.is_empty() {
            return Ok(());
        }
        let command_buffer = &mut *command_buffer.debug_group("dspark accept");
        let row_count = accepted_indices.len() as u32;
        assert!(state.context_length + row_count <= self.max_context_length, "DSpark context exceeds its RoPE capacity");

        let features = target_features.collect::<Vec<_>>();
        let &[tap0, tap1, tap2, tap3] = &features[..] else {
            panic!("DSpark reads {TAP_COUNT} taps, got {}", features.len());
        };
        let row_indices = command_buffer.allocate_constant_from_slice(accepted_indices).map_err(DSparkEncodeError::Backend)?;
        let mix_count = self.layers.len() as u32 + 1;
        let mut mixes = command_buffer
            .allocate_scratch_for_shape(&[mix_count, row_count, self.model_dim], self.data_type)
            .map_err(DSparkEncodeError::Backend)?;
        self.tap_mix.encode(
            tap0,
            tap1,
            tap2,
            tap3,
            &row_indices,
            &self.mix_weights,
            &self.mix_norm_scales,
            &mut mixes,
            row_count,
            self.mix_epsilon,
            command_buffer,
        );

        let mix_bytes = size_for_shape(&[row_count, self.model_dim], self.data_type);
        for (index, (layer, mixer_state)) in self.layers.iter().zip(state.layer_states.iter_mut()).enumerate() {
            let key_values = layer.context_kv_projection.encode_borrowed(&mixes, index * mix_bytes, row_count, command_buffer)?;
            mixer_state.prepare(state.context_length, row_count).map_err(DSparkEncodeError::Backend)?;
            let attention = (layer.layer.mixer.as_ref() as &dyn Any)
                .downcast_ref::<Attention<B>>()
                .expect("DSpark draft layers use attention");
            let attention_state = (mixer_state.as_mut() as &mut dyn Any)
                .downcast_mut::<AttentionState<B>>()
                .expect("DSpark draft layer states are attention states");
            attention
                .append_projected_kv(
                    &key_values,
                    row_count,
                    Some(&layer.context_key_norm_scales),
                    attention_state,
                    command_buffer,
                )
                .map_err(DSparkEncodeError::Backend)?;
        }

        let row_bytes = size_for_shape(&[self.model_dim], self.data_type);
        let newest = (mix_count as usize - 1) * mix_bytes + (row_count as usize - 1) * row_bytes;
        command_buffer.encode_copy(mixes.subrange(newest..newest + row_bytes), &mut state.markov_context);
        state.context_length += row_count;
        Ok(())
    }

    /// The draft block of `depth` rows: the anchor, then masks. Returns its normalized hidden states and the target
    /// LM head's logits of the rows after the anchor.
    pub fn encode_draft(
        &self,
        target: &Decoder<B>,
        anchor: impl BufferRef<Backend = B>,
        depth: u32,
        state: &mut DSparkState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DFlashOutput<B>, DSparkEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("dspark draft");
        assert!((2..=self.block_size).contains(&depth), "DSpark draft depth outside 2..=block size");
        assert!(state.context_length + depth <= self.max_context_length, "DSpark block exceeds its RoPE capacity");

        let anchor_embedding = target.embedding().encode_lookup(anchor, 1, command_buffer)?;
        let anchor_embedding = match (self.normalizes_anchor_embedding, target.embedding_norm()) {
            (true, Some(norm)) => norm
                .encode(&anchor_embedding, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DSparkEncodeError::Backend)?,
            _ => anchor_embedding,
        };
        let row_bytes = size_for_shape(&[self.model_dim], self.data_type);
        let mut block = command_buffer
            .allocate_scratch_for_shape(&[depth, self.model_dim], self.data_type)
            .map_err(DSparkEncodeError::Backend)?;
        command_buffer.encode_copy(&anchor_embedding, (&mut block).subrange_mut(..row_bytes));
        command_buffer.encode_copy(
            (&self.mask_rows).subrange(..(depth as usize - 1) * row_bytes),
            (&mut block).subrange_mut(row_bytes..),
        );

        let batch_topology = BatchTopology::flat(depth);
        let mut stream = block;
        for (layer, mixer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            mixer_state.prepare(state.context_length, depth).map_err(DSparkEncodeError::Backend)?;
            stream = layer.layer.encode(
                stream,
                None::<&B::ScratchBuffer>,
                &batch_topology,
                Some(MaybeMut::Mut(mixer_state.as_mut())),
                None,
                false,
                command_buffer,
            )?;
        }
        let draft_hidden = self
            .output_norm
            .encode(&stream, 0, depth, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(DSparkEncodeError::Backend)?;

        let proposal_rows = Range::from(row_bytes..depth as usize * row_bytes);
        let mut proposal_hidden =
            command_buffer.allocate_scratch(proposal_rows.iter().len()).map_err(DSparkEncodeError::Backend)?;
        command_buffer.encode_copy(draft_hidden.subrange(proposal_rows), &mut proposal_hidden);
        let logits = target.embedding().encode_readout(
            &proposal_hidden,
            depth - 1,
            self.vocab_size,
            None::<&B::ScratchBuffer>,
            false,
            command_buffer,
        )?;
        Ok(DFlashOutput {
            draft_hidden,
            logits,
        })
    }

    /// Proposes `chain_length` tokens after `anchor` left to right: each position adds the Markov head's bias to its
    /// logits and samples like the target, with `seeds[position]` when sampling is stochastic.
    pub fn encode_markov_chain(
        &self,
        draft: &DFlashOutput<B>,
        anchor: impl BufferRef<Backend = B>,
        chain_length: u32,
        state: &DSparkState<B>,
        sampling: &Sampling<B>,
        sampling_method: &SamplingMethod,
        seeds: Option<&[u64]>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::GlobalGpuBuffer, DSparkEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("dspark markov");
        let markov = &self.markov;
        let rank = markov.rank;
        let width = 2 * rank + 2 * self.model_dim;
        let element = self.data_type.size_in_bytes();
        let step_bytes = width as usize * element;
        let state_bytes = rank as usize * element;
        let logit_bytes = self.vocab_size as usize * element;

        let mut inputs = command_buffer
            .allocate_scratch_for_shape(&[chain_length, width], self.data_type)
            .map_err(DSparkEncodeError::Backend)?;
        markov.start.encode(
            anchor,
            &markov.token_embeddings,
            &draft.draft_hidden,
            &state.markov_context,
            &mut inputs,
            chain_length,
            command_buffer,
        );
        let mut latent =
            command_buffer.allocate_scratch_for_shape(&[rank], self.data_type).map_err(DSparkEncodeError::Backend)?;
        let mut bias = command_buffer
            .allocate_scratch_for_shape(&[self.vocab_size], self.data_type)
            .map_err(DSparkEncodeError::Backend)?;
        let mut tokens = command_buffer
            .context()
            .create_buffer_gpu(chain_length as usize * DataType::U32.size_in_bytes())
            .map_err(DSparkEncodeError::Backend)?;
        let batch_topology = BatchTopology::flat(1);

        for step in 0..chain_length as usize {
            let has_next = step + 1 < chain_length as usize;
            let projection = markov.recurrence.encode_borrowed(&inputs, step * step_bytes, 1, command_buffer)?;
            {
                let (current, next) = (&mut inputs).split_at((step + 1) * step_bytes);
                let step_state = current.as_ref().subrange(step * step_bytes..step * step_bytes + state_bytes);
                markov.step[usize::from(has_next)].encode(
                    &projection,
                    step_state,
                    has_next.then(|| next.subrange_mut(..state_bytes)),
                    &mut latent,
                    command_buffer,
                );
            }
            // Profiling aid: UZU_DSPARK_NO_MARKOV proposes from the LM head alone, with a zero Markov bias.
            if std::env::var_os("UZU_DSPARK_NO_MARKOV").is_some() {
                command_buffer.encode_fill(&mut bias, 0);
            } else {
            markov
                .fold_matmul
                .lock()
                .encode(
                    MatmulArguments {
                        a: MatmulA::full_precision(&latent),
                        b: MatmulB::Quantized(QuantizedB {
                            codes: &markov.fold.codes,
                            scales: &markov.fold.scales,
                            correction: QuantizedCorrection::Biases(&markov.fold.biases),
                            params: QuantParams::new(QuantParamsLayout::GroupOutput, markov.fold.rows, markov.fold.groups),
                            mode: QuantizationMode::U4,
                            group_size: FOLD_GROUP_SIZE as u32,
                            signed_codes: false,
                            weights_layout: WeightsLayout::<B>::default(),
                        }),
                        b_leading_dimension: None,
                        b_transpose: true,
                        output: MatmulOutput::new(&mut bias, MatmulDOps::none()),
                        gather: None::<MatmulGather<&B::GlobalGpuBuffer>>,
                        m: 1,
                        n: self.vocab_size,
                        k: rank,
                    },
                    command_buffer,
                )
                .map_err(DSparkEncodeError::Matmul)?;
            }
            let mut logits = command_buffer
                .allocate_scratch_for_shape(&[self.vocab_size], self.data_type)
                .map_err(DSparkEncodeError::Backend)?;
            markov.logits.encode(
                draft.logits.subrange(step * logit_bytes..(step + 1) * logit_bytes),
                &bias,
                &markov.fold_bias,
                &mut logits,
                self.vocab_size,
                markov.soft_cap,
                command_buffer,
            );
            let seed = seeds
                .map(|seeds| command_buffer.allocate_constant_from_slice(&seeds[step..step + 1]))
                .transpose()
                .map_err(DSparkEncodeError::Backend)?;
            let token = sampling
                .encode(
                    &logits,
                    seed.as_ref(),
                    None::<&B::ConstantBuffer>,
                    None::<&B::ConstantBuffer>,
                    None::<&B::GlobalGpuBuffer>,
                    &batch_topology,
                    (0..1).into(),
                    sampling_method,
                    command_buffer,
                )
                .map_err(DSparkEncodeError::Backend)?;
            let word = DataType::U32.size_in_bytes();
            command_buffer.encode_copy(&token, (&mut tokens).subrange_mut(step * word..(step + 1) * word));
            if has_next {
                let embedding = (step + 1) * step_bytes + state_bytes;
                markov.embed.encode(
                    &token,
                    &markov.token_embeddings,
                    (&mut inputs).subrange_mut(embedding..embedding + state_bytes),
                    command_buffer,
                );
            }
        }
        Ok(tokens)
    }
}
