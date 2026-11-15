//! Shared multimodal/VLM types.
//!
//! These used to live in `pipeline.rs` (MiniCPM), which forced LFM3-VL to
//! depend on the MiniCPM module. They are pipeline-agnostic: the tokenizer
//! sampler, streaming decoder, KV-state wrapper, multimodal message and reply
//! rendering helpers are used by both pipelines.

use std::range::Range;

use half::f16;
use image::GenericImageView;
use tokenizers::Tokenizer;

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu, GpuBuffer,
        },
        metal::Metal,
    },
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{PRng, SamplingMethod},
        transformer::TransformerState,
    },
    engine::language_model::LanguageModel,
    utils::size_for_shape::size_for_shape,
};

#[derive(Clone, Debug, PartialEq)]
pub struct MultimodalToolCall {
    pub name: String,
    /// Tool arguments as a JSON object string, the OpenAI wire form.
    pub arguments: String,
}

#[derive(Clone, Debug)]
pub struct MultimodalMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub images: Vec<image::DynamicImage>,
    /// Optional video as a pre-sampled frame sequence. Only LFM3-VL consumes
    /// these; the MiniCPM template has no video support and rejects them.
    pub videos: Vec<Vec<image::DynamicImage>>,
    /// Tool calls carried by an assistant message.
    pub tool_calls: Vec<MultimodalToolCall>,
    /// Identifier of the answered call on a tool-result message.
    pub tool_call_id: Option<String>,
}

impl MultimodalMessage {
    pub fn new(
        role: &str,
        content: Option<String>,
    ) -> Self {
        Self {
            role: role.to_string(),
            content,
            reasoning_content: None,
            images: Vec::new(),
            videos: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// The system-preamble header the chat template renders when tools are declared.
const TOOLS_PREAMBLE_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// The format contract appended after the declarations; the model must echo it verbatim.
const TOOLS_PREAMBLE_INSTRUCTIONS: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Builds the tool-declaration preamble that the chat template prepends to the system
/// message. `tool_declarations` are the tools as JSON strings, one per line, exactly the
/// objects the client sent.
pub fn tools_system_preamble(tool_declarations: &[String]) -> String {
    let mut preamble = String::from(TOOLS_PREAMBLE_HEADER);
    for declaration in tool_declarations {
        preamble.push('\n');
        preamble.push_str(declaration);
    }
    preamble.push_str(TOOLS_PREAMBLE_INSTRUCTIONS);
    preamble
}

/// Renders an assistant history message: the think block the template always emits,
/// the visible content, then any tool calls in the markup format.
pub fn render_assistant_text(msg: &MultimodalMessage) -> String {
    let content = msg.content.as_deref().unwrap_or_default();
    let reasoning = msg.reasoning_content.as_deref().map(str::trim).filter(|r| !r.is_empty());
    let mut out = match reasoning {
        Some(reasoning) => format!("<think>\n{reasoning}\n</think>\n\n"),
        None => "<think>\n\n</think>\n\n".to_string(),
    };
    out.push_str(content);
    for (index, call) in msg.tool_calls.iter().enumerate() {
        let lead = match index {
            0 if content.trim().is_empty() => format!("<tool_call>\n<function={}>\n", call.name),
            0 => format!("\n\n<tool_call>\n<function={}>\n", call.name),
            _ => format!("\n<tool_call>\n<function={}>\n", call.name),
        };
        out.push_str(&lead);
        for (key, value) in tool_call_parameter_pairs(&call.arguments) {
            out.push_str("<parameter=");
            out.push_str(&key);
            out.push_str(">\n");
            out.push_str(&value);
            out.push_str("\n</parameter>\n");
        }
        out.push_str("</function>\n</tool_call>");
    }
    out
}

/// Tool arguments JSON object -> ordered (key, rendered value) pairs. Values render the
/// way the chat template stringifies them: JSON for objects and arrays, raw text for scalars.
fn tool_call_parameter_pairs(arguments: &str) -> Vec<(String, String)> {
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return Vec::new();
    };
    map.into_iter()
        .map(|(key, value)| {
            let rendered = match &value {
                serde_json::Value::String(text) => text.clone(),
                serde_json::Value::Null => "null".to_string(),
                other => other.to_string(),
            };
            (key, rendered)
        })
        .collect()
}

/// Renders consecutive tool-result messages into the user turn the template emits:
/// one `<tool_response>` block per result, joined by newlines inside a single turn.
pub fn render_tool_response_text(contents: &[&str]) -> String {
    contents
        .iter()
        .map(|content| format!("<tool_response>\n{}\n</tool_response>", content.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Default)]
pub struct VlmSamplingConfig {
    pub temperature: Option<f32>,
    pub min_p: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub grammar: Option<crate::engine::language_model::grammar::Grammar>,
    /// Number of real tokenizer entries. Decoder logits may be wider (the
    /// checkpoint pads the vocabulary, e.g. 128000 vs 125031); ids at or above
    /// this limit are padding and must never be sampled.
    pub vocab_limit: Option<usize>,
}

impl std::fmt::Debug for VlmSamplingConfig {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("VlmSamplingConfig")
            .field("temperature", &self.temperature)
            .field("min_p", &self.min_p)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("seed", &self.seed)
            .field("has_grammar", &self.grammar.is_some())
            .field("vocab_limit", &self.vocab_limit)
            .finish()
    }
}

pub struct VlmSampler {
    config: VlmSamplingConfig,
    /// Base seed for the Philox RNG consumed by the GPU sampling kernel.
    seed: u64,
}

impl VlmSampler {
    pub fn set_vocab_limit(
        &mut self,
        limit: usize,
    ) {
        self.config.vocab_limit = Some(limit);
    }

    pub fn vocab_limit(&self) -> Option<usize> {
        self.config.vocab_limit
    }

    pub fn new(config: VlmSamplingConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random::<u64>);
        Self { config, seed }
    }

    /// Whether a repetition penalty is configured; the caller must provide a
    /// per-turn context ring when this is `Some`.
    pub fn repetition_penalty(&self) -> Option<f32> {
        self.config.repetition_penalty.filter(|penalty| *penalty > 1.0)
    }

    pub fn has_grammar(&self) -> bool {
        self.config.grammar.is_some()
    }

    /// The engine sampling method for this step. `suffix_repetition_length`
    /// must carry the context-ring capacity when a repetition penalty is set
    /// (the penalty kernel walks the ring); pass `None` otherwise.
    pub fn sampling_method(&self, suffix_repetition_length: Option<u32>) -> SamplingMethod {
        let temperature = self.config.temperature;
        if temperature.unwrap_or(0.0) <= 0.0 {
            return SamplingMethod::Greedy;
        }
        SamplingMethod::Stochastic {
            temperature: Some(temperature.unwrap().max(1e-5)),
            top_k: self.config.top_k.filter(|k| *k > 0).map(|k| k as u32),
            top_p: self.config.top_p,
            min_p: self.config.min_p,
            repetition_penalty: self.repetition_penalty(),
            suffix_repetition_length,
        }
    }

    /// Seed for this step's GPU sampling kernel, derived from the base seed
    /// and the sampled position (matches the text engine's `PRng::derive`).
    pub fn sampling_seed(&self, position: u64) -> u64 {
        PRng::new(self.seed).derive(position)
    }

    /// Grammar bitmask for this step; `false` when the grammar is absent or
    /// currently unconstrained, in which case the buffer content is garbage.
    pub fn grammar_bitmask(&mut self, bitmask: &mut [u32]) -> bool {
        match &mut self.config.grammar {
            Some(grammar) => grammar.next_bitmask(bitmask),
            None => false,
        }
    }

    /// Feeds a sampled token to the grammar (engagement/termination state).
    pub fn accept_token(&mut self, token: u64) {
        if let Some(grammar) = &mut self.config.grammar {
            let _ = grammar.accept_token(token);
        }
    }
}

/// Bitmask (1 = valid id) masking padded decoder vocabulary rows past the
/// tokenizer vocabulary; empty when the decoder width matches the tokenizer.
pub(crate) fn vocab_padding_bitmask(decoder_vocab: usize, vocab_limit: usize) -> Vec<u32> {
    if decoder_vocab <= vocab_limit {
        return Vec::new();
    }
    let words = decoder_vocab.div_ceil(32);
    let mut mask = vec![u32::MAX; words];
    if vocab_limit % 32 != 0 {
        mask[vocab_limit / 32] = (1u32 << (vocab_limit % 32)) - 1;
    }
    for word in mask.iter_mut().skip(vocab_limit.div_ceil(32)) {
        *word = 0;
    }
    mask
}

/// Creates an empty per-turn context ring for the GPU repetition penalty
/// kernel (`RingParams{offset, length}` header followed by token slots).
pub(crate) fn create_context_ring(
    context: &<Metal as Backend>::Context,
    suffix_repetition_length: u32,
) -> Result<<Metal as Backend>::GlobalGpuBuffer, anyhow::Error> {
    let mut ring = context
        .create_buffer_cpu_gpu(size_for_shape(&[2 + suffix_repetition_length], crate::data_type::DataType::U32))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    ring.copyin(&vec![0u32; 2 + suffix_repetition_length as usize]);
    ring.handoff().map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// Encodes one GPU sampling step for a VLM pipeline: builds the engine
/// sampling method from the sampler config, merges the pipeline's vocab
/// padding bitmask with the grammar bitmask, derives the Philox seed from the
/// sampled position, and dispatches the unified sampling kernel. Only the
/// token ids (4 bytes per sampled row) come back to the host.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_vlm_sampling(
    model: &LanguageModel<Metal>,
    command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    logits: impl BufferRef<Backend = Metal, Buffer: Sized>,
    token_ids: &impl GpuBuffer<Backend = Metal>,
    context_ring: Option<(&<Metal as Backend>::GlobalGpuBuffer, u32)>,
    batch_dim: &BatchTopology,
    sampling_range: Range<u32>,
    sampler: &mut VlmSampler,
    padding_bitmask: &[u32],
    position: u32,
) -> Result<<Metal as Backend>::GlobalGpuBuffer, anyhow::Error> {
    let suffix_repetition_length = context_ring.map(|(_, suffix)| suffix);
    let method = sampler.sampling_method(suffix_repetition_length);
    if method.suffix_repetition_length().is_some() != context_ring.is_some() {
        anyhow::bail!("repetition penalty sampling requires a context ring");
    }

    let seeds = if matches!(method, SamplingMethod::Stochastic { .. }) {
        Some(
            command_buffer
                .allocate_constant_from_slice(&[sampler.sampling_seed(position as u64)])
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
        )
    } else {
        None
    };

    let bitmask = if sampler.has_grammar() || !padding_bitmask.is_empty() {
        let mut words = padding_bitmask.to_vec();
        if words.is_empty() {
            words = vec![u32::MAX; model.vocab_size().div_ceil(32)];
        }
        if sampler.has_grammar() && sampler.grammar_bitmask(&mut words) {
            for (word, pad) in words.iter_mut().zip(padding_bitmask) {
                *word &= *pad;
            }
        }
        // A fully-open mask constrains nothing; skip the kernel variant.
        (words.iter().any(|&word| word != u32::MAX)).then(|| {
            command_buffer.allocate_constant_from_slice(&words).map_err(|e| anyhow::anyhow!("{e:?}"))
        }).transpose()?
    } else {
        None
    };

    model
        .sampling()
        .encode(
            logits,
            seeds.as_ref(),
            bitmask.as_ref(),
            context_ring.map(|(ring, _)| ring),
            Some(token_ids),
            batch_dim,
            sampling_range,
            &method,
            command_buffer,
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// Stateful streaming token decoder that completely eliminates UTF-8 multi-byte character corruption.
///
/// Why this exists: in BPE/SentencePiece tokenizers, a single Chinese/Unicode
/// character (typically 3 UTF-8 bytes) can be sliced across token boundaries
/// (e.g. 1 byte in token A, 2 bytes in token B). Calling
/// `tokenizer.decode(&[token_id])` on a single isolated token causes the
/// tokenizer to fail UTF-8 validation and emit Unicode replacement characters
/// `\u{FFFD}` (e.g. "水面波光").
///
/// This decoder wraps HuggingFace `tokenizers::step_decode_stream`, maintaining
/// cumulative prefix state and only emitting clean, complete UTF-8 strings as
/// soon as char boundaries are satisfied.
pub struct StreamingTokenDecoder<'a> {
    tokenizer: &'a Tokenizer,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    prefix_index: usize,
    skip_special_tokens: bool,
}

impl<'a> StreamingTokenDecoder<'a> {
    pub fn new(
        tokenizer: &'a Tokenizer,
        skip_special_tokens: bool,
    ) -> Self {
        Self {
            tokenizer,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            prefix_index: 0,
            skip_special_tokens,
        }
    }

    pub fn step(
        &mut self,
        token_id: u32,
    ) -> Result<Option<String>, anyhow::Error> {
        let chunk = tokenizers::step_decode_stream(
            self.tokenizer,
            vec![token_id],
            self.skip_special_tokens,
            &mut self.decode_ids,
            &mut self.decode_prefix,
            &mut self.prefix_index,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(chunk)
    }

    pub fn finish(&mut self) -> Result<Option<String>, anyhow::Error> {
        if self.decode_ids.is_empty() {
            return Ok(None);
        }
        let full =
            self.tokenizer.decode(&self.decode_ids, self.skip_special_tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
        if full.len() > self.decode_prefix.len() && full.starts_with(&self.decode_prefix) {
            let rem = &full[self.decode_prefix.len()..];
            let clean = rem.trim_matches('\u{FFFD}');
            if !clean.is_empty() {
                let out = clean.to_string();
                self.decode_prefix = full;
                return Ok(Some(out));
            }
        }
        Ok(None)
    }
}

pub struct VlmState {
    pub(crate) state: TransformerState<Metal>,
    /// Per-turn GPU ring of recent tokens; only present while a repetition
    /// penalty is configured (recreated by each prefill).
    pub(crate) context_ring: Option<(<Metal as Backend>::GlobalGpuBuffer, u32)>,
}

impl VlmState {
    pub fn context_length(&self) -> u32 {
        self.state.context_length
    }

    pub fn reset(&mut self) -> Result<(), anyhow::Error> {
        self.state.reset().map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

pub fn hash_image(img: &image::DynamicImage) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    img.dimensions().hash(&mut hasher);
    img.color().hash(&mut hasher);
    let bytes = img.as_bytes();
    bytes.len().hash(&mut hasher);
    // Hash u64 words: a collision here silently serves another image's cached
    // visual features, and a full hash is still dwarfed by the tower forward.
    let (words, rest) = bytes.as_chunks::<8>();
    for word in words {
        u64::from_le_bytes(*word).hash(&mut hasher);
    }
    rest.hash(&mut hasher);
    hasher.finish()
}

/// f32 -> f16 conversion over multiple threads; the patchifier input can be
/// ~18M values for a medium image, and the scalar loop is a visible fraction of
/// the tower time.
pub(crate) fn f32_to_f16_parallel(values: &[f32]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; values.len()];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    let chunk = values.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (src, dst) in values.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (destination, source) in dst.iter_mut().zip(src) {
                    *destination = f16::from_f32(*source);
                }
            });
        }
    });
    out
}
