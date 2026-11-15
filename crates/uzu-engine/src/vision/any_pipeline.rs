//! Model-agnostic VLM facade for the CLI.
//!
//! MiniCPM uses the canvas/siglip pipeline in `pipeline.rs` (3D mRoPE), while
//! LFM3-VL has its own Qwen3VL-style processor/merger and plain 1D RoPE. The
//! two pipelines share `VlmState` and `VlmSampler`, so a thin enum is enough
//! to drive either from one code path.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::{
    backends::metal::Metal,
    engine::language_model::LanguageModel,
    vision::{
        Lfm3PreparedInput, Lfm3VlPipeline, MultimodalMessage, PreparedMultimodalInput, VlmPipeline, VlmSampler,
        VlmState,
    },
};

/// Prompt tokens and visual features, whichever processor produced them.
pub enum AnyPreparedInput {
    MiniCpm(PreparedMultimodalInput),
    Lfm3(Lfm3PreparedInput),
}

impl AnyPreparedInput {
    pub fn prompt_tokens(&self) -> &[u64] {
        match self {
            Self::MiniCpm(input) => &input.prompt_tokens,
            Self::Lfm3(input) => &input.prompt_tokens,
        }
    }
}

pub enum AnyVlmPipeline {
    MiniCpm(VlmPipeline),
    Lfm3(Lfm3VlPipeline),
}

impl AnyVlmPipeline {
    /// Picks the pipeline from `vision_config.json` (`type` starts with
    /// `Lfm3Vl`), falling back to the MiniCPM canvas pipeline.
    pub fn load_from_model_dir(
        model_dir: &Path,
        tokenizer: &Tokenizer,
    ) -> Result<Self, anyhow::Error> {
        if Lfm3VlPipeline::is_lfm3_vl(model_dir) {
            Ok(Self::Lfm3(Lfm3VlPipeline::load_from_model_dir(model_dir, tokenizer)?))
        } else {
            Ok(Self::MiniCpm(VlmPipeline::load_from_model_dir(model_dir)?))
        }
    }

    pub fn is_lfm3_vl(&self) -> bool {
        matches!(self, Self::Lfm3(_))
    }

    /// Vision encoder label for CLI logs (`Metal Vision Tower` / `CPU vision tower`).
    pub fn vision_backend(&self) -> &'static str {
        match self {
            Self::MiniCpm(_) => "metal",
            Self::Lfm3(pipeline) => pipeline.vision_backend(),
        }
    }

    pub fn prepare_inputs(
        &self,
        images: &[image::DynamicImage],
        prompt_text: &str,
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<AnyPreparedInput, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => Ok(AnyPreparedInput::MiniCpm(pipeline.prepare_inputs(
                images,
                prompt_text,
                tokenizer,
                enable_thinking,
            )?)),
            Self::Lfm3(pipeline) => {
                Ok(AnyPreparedInput::Lfm3(pipeline.prepare_inputs(images, prompt_text, tokenizer, enable_thinking)?))
            },
        }
    }

    /// Encodes the images referenced by `messages` and renders the prompt.
    /// `tools_preamble` is only supported by the MiniCPM pipeline for now.
    pub fn prepare_conversation_with_tools(
        &self,
        messages: &[MultimodalMessage],
        tokenizer: &Tokenizer,
        enable_thinking: bool,
        tools_preamble: Option<&str>,
    ) -> Result<AnyPreparedInput, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => Ok(AnyPreparedInput::MiniCpm(
                pipeline.prepare_conversation_with_tools(messages, tokenizer, enable_thinking, tools_preamble)?,
            )),
            Self::Lfm3(pipeline) => Ok(AnyPreparedInput::Lfm3(pipeline.prepare_conversation_with_tools(
                messages,
                tokenizer,
                enable_thinking,
                tools_preamble,
            )?)),
        }
    }

    /// Renders tool declarations in the pipeline's chat-template format.
    /// `declarations` are OpenAI tool objects as JSON strings.
    pub fn tools_preamble(
        &self,
        declarations: &[String],
    ) -> String {
        match self {
            Self::MiniCpm(_) => crate::vision::tools_system_preamble(declarations),
            Self::Lfm3(pipeline) => pipeline.tools_block(declarations),
        }
    }

    /// CLI/server entry point for a single video (frame list) + prompt.
    pub fn prepare_video_inputs(
        &self,
        frames: &[image::DynamicImage],
        prompt_text: &str,
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<AnyPreparedInput, anyhow::Error> {
        match self {
            Self::MiniCpm(_) => anyhow::bail!("MiniCPM-V does not support video input"),
            Self::Lfm3(pipeline) => Ok(AnyPreparedInput::Lfm3(
                pipeline.prepare_video_inputs(frames, prompt_text, tokenizer, enable_thinking)?,
            )),
        }
    }

    pub fn acquire_state(
        &self,
        model: &LanguageModel<Metal>,
        max_context_tokens: Option<u32>,
    ) -> Result<VlmState, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => pipeline.acquire_state(model, max_context_tokens),
            Self::Lfm3(pipeline) => pipeline.acquire_state(model, max_context_tokens),
        }
    }

    pub fn prefill(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &AnyPreparedInput,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        match (self, prepared) {
            (Self::MiniCpm(pipeline), AnyPreparedInput::MiniCpm(prepared)) => {
                pipeline.prefill_with_state(model, state, prepared, sampler)
            },
            (Self::Lfm3(pipeline), AnyPreparedInput::Lfm3(prepared)) => {
                pipeline.prefill(model, state, prepared, sampler)
            },
            _ => anyhow::bail!("prepared input does not belong to this pipeline"),
        }
    }

    /// Server-facing alias for `prefill`.
    pub fn prefill_with_state(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &AnyPreparedInput,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        self.prefill(model, state, prepared, sampler)
    }

    /// Incremental text-only prefill for raw completions: `new_tokens` go
    /// onto the state verbatim (no template), returning the first sampled
    /// token. Forwarded per pipeline, like `prefill` above.
    pub fn prefill_incremental_text(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        new_tokens: &[u64],
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => pipeline.prefill_incremental_text(model, state, new_tokens, sampler),
            Self::Lfm3(pipeline) => pipeline.prefill_incremental_text(model, state, new_tokens, sampler),
        }
    }

    /// Tokenizes the next user turn on top of an existing session state.
    pub fn format_followup_tokens(
        &self,
        prompt_text: &str,
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<Vec<u64>, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => pipeline.format_followup_tokens(prompt_text, tokenizer, enable_thinking),
            Self::Lfm3(pipeline) => pipeline.format_followup_tokens(prompt_text, tokenizer, enable_thinking),
        }
    }

    pub fn decode_step_with_state(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        token: u64,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => pipeline.decode_step_with_state(model, state, token, sampler),
            Self::Lfm3(pipeline) => pipeline.decode_step_with_state(model, state, token, sampler),
        }
    }

    pub fn prefill_followup_turn(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prompt_text: &str,
        tokenizer: &Tokenizer,
        sampler: &mut VlmSampler,
        enable_thinking: bool,
    ) -> Result<u64, anyhow::Error> {
        match self {
            Self::MiniCpm(pipeline) => {
                pipeline.prefill_followup_turn(model, state, prompt_text, tokenizer, sampler, enable_thinking)
            },
            Self::Lfm3(pipeline) => {
                pipeline.prefill_followup_turn(model, state, prompt_text, tokenizer, sampler, enable_thinking)
            },
        }
    }

    pub fn release_state(
        &self,
        state: VlmState,
    ) {
        match self {
            Self::MiniCpm(pipeline) => pipeline.release_state(state),
            Self::Lfm3(pipeline) => pipeline.release_state(state),
        }
    }
}
