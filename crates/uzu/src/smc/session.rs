//! `SmcSession` — Phase-0 plumbing spike for Sequential Monte Carlo
//! speculative decoding.
//!
//! Today this just loads a target + draft `ChatSession` pair and forwards
//! `generate` to the target while verifying the draft also loads. That gives
//! us the two-model ownership shape we need before doing the real work
//! (see `docs/smcsd/design.md` for what's still missing — logit exposure,
//! per-particle state, paged KV cache, batched rescore).
//!
//! DO NOT treat this as producing any speculative-decoding speedup yet.

use std::path::PathBuf;

use crate::{
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        types::{Error as SessionError, Input, Output},
    },
    smc::{config::SmcConfig, error::SmcError},
};

pub struct SmcSession {
    cfg: SmcConfig,
    target: ChatSession,
    /// Draft model. Held but not yet exercised for drafting in Phase 0.
    /// Kept as `Option` so we can briefly drop it to free memory during
    /// measurement runs where we want to compare against pure target decoding.
    draft: Option<ChatSession>,
}

impl SmcSession {
    /// Load both models. Fails fast if either path is missing or
    /// tokenizers/configs disagree.
    pub fn new(cfg: SmcConfig) -> Result<Self, SmcError> {
        if cfg.num_particles == 0 {
            return Err(SmcError::InvalidConfig(
                "num_particles must be >= 1".to_string(),
            ));
        }
        if cfg.gamma == 0 {
            return Err(SmcError::InvalidConfig(
                "gamma must be >= 1".to_string(),
            ));
        }
        if cfg.num_particles > 1 {
            // Phase 0 only handles N=1. Don't silently lie to the caller.
            return Err(SmcError::Unimplemented(
                "num_particles > 1 requires phase 2 (batched rescore)",
            ));
        }

        let decoding_config = DecodingConfig::default();

        let target = ChatSession::new(cfg.target_model_path.clone(), decoding_config.clone())
            .map_err(SmcError::TargetLoad)?;
        let draft = ChatSession::new(cfg.draft_model_path.clone(), decoding_config)
            .map_err(SmcError::DraftLoad)?;

        check_vocab_compatibility(&target, &draft)?;

        Ok(Self {
            cfg,
            target,
            draft: Some(draft),
        })
    }

    /// The draft model's path — so callers can assert the session picked up
    /// what they expected.
    pub fn draft_model_path(&self) -> &PathBuf {
        &self.cfg.draft_model_path
    }

    /// Currently: delegate to the target. This exists so the public API shape
    /// (`cfg -> new -> generate`) matches the eventual SMC-SD path even while
    /// the middle is a stub.
    pub fn generate<F>(
        &mut self,
        input: Input,
        run_cfg: RunConfig,
        progress: Option<F>,
    ) -> Result<Output, SessionError>
    where
        F: Fn(Output) -> bool,
    {
        self.target.run(input, run_cfg, progress)
    }

    /// Same as `generate`, but against the draft — useful for measuring the
    /// draft's standalone latency as a baseline.
    pub fn generate_draft_only<F>(
        &mut self,
        input: Input,
        run_cfg: RunConfig,
        progress: Option<F>,
    ) -> Result<Output, SessionError>
    where
        F: Fn(Output) -> bool,
    {
        let draft = self.draft.as_mut().ok_or(SessionError::LanguageModelGeneratorNotLoaded)?;
        draft.run(input, run_cfg, progress)
    }

    pub fn config(&self) -> &SmcConfig {
        &self.cfg
    }
}

fn check_vocab_compatibility(
    target: &ChatSession,
    draft: &ChatSession,
) -> Result<(), SmcError> {
    // We don't yet expose tokenizer vocabulary from ChatSession; do the
    // strictest check we can — model_config identity — and defer a real
    // vocab diff to Phase 1 when we expose per-session logit shapes.
    let t_vocab = target
        .model_metadata
        .model_config
        .as_language_model()
        .and_then(|lm| lm.decoder_config().ok())
        .map(|dc| dc.vocab_size);
    let d_vocab = draft
        .model_metadata
        .model_config
        .as_language_model()
        .and_then(|lm| lm.decoder_config().ok())
        .map(|dc| dc.vocab_size);
    match (t_vocab, d_vocab) {
        (Some(t), Some(d)) if t != d => {
            Err(SmcError::VocabMismatch {
                target: t,
                draft: d,
            })
        },
        _ => Ok(()),
    }
}
