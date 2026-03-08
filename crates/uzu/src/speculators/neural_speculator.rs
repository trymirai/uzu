use std::{collections::HashMap, fs::File, io::BufReader, path::Path, sync::Mutex};

use crate::{
    ModelMetadata,
    backends::common::Backend,
    language_model::language_model_generator::{LanguageModelGenerator, LanguageModelGeneratorTrait},
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        parameter::SamplingMethod,
        types::Error,
    },
    speculators::{empty_speculator::EmptySpeculator, speculator::Speculator},
    trie::TrieCreationConfig,
};

/// Neural speculative decoding draft model based on PARD.
///
/// A single forward pass over `[last, pard_token × n_pard]` yields n_pard+1 logit
/// distributions, so the main model can verify n_pard+1 candidates per step.
///
/// Distributions are cached keyed by the base prefix: when the trie descends the
/// trunk (calling `speculate` with prefix extended by one token per depth), the
/// cache returns the pre-computed distribution for that depth without an additional
/// GPU pass.
///
/// # Safety
/// `unsafe impl Send + Sync` is required because `B: Backend` does not carry
/// `Send + Sync` bounds. Interior mutability is handled by `Mutex`, so concurrent
/// calls to `prepare`/`speculate` are safe.
pub struct NeuralSpeculator<B: Backend> {
    draft: Mutex<LanguageModelGenerator<B>>,
    pard_token: u64,
    top_k: usize,
    /// Number of PARD placeholder tokens per step. The draft model produces
    /// n_pard+1 logit distributions: one for each of [last, pard×n_pard].
    n_pard: usize,
    /// Cache of the last PARD forward pass: (base_prefix, distributions).
    /// distributions[k] is the predicted distribution at trunk depth k.
    speculative_cache: Mutex<Option<(Vec<u64>, Vec<HashMap<u64, f32>>)>>,
}

unsafe impl<B: Backend> Send for NeuralSpeculator<B> {}
unsafe impl<B: Backend> Sync for NeuralSpeculator<B> {}

impl<B: Backend> NeuralSpeculator<B> {
    /// Load the PARD draft model from `draft_model_path`.
    ///
    /// `pard_token` is read from `model_config.pard_token` in the model's
    /// `config.json`. `n_pard` is the number of PARD placeholder tokens per step;
    /// the draft model produces n_pard+1 logit distributions per call.
    ///
    /// The draft model's generate buffer is sized for 2*n_pard+1 tokens to
    /// accommodate the combined KV-update pass: up to n_pard accepted tokens
    /// (new_context) + 1 last token + n_pard pard tokens.
    pub fn new(
        draft_model_path: &Path,
        n_pard: usize,
        top_k: usize,
    ) -> Result<Self, Error> {
        let config_path = draft_model_path.join("config.json");
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;
        let pard_token = model_metadata
            .model_config
            .as_language_model()
            .and_then(|lm| lm.model_config.pard_token)
            .ok_or(Error::UnsupportedSpeculatorConfigForModel)?;

        // The draft model itself uses EmptySpeculator. The generate-mode buffer is sized for
        // 2*n_pard+1 tokens (generate_suffix_length = 2*n_pard+1) to support the combined pass:
        // forward([accepted_1..k, last, pard×n_pard]) where k ≤ n_pard.
        let draft_speculator_config = SpeculatorConfig::new(2 * n_pard, std::sync::Arc::new(EmptySpeculator {}));

        let decoding_config =
            DecodingConfig::default().with_speculator_config(draft_speculator_config).with_allow_pre_encode(false); // one-shot calls don't benefit from pre-encode

        let draft = LanguageModelGenerator::new(draft_model_path, decoding_config)?;

        Ok(Self {
            draft: Mutex::new(draft),
            pard_token,
            top_k,
            n_pard,
            speculative_cache: Mutex::new(None),
        })
    }

    /// Run a single PARD forward pass for `prefix`, returning `n_pard+1` distributions.
    ///
    /// 1. Compute `accepted` = tokens not yet in the draft KV cache.
    /// 2. Run one GPU pass:
    ///    - `accepted` is empty → `get_multi_logits([last, pard×n_pard])`.
    ///    - `accepted.len() <= n_pard` → `extend_kv_and_get_logits(accepted, [last, pard×n_pard])`:
    ///      updates KV and gets logits in one round-trip.
    ///    - `accepted.len() > n_pard` (safety fallback) → `prefill(accepted)` + `get_multi_logits`.
    fn pard_forward(
        &self,
        prefix: &[u64],
    ) -> Vec<HashMap<u64, f32>> {
        let n = self.n_pard + 1;
        let mut draft = self.draft.lock().unwrap();

        let target_cached = &prefix[..prefix.len() - 1];
        let last_token = *prefix.last().unwrap();

        let reuse_len = draft.tokens.iter().zip(target_cached.iter()).take_while(|(a, b)| a == b).count();

        if reuse_len < draft.tokens.len() {
            draft.reset_state();
        }

        let cache_len = draft.tokens.len();
        let accepted = &target_cached[cache_len..];

        // extra_tokens = [last, pard×n_pard]
        let extra_tokens: Vec<u64> =
            std::iter::once(last_token).chain(std::iter::repeat(self.pard_token).take(n - 1)).collect();

        let mut logits = if accepted.is_empty() {
            draft.get_multi_logits(&extra_tokens, self.top_k)
        } else if accepted.len() <= self.n_pard {
            // Combined pass: update KV for accepted tokens and get logits for extra_tokens
            // in one GPU round-trip. Buffer (2*n_pard+1) fits: accepted.len() + extra_tokens.len()
            // ≤ n_pard + (n_pard+1) = 2*n_pard+1.
            draft.extend_kv_and_get_logits(accepted, &extra_tokens, self.top_k)
        } else {
            // Safety fallback: accepted.len() > n_pard should not occur in normal operation.
            let _ = draft.prefill(accepted.to_vec(), None, SamplingMethod::default(), cache_len, false);
            draft.get_multi_logits(&extra_tokens, self.top_k)
        };

        logits.resize_with(n, HashMap::new);
        logits
    }
}

impl<B: Backend> Speculator for NeuralSpeculator<B> {
    /// Runs the PARD forward pass for `prefix` and caches all `n_pard+1`
    /// distributions. Must be called once at the start of each trie build so
    /// that the cache is valid for the current generation step.
    fn prepare(
        &self,
        prefix: &[u64],
    ) {
        if prefix.is_empty() {
            *self.speculative_cache.lock().unwrap() = None;
            return;
        }
        let dists = self.pard_forward(prefix);
        *self.speculative_cache.lock().unwrap() = Some((prefix.to_vec(), dists));
    }

    /// Returns the cached distribution for `prefix`.
    ///
    /// Valid only after `prepare(base)` has been called for the
    /// current trie build. Returns the distribution at trunk depth
    /// `prefix.len() - base.len()`, or an empty map if the trunk horizon
    /// (`n_pard`) is exhausted (which stops trie growth naturally).
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        if prefix.is_empty() {
            return HashMap::new();
        }
        let cache = self.speculative_cache.lock().unwrap();
        if let Some((ref base, ref dists)) = *cache {
            if prefix.starts_with(base.as_slice()) {
                let depth = prefix.len() - base.len();
                return dists.get(depth).cloned().unwrap_or_default();
            }
        }
        HashMap::new()
    }

    /// PARD predicts tokens sequentially: dist[0] is for position 0 conditioned
    /// on the prefix, dist[1] for position 1 conditioned on the prefix + trunk[0],
    /// etc. The trie must be a trunk (linear chain) so the correct depth-k
    /// distribution is used for the k-th speculated token. width=1 enforces this.
    fn trie_creation_config(&self) -> TrieCreationConfig {
        TrieCreationConfig {
            width: 1,
        }
    }
}
