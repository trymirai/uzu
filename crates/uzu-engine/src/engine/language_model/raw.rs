//! Raw autoregressive primitives over a [`LanguageModelState`]: prefill an
//! arbitrary token sequence and decode one token at a time, with no chat
//! template, history or reply parsing in between.
//!
//! These are the engine-level building blocks behind the server's raw
//! completion API (`POST /v1/completions`): the caller tokenizes their prompt
//! (or passes token ids directly), drives the loop, and observes exactly what
//! the model generates. Sampling reuses the vision stack's GPU sampler, which
//! is model-agnostic — it only consumes the decoder's logits.
//!
//! Limits vs. the chat paths: no speculator (MTP) support and no
//! repetition-penalty context ring — raw completions are a debugging/
//! evaluation surface, not a performance path. `VlmSampler` rejects or ignores
//! what it cannot honor; pass `repetition_penalty: None`.

use futures::executor::block_on;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    backends::metal::Metal,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::attention::ATTENTION_SUFFIX_CAPACITY,
    },
    engine::language_model::{LanguageModel, state::LanguageModelState},
    vision::common::{encode_vlm_sampling, vocab_padding_bitmask},
    vision::VlmSampler,
};

fn vocab_mask(model: &LanguageModel<Metal>, sampler: &VlmSampler) -> Vec<u32> {
    // Built per call: a ~8K-entry bitmask over the padded vocabulary, cheap
    // relative to a decode step; the pipeline's OnceLock cache only exists
    // because the shared pipeline serves many concurrent requests.
    vocab_padding_bitmask(model.vocab_size(), sampler.vocab_limit().unwrap_or(model.vocab_size()))
}

/// Prefills `new_tokens` onto the (empty or continuing) state and samples the
/// first generated token. No template is applied; `new_tokens` must already be
/// exactly what the caller wants in the KV cache.
pub fn raw_prefill(
    model: &LanguageModel<Metal>,
    state: &mut LanguageModelState<Metal>,
    new_tokens: &[u64],
    sampler: &mut VlmSampler,
) -> Result<u64, anyhow::Error> {
    if sampler.repetition_penalty().is_some() {
        anyhow::bail!("raw completions do not support repetition_penalty (no context ring on the raw state)");
    }
    let context = model.context();
    let suffix_len = new_tokens.len() as u32;
    if suffix_len == 0 {
        anyhow::bail!("cannot prefill empty tokens");
    }

    // The attention suffix is capped; all but the last chunk are committed to
    // the KV prefix via encode_accept, mirroring the text and vision paths.
    let max_batch = ATTENTION_SUFFIX_CAPACITY as usize;
    let batch_count = new_tokens.len().div_ceil(max_batch);
    state
        .transformer_state
        .prepare(
            state.transformer_state.context_length + ((batch_count - 1) * max_batch) as u32,
            usize::min(max_batch, new_tokens.len()) as u32,
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut command_buffer =
        context.create_command_buffer(Some("raw_prefill"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mut sampled_buf: Option<<Metal as Backend>::GlobalGpuBuffer> = None;
    let mut pos = 0usize;
    for batch in 0..batch_count {
        let is_last = batch + 1 == batch_count;
        let len = usize::min(max_batch, new_tokens.len() - pos);
        let input_u32: Vec<u32> = new_tokens[pos..pos + len].iter().map(|&x| x as u32).collect();
        let token_ids =
            command_buffer.allocate_constant_from_slice(&input_u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let batch_dim = BatchTopology::flat(len as u32, true);
        let output = model
            .decoder()
            .encode(
                &token_ids,
                &batch_dim,
                is_last.then(|| (len as u32 - 1..len as u32).into()),
                None,
                &mut state.transformer_state,
                &mut command_buffer,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        if is_last {
            let logits = output.logits.as_ref().expect("prefill produced no logits");
            let mask = vocab_mask(model, sampler);
            sampled_buf = Some(
                encode_vlm_sampling(
                    model,
                    &mut command_buffer,
                    logits,
                    &token_ids,
                    None,
                    &batch_dim,
                    (len as u32 - 1..len as u32).into(),
                    sampler,
                    &mask,
                    state.transformer_state.context_length - 1,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
            );
        }
        state
            .transformer_state
            .encode_accept(&(0..len as u32).collect::<Box<[u32]>>(), &mut command_buffer)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        drop(token_ids);
        pos += len;
    }
    let buf = sampled_buf.expect("prefill produced no sampled tokens");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let ids: Vec<u32> = block_on(context.copyout(&buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let first_token = ids.first().copied().unwrap_or(0) as u64;
    sampler.accept_token(first_token);
    Ok(first_token)
}

/// Decodes one token and samples the next. `token` is the previously sampled
/// (or last prefilled) token id; the returned id is the next generation step.
pub fn raw_decode_step(
    model: &LanguageModel<Metal>,
    state: &mut LanguageModelState<Metal>,
    token: u64,
    sampler: &mut VlmSampler,
) -> Result<u64, anyhow::Error> {
    if sampler.repetition_penalty().is_some() {
        anyhow::bail!("raw completions do not support repetition_penalty (no context ring on the raw state)");
    }
    let context = model.context();
    state.transformer_state.prepare(state.transformer_state.context_length, 1).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mut command_buffer =
        context.create_command_buffer(Some("raw_decode"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let next_u32 = [token as u32];
    let token_ids = command_buffer.allocate_constant_from_slice(&next_u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let batch_dim = BatchTopology::flat(1, true);

    let output = model
        .decoder()
        .encode(&token_ids, &batch_dim, Some((0..1).into()), None, &mut state.transformer_state, &mut command_buffer)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let logits = output.logits.as_ref().expect("decode produced no logits");
    let mask = vocab_mask(model, sampler);
    let sampled_buf = encode_vlm_sampling(
        model,
        &mut command_buffer,
        logits,
        &token_ids,
        None,
        &batch_dim,
        (0u32..1u32).into(),
        sampler,
        &mask,
        state.transformer_state.context_length,
    )
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    state.transformer_state.encode_accept(&[0], &mut command_buffer).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    drop(token_ids);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let ids: Vec<u32> = block_on(context.copyout(&sampled_buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let next_token = ids.first().copied().unwrap_or(0) as u64;
    sampler.accept_token(next_token);
    Ok(next_token)
}
