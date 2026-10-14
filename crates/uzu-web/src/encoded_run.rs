use hanashi::Encoding as _;
use js_sys::{Array, Function, Promise};
use serde::Serialize;
use serde_json::{Value, json};
use shoji::types::{
    basic::ToolDescription,
    session::chat::{ChatContentBlock, ChatMessage},
};
use uzu_engine::engine::language_model::stream::{LanguageModelStreamOptions, SamplingMethod};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::{
    chat::{Message, RunParams, sampling_method},
    encoding::{has_tool_candidates, sanitize},
    session::{Inner, LoadedModel, emit, error, monotonic_now, rate, yield_to_browser},
};

fn reply_items(
    reply: Option<&ChatMessage>,
    completed: bool,
    promote: bool,
) -> Vec<Value> {
    let Some(reply) = reply else {
        return vec![];
    };
    let mut items = vec![];
    let thinking = sanitize(&reply.reasoning().unwrap_or_default());
    let text = sanitize(&reply.text().unwrap_or_default());
    if !thinking.is_empty() {
        items.push(if promote && text.is_empty() {
            json!({"type":"text","text":thinking})
        } else {
            json!({"type":"thinking","text":thinking,"completed":completed || !text.is_empty()})
        });
    }
    if !text.is_empty() {
        items.push(json!({"type":"text","text":text}));
    }
    for call in reply.tool_calls() {
        let args: Value = serde_json::from_str(&call.arguments.json).unwrap_or(Value::Null);
        let hidden = args["hidden"] == true
            || args["hidden"].as_str().is_some_and(|value| value.trim().eq_ignore_ascii_case("true"));
        if call.name != "set_chat_name" || !hidden {
            items.push(json!({"type":"toolCall","name":call.name,"called":false}));
        }
    }
    items
}

fn publish(
    callback: &Function,
    previous: &mut Vec<Value>,
    next: Vec<Value>,
) -> Result<(), JsValue> {
    if *previous == next {
        return Ok(());
    }
    // Use the existing delta event when only one text field grew.
    let changes: Vec<_> = previous.iter().zip(&next).enumerate().filter(|(_, (a, b))| a != b).collect();
    if previous.len() == next.len() && changes.len() == 1 {
        let (index, (before, after)) = changes[0];
        if before["type"] == after["type"]
            && before["completed"] == after["completed"]
            && let (Some(a), Some(b)) = (before["text"].as_str(), after["text"].as_str())
            && let Some(delta) = b.strip_prefix(a)
        {
            emit(callback, json!({"type":"transcriptDelta","index":index,"delta":delta}))?;
            *previous = next;
            return Ok(());
        }
    }
    emit(callback, json!({"type":"transcript","items":next}))?;
    *previous = next;
    Ok(())
}

// Cancellation must also release a run while a tool promise is pending. The
// callback itself is user-supplied; it need not settle after the session stops.
async fn await_tool(
    inner: &Inner,
    value: JsValue,
) -> Result<JsValue, JsValue> {
    if inner.cancelled.get() {
        return Err(error("Tool execution cancelled"));
    }
    let cancelled = Promise::new(&mut |resolve, _reject| {
        *inner.tool_abort.borrow_mut() = Some(resolve);
    });
    let result = JsFuture::from(Promise::race(&Array::of2(&Promise::resolve(&value), &cancelled))).await;
    inner.tool_abort.borrow_mut().take();
    if inner.cancelled.get() {
        Err(error("Tool execution cancelled"))
    } else {
        result
    }
}

pub(crate) async fn run(
    inner: &Inner,
    loaded: &mut LoadedModel,
    params: RunParams,
    callback: &Function,
    tool_callback: Option<&Function>,
) -> Result<(), JsValue> {
    if params.messages.is_empty() {
        return Err(error("Add a message before generating a reply"));
    }
    if params.max_output_tokens.is_some_and(|limit| limit == 0 || limit > 65536) {
        return Err(error("maxOutputTokens must be between 1 and 65536"));
    }
    let support = inner.encoding_support.as_ref().unwrap();
    if !params.tools.is_empty() && !support.supports_tools {
        return Err(error("This model does not support tool calls"));
    }
    if !params.tools.is_empty() && tool_callback.is_none() {
        return Err(error("Tool execution callback is missing"));
    }
    let effort = support.reasoning.effective(params.reasoning_effort.as_deref());
    let promote = effort == Some(shoji::types::basic::ReasoningEffort::Disabled);
    let LoadedModel {
        model,
        state,
        encoded,
    } = loaded;
    let encoded = encoded.as_mut().unwrap();
    let mut input = encoded.prepare(&params, effort).map_err(error)?;
    let sampling = sampling_method(params.sampling_policy.as_ref(), model.default_sampling_method()).map_err(error)?;
    let start = monotonic_now();
    let mut prefill_seconds = 0.0;
    let mut input_count = 0;
    let mut count = 0;
    let mut transcript = Vec::new();
    let mut published = Vec::new();
    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut chat_name = None;
    let mut finish_reason;
    let mut grammar = params
        .grammar
        .as_ref()
        .map(|request| {
            if !params.tools.is_empty() {
                return Err(error("Select either structured output or tool calls for a run"));
            }
            let starts_thinking = inner
                .codec
                .thinking_opening_tag()
                .is_some_and(|tag| encoded.encoder.state().text().trim_end().ends_with(&tag));
            let trigger = if starts_thinking {
                inner
                    .codec
                    .thinking_tag()
                    .map(|tag| {
                        model
                            .tokenizer()
                            .encode(tag, false)
                            .map(|encoding| encoding.get_ids().iter().map(|&id| u64::from(id)).collect())
                            .map_err(error)
                    })
                    .transpose()?
            } else {
                None
            };
            let stops: Vec<i32> = model.generation_config().stop_token_ids.iter().map(|&id| id as i32).collect();
            uzu_engine::engine::language_model::grammar::Grammar::new(
                &request.config(),
                model.tokenizer(),
                trigger,
                Some(&stops),
            )
            .map_err(error)
        })
        .transpose()?;
    loop {
        if inner.cancelled.get() {
            finish_reason = "Cancelled";
            break;
        }
        if input.is_empty() {
            return Err(error("The chat template produced an empty prompt"));
        }
        if let Some(capacity) = inner.context_length
            && input.len() >= capacity as usize
        {
            if count == 0 {
                return Err(error(format!(
                    "Conversation is too long for this model's context window ({} tokens; {} available).",
                    input.len(),
                    capacity
                )));
            }
            finish_reason = "ContextLimitReached";
            break;
        }
        if state.as_ref().is_none_or(|state| !input.starts_with(state.tokens())) {
            *state = Some(model.create_empty_state(inner.context_length, inner.new_state_seed()).map_err(error)?);
        }
        let state = state.as_mut().unwrap();
        let mut offset = state.tokens().len();
        #[cfg(feature = "kernel-tests")]
        if count == 0 {
            inner.reused_tokens.set(offset as u32);
        }
        input_count += input.len() - offset;
        let prefill_start = monotonic_now();
        while input.len() - offset > 256 && !inner.cancelled.get() {
            let end = offset + 256;
            drop(
                model
                    .stream(
                        &input[offset..end],
                        state,
                        LanguageModelStreamOptions {
                            sampling_method: SamplingMethod::Greedy,
                            grammar: None,
                        },
                    )
                    .map_err(error)?,
            );
            offset = end;
            yield_to_browser().await;
        }
        finish_reason = "Cancelled";
        if !inner.cancelled.get() {
            let mut stream = model
                .stream(
                    &input[offset..],
                    state,
                    LanguageModelStreamOptions {
                        sampling_method: sampling.clone(),
                        grammar: grammar.take(),
                    },
                )
                .map_err(error)?;
            let mut turn_first = true;
            loop {
                if inner.cancelled.get() {
                    finish_reason = "Cancelled";
                    break;
                }
                if params.max_output_tokens.is_some_and(|limit| count >= limit) {
                    finish_reason = "Length";
                    break;
                }
                let Some(token) = stream.next() else {
                    finish_reason = "ContextLimitReached";
                    break;
                };
                let token = token.map_err(error)?;
                if turn_first {
                    prefill_seconds += (monotonic_now() - prefill_start).max(0.0) / 1000.0;
                    turn_first = false;
                }
                encoded.encoder.decode(vec![token as u32]).map_err(error)?;
                if model.generation_config().stop_token_ids.contains(&token) {
                    finish_reason = "Stop";
                    break;
                }
                count += 1;
                #[cfg(feature = "kernel-tests")]
                inner.last_tokens.borrow_mut().push(token as u32);
                let mut next = transcript.clone();
                next.extend(reply_items(encoded.reply(), false, promote && params.tools.is_empty()));
                publish(callback, &mut published, next)?;
                yield_to_browser().await;
            }
        }
        let Some(reply) = encoded.reply().cloned() else {
            break;
        };
        let is_tool = finish_reason == "Stop" && has_tool_candidates(&reply);
        let mut text = sanitize(&reply.text().unwrap_or_default());
        let mut thinking = sanitize(&reply.reasoning().unwrap_or_default());
        let promoted = promote && !is_tool && text.is_empty() && !thinking.is_empty();
        if promoted {
            text = std::mem::take(&mut thinking);
        }
        if !text.is_empty() {
            text_parts.push(text.clone());
        }
        if !thinking.is_empty() {
            thinking_parts.push(thinking.clone());
        }
        transcript.extend(reply_items(Some(&reply), true, promoted));
        if !is_tool {
            // Only completed visible replies are eligible for incremental replay.
            if finish_reason == "Stop" && !promoted && !text.is_empty() {
                encoded.visible = params.messages.clone();
                encoded.visible.push(Message {
                    role: "assistant".into(),
                    content: text_parts.join("\n\n"),
                    reasoning_content: (!thinking_parts.is_empty())
                        .then(|| thinking_parts.join("\n\n").trim().to_string()),
                });
            }
            break;
        }
        let calls = reply.tool_calls();
        if calls.is_empty()
            || reply.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }))
            || calls.iter().any(|call| {
                !params.tools.iter().any(|tool| match tool {
                    ToolDescription::Function {
                        tool_function,
                    } => tool_function.name == call.name,
                })
            })
        {
            finish_reason = "ToolCalls";
            break;
        }
        let mut all = encoded.encoder.state().messages.clone();
        for call in calls {
            if inner.cancelled.get() {
                break;
            }
            let known = params.tools.iter().any(|tool| match tool {
                ToolDescription::Function {
                    tool_function,
                } => tool_function.name == call.name,
            });
            let arguments: Value = serde_json::from_str(&call.arguments.json).map_err(error)?;
            let result = if known {
                let call_value = json!({"name":call.name,"arguments":arguments});
                let value = call_value.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(error)?;
                match tool_callback.unwrap().call1(&JsValue::UNDEFINED, &value) {
                    Ok(value) => match await_tool(inner, value).await {
                        Ok(value) => serde_wasm_bindgen::from_value(value).map_err(error)?,
                        Err(e) => json!({"failed":true,"value":{"error":format!("Tool execution failed: {e:?}")}}),
                    },
                    Err(e) => json!({"failed":true,"value":{"error":format!("Tool execution failed: {e:?}")}}),
                }
            } else {
                json!({"failed":true,"value":{"error":format!("Unknown or disabled tool: {}", call.name)}})
            };
            if inner.cancelled.get() {
                break;
            }
            if let Some(item) = transcript
                .iter_mut()
                .rev()
                .find(|item| item["type"] == "toolCall" && item["name"] == call.name && item["called"] == false)
            {
                item["called"] = json!(true);
                item["failed"] = json!(result["failed"] == true);
            }
            if let Some(name) = result["chatName"].as_str() {
                chat_name = Some(name.to_string());
                emit(callback, json!({"type":"chatName","name":name}))?;
            }
            if let Some(chart) = result.get("chart") {
                transcript.push(json!({"type":"chart","chart":chart}));
            }
            all.push(ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: call.identifier,
                name: Some(call.name),
                value: result.get("value").cloned().unwrap_or(Value::Null).into(),
            }));
        }
        publish(callback, &mut published, transcript.clone())?;
        if inner.cancelled.get() {
            finish_reason = "Cancelled";
            break;
        }
        if params.max_output_tokens.is_some_and(|limit| count >= limit) {
            finish_reason = "Length";
            break;
        }
        input = encoded.prepare_messages(all).map_err(error)?;
        yield_to_browser().await;
    }
    publish(callback, &mut published, transcript.clone())?;
    let end = monotonic_now();
    let total = (end - start).max(0.0) / 1000.0;
    let generation = (total - prefill_seconds).max(0.0);
    let mut done = json!({"type":"done","text":text_parts.join("\n\n"),"finishReason":finish_reason,
        "parsed":{"response":text_parts.join("\n\n"),"chainOfThought":thinking_parts.join("\n\n")},"transcript":transcript,
        "stats":{"prefillStats":{"duration":prefill_seconds,"tokensCount":input_count,"tokensPerSecond":rate(input_count as u32,prefill_seconds)},
        "generateStats":{"duration":generation,"tokensCount":count,"tokensPerSecond":rate(count,generation)},
        "totalStats":{"duration":total,"tokensCountInput":input_count,"tokensCountOutput":count}}});
    if let Some(name) = chat_name {
        done["chatName"] = json!(name);
    }
    emit(callback, done)
}
