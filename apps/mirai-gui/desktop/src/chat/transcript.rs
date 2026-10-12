use uzu::types::{
    basic::ToolCall,
    session::chat::{ChatContentBlock, ChatMessage, ChatReply, ChatReplyFinishReason, ChatRole},
};

use super::{
    chart::ChartSpec,
    messages::sanitize,
    payloads::{RunEvent, TranscriptItem},
};

/// Only this run's messages: replayed history belongs to earlier UI messages.
/// Result events arrive before the next prefill, so completion never waits for
/// another generated token or the session's inference lock.
#[derive(Default)]
pub(super) struct Transcript {
    messages: Vec<ChatMessage>,
    finish_reason: Option<ChatReplyFinishReason>,
    promote_reasoning: bool,
    stream_snapshot_valid: bool,
    last_item_index: Option<usize>,
    reply_dirty: bool,
}

impl Transcript {
    /// Append to the last visible item while the current message keeps its
    /// structure. The caller owns the latest reply; materialize it only when a
    /// snapshot or final/error transcript needs the complete message history.
    pub(super) fn stream_update(
        &mut self,
        previous: Option<&ChatReply>,
        reply: &ChatReply,
        promote_reasoning: bool,
    ) -> Option<RunEvent> {
        if self.stream_snapshot_valid
            && self.promote_reasoning == promote_reasoning
            && reply.finish_reason.is_none()
            && let Some(previous) = previous.filter(|previous| previous.finish_reason.is_none())
            && let Some(delta) = visible_delta(&previous.message, &reply.message, promote_reasoning)
            && (delta.is_empty() || self.last_item_index.is_some())
        {
            self.reply_dirty = true;
            return (!delta.is_empty()).then(|| RunEvent::TranscriptDelta {
                index: self.last_item_index.expect("visible text item"),
                delta,
            });
        }

        self.update(reply);
        self.promote_reasoning = promote_reasoning;
        let items = self.items(false);
        self.last_item_index = items.len().checked_sub(1);
        self.stream_snapshot_valid = true;
        Some(RunEvent::Transcript {
            items,
        })
    }

    pub(super) fn flush_reply(
        &mut self,
        reply: Option<&ChatReply>,
    ) {
        if self.reply_dirty
            && let Some(reply) = reply
        {
            self.update(reply);
        }
    }

    pub(super) fn update(
        &mut self,
        reply: &ChatReply,
    ) {
        if self.finish_reason.is_some() || self.messages.is_empty() {
            self.messages.push(reply.message.clone());
        } else {
            *self.messages.last_mut().expect("current assistant message") = reply.message.clone();
        }
        self.finish_reason = reply.finish_reason.clone();
        self.reply_dirty = false;
        self.stream_snapshot_valid = false;
    }

    pub(super) fn tool_results(
        &mut self,
        messages: Vec<ChatMessage>,
    ) {
        self.messages.extend(messages);
        self.finish_reason = Some(ChatReplyFinishReason::ToolCalls);
        self.stream_snapshot_valid = false;
    }

    pub(super) fn set_reasoning_promotion(
        &mut self,
        enabled: bool,
    ) {
        if self.promote_reasoning != enabled {
            self.promote_reasoning = enabled;
            self.stream_snapshot_valid = false;
        }
    }

    pub(super) fn items(
        &self,
        finished: bool,
    ) -> Vec<TranscriptItem> {
        let last_assistant = self.messages.iter().rposition(|message| message.role == (ChatRole::Assistant {}));
        let mut items = Vec::new();
        let mut pending: Vec<(&ToolCall, Option<usize>)> = Vec::new();
        for (message_index, message) in self.messages.iter().enumerate() {
            if message.role == (ChatRole::Assistant {}) {
                let last = Some(message_index) == last_assistant;
                for block in &message.content {
                    match block {
                        ChatContentBlock::Reasoning {
                            value,
                        } if !value.trim().is_empty() => {
                            if last && self.promote_reasoning {
                                complete_thinking(&mut items);
                                items.push(TranscriptItem::Text {
                                    text: sanitize(value),
                                });
                            } else {
                                let text = sanitize(value.trim());
                                if let Some(TranscriptItem::Thinking {
                                    text: previous,
                                    ..
                                }) = items.last_mut()
                                {
                                    previous.push_str("\n\n");
                                    previous.push_str(&text);
                                } else {
                                    items.push(TranscriptItem::Thinking {
                                        text,
                                        completed: false,
                                    });
                                }
                            }
                        },
                        ChatContentBlock::Text {
                            value,
                        } if !value.trim().is_empty() => {
                            complete_thinking(&mut items);
                            items.push(TranscriptItem::Text {
                                text: sanitize(value),
                            });
                        },
                        ChatContentBlock::ToolCall {
                            value,
                        } => {
                            let item_index = if hidden_call(value) {
                                None
                            } else {
                                complete_thinking(&mut items);
                                let index = items.len();
                                items.push(TranscriptItem::ToolCall {
                                    name: value.name.clone(),
                                    called: false,
                                    failed: false,
                                });
                                Some(index)
                            };
                            // Hidden calls still consume their own results.
                            pending.push((value, item_index));
                        },
                        _ => {},
                    }
                }
            } else if message.role == (ChatRole::Tool {}) {
                for block in &message.content {
                    let ChatContentBlock::ToolCallResult {
                        identifier,
                        name,
                        value,
                    } = block
                    else {
                        continue;
                    };
                    // IDs take precedence. When a template omits IDs, consume
                    // just one matching name in call order, never every call.
                    let exact = identifier
                        .as_ref()
                        .and_then(|id| pending.iter().position(|(call, _)| call.identifier.as_ref() == Some(id)));
                    let matched = exact.or_else(|| {
                        let name = name.as_ref()?;
                        pending.iter().position(|(call, _)| {
                            (identifier.is_none() || call.identifier.is_none()) && &call.name == name
                        })
                    });
                    if let Some(index) = matched {
                        let (call, item_index) = pending.remove(index);
                        if let Some(index) = item_index {
                            let result = serde_json::from_str::<serde_json::Value>(&value.json).ok();
                            let failed = result.as_ref().is_some_and(|value| value.get("error").is_some());
                            let chart = if call.name == "show_chart" && !failed {
                                result
                                    .and_then(|value| serde_json::from_value::<ChartSpec>(value).ok())
                                    .filter(|chart| chart.validate().is_ok())
                            } else {
                                None
                            };
                            items[index] = match chart {
                                Some(chart) => TranscriptItem::Chart {
                                    chart,
                                },
                                None => TranscriptItem::ToolCall {
                                    name: call.name.clone(),
                                    called: true,
                                    failed: failed || call.name == "show_chart",
                                },
                            };
                        }
                    }
                }
            }
        }
        // ToolCalls starts another assistant turn. Hidden housekeeping creates
        // no visible boundary: leave thinking open until content resumes or
        // the whole run ends, including cancellation and errors.
        if finished || self.finish_reason.as_ref().is_some_and(|reason| *reason != ChatReplyFinishReason::ToolCalls) {
            complete_thinking(&mut items);
        }
        items
    }
}

/// Return only an appended suffix, or require a snapshot if any visible block
/// changes shape. Invisible candidates and housekeeping calls do not interrupt
/// streaming. A decoder correction (including a replaced partial Unicode
/// character) falls back to a snapshot rather than using byte offsets blindly.
fn visible_delta(
    previous: &ChatMessage,
    current: &ChatMessage,
    promote_reasoning: bool,
) -> Option<String> {
    if previous.role != (ChatRole::Assistant {})
        || current.role != previous.role
        || current.metadata != previous.metadata
    {
        return None;
    }
    let visible = |block: &&ChatContentBlock| match block {
        ChatContentBlock::Reasoning {
            value,
        }
        | ChatContentBlock::Text {
            value,
        } => !value.trim().is_empty(),
        ChatContentBlock::ToolCall {
            value,
        } => !hidden_call(value),
        _ => false,
    };
    let mut previous = previous.content.iter().filter(visible).peekable();
    let mut current = current.content.iter().filter(visible).peekable();
    while let Some(before) = previous.next() {
        let after = current.next()?;
        let last = previous.peek().is_none();
        if last != current.peek().is_none() {
            return None;
        }
        if before == after {
            continue;
        }
        if !last {
            return None;
        }
        let (before, after) = match (before, after) {
            (
                ChatContentBlock::Text {
                    value: before,
                },
                ChatContentBlock::Text {
                    value: after,
                },
            ) => (before.as_str(), after.as_str()),
            (
                ChatContentBlock::Reasoning {
                    value: before,
                },
                ChatContentBlock::Reasoning {
                    value: after,
                },
            ) => {
                if promote_reasoning {
                    (before.as_str(), after.as_str())
                } else {
                    (before.trim(), after.trim())
                }
            },
            _ => return None,
        };
        return after.strip_prefix(before).map(sanitize);
    }
    current.next().is_none().then(String::new)
}

fn complete_thinking(items: &mut [TranscriptItem]) {
    if let Some(TranscriptItem::Thinking {
        completed,
        ..
    }) = items.last_mut()
    {
        *completed = true;
    }
}

fn hidden_call(call: &ToolCall) -> bool {
    if call.name != "set_chat_name" {
        return false;
    }
    let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.arguments.json) else {
        return false;
    };
    // Match ToolDescriptor's boolean coercion for markup-based templates.
    match arguments.get("hidden") {
        Some(serde_json::Value::Bool(value)) => *value,
        Some(serde_json::Value::String(value)) => value.trim().eq_ignore_ascii_case("true"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use uzu::types::{basic::Value, session::chat::ChatReplyFinishReason};

    use super::*;

    fn call(
        name: &str,
        id: Option<&str>,
    ) -> ToolCall {
        ToolCall {
            identifier: id.map(str::to_string),
            name: name.to_string(),
            arguments: Value::from(serde_json::json!({})),
        }
    }

    fn naming_call(hidden: serde_json::Value) -> ToolCall {
        ToolCall {
            arguments: serde_json::json!({ "name": "A name", "hidden": hidden }).into(),
            ..call("set_chat_name", None)
        }
    }

    fn result(
        name: Option<&str>,
        id: Option<&str>,
    ) -> ChatMessage {
        ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: id.map(str::to_string),
            name: name.map(str::to_string),
            value: Value::from(serde_json::json!("result")),
        })
    }

    fn reply(
        message: ChatMessage,
        finished: bool,
    ) -> ChatReply {
        ChatReply {
            message,
            stats: Default::default(),
            finish_reason: finished.then_some(ChatReplyFinishReason::ToolCalls),
        }
    }

    #[derive(Default)]
    struct StreamOracle {
        transcript: Transcript,
        oracle: Transcript,
        displayed: Vec<TranscriptItem>,
        previous: Option<ChatReply>,
        wire_bytes: usize,
    }

    impl StreamOracle {
        fn send(
            &mut self,
            reply: ChatReply,
            promote_reasoning: bool,
        ) -> Option<serde_json::Value> {
            let event = self.transcript.stream_update(self.previous.as_ref(), &reply, promote_reasoning);
            self.oracle.update(&reply);
            self.oracle.set_reasoning_promotion(promote_reasoning);
            let serialized = event.as_ref().map(|event| {
                let json = serde_json::to_string(event).unwrap();
                self.wire_bytes += json.len();
                serde_json::from_str::<serde_json::Value>(&json).unwrap()
            });
            match event {
                Some(RunEvent::Transcript {
                    items,
                }) => self.displayed = items,
                Some(RunEvent::TranscriptDelta {
                    index,
                    delta,
                }) => match &mut self.displayed[index] {
                    TranscriptItem::Text {
                        text,
                    }
                    | TranscriptItem::Thinking {
                        text,
                        ..
                    } => text.push_str(&delta),
                    item => panic!("delta targets a non-text item: {item:?}"),
                },
                None => {},
                _ => panic!("unexpected transcript event"),
            }
            assert_eq!(self.displayed, self.oracle.items(false));
            self.previous = Some(reply);
            serialized
        }

        fn tool_results(
            &mut self,
            results: Vec<ChatMessage>,
        ) {
            self.transcript.flush_reply(self.previous.as_ref());
            self.transcript.tool_results(results.clone());
            self.oracle.tool_results(results);
            self.displayed = self.transcript.items(false);
            assert_eq!(self.displayed, self.oracle.items(false));
        }

        fn finish(&mut self) {
            self.transcript.flush_reply(self.previous.as_ref());
            assert_eq!(self.transcript.items(true), self.oracle.items(true));
        }
    }

    #[test]
    fn stream_deltas_preserve_whitespace_unicode_and_decoder_corrections() {
        for reasoning in [false, true] {
            let mut stream = StreamOracle::default();
            for text in [
                "",
                "  ",
                "  A",
                "  A ",
                "  A \n",
                "  A \nB",
                "  A \nB�",
                "  A \nB🙂",
                "  A \nB🙂\u{200d}",
                "  A \nB🙂\u{200d}↔\u{fe0f}",
                "  A \nB🙂\u{200d}↔\u{fe0f}\u{0000}",
                "Replaced",
                "R",
            ] {
                let message = if reasoning {
                    ChatMessage::assistant().with_reasoning(text.into())
                } else {
                    ChatMessage::assistant().with_text(text.into())
                };
                stream.send(reply(message, false), false);
            }
            stream.finish();
        }
    }

    #[test]
    fn growing_text_keeps_the_owned_reply_only_at_snapshot_boundaries() {
        let mut stream = StreamOracle::default();
        let initial = ChatMessage::assistant().with_text("Hello".into());
        stream.send(reply(initial.clone(), false), false);
        let event = stream.send(reply(ChatMessage::assistant().with_text("Hello world".into()), false), false);
        assert_eq!(event.unwrap(), serde_json::json!({ "type": "transcriptDelta", "index": 0, "delta": " world" }));
        assert_eq!(stream.transcript.messages, [initial]);
        assert!(stream.transcript.reply_dirty);
        stream.finish();
        assert!(!stream.transcript.reply_dirty);
    }

    #[test]
    fn hidden_naming_candidates_and_merged_reasoning_stream_without_extra_snapshots() {
        let mut stream = StreamOracle::default();
        let initial = ChatMessage::assistant().with_reasoning("Before".into());
        stream.send(reply(initial.clone(), false), false);
        for candidate in ["<tool_call>", "<tool_call>set_chat_name"] {
            assert!(
                stream
                    .send(
                        reply(
                            initial.with_block(ChatContentBlock::ToolCallCandidate {
                                value: serde_json::json!(candidate).into(),
                            }),
                            false,
                        ),
                        false,
                    )
                    .is_none()
            );
        }
        stream.send(reply(initial.with_tool_call(naming_call(serde_json::json!(true))), true), false);
        stream.tool_results(vec![result(Some("set_chat_name"), None)]);
        stream.send(reply(ChatMessage::assistant().with_reasoning("After ".into()), false), false);
        let event = stream.send(reply(ChatMessage::assistant().with_reasoning("After naming".into()), false), false);
        assert_eq!(event.unwrap(), serde_json::json!({ "type": "transcriptDelta", "index": 0, "delta": " naming" }));
        assert_eq!(
            stream.displayed,
            [TranscriptItem::Thinking {
                text: "Before\n\nAfter naming".into(),
                completed: false,
            }]
        );
        stream.send(
            reply(ChatMessage::assistant().with_reasoning("After naming".into()).with_text("Answer".into()), false),
            false,
        );
        stream.finish();
    }

    #[test]
    fn structural_changes_promotion_and_terminal_metadata_require_snapshots() {
        let mut stream = StreamOracle::default();
        let message = ChatMessage::assistant().with_reasoning("  Answer ".into());
        stream.send(reply(message.clone(), false), false);
        let promoted = stream.send(reply(message, false), true).unwrap();
        assert_eq!(promoted["type"], "transcript");
        let message = ChatMessage::assistant().with_reasoning("  Answer now ".into());
        assert_eq!(stream.send(reply(message.clone(), false), true).unwrap()["type"], "transcriptDelta");
        assert_eq!(stream.send(reply(message, false), false).unwrap()["type"], "transcript");
        let message = ChatMessage::assistant().with_reasoning("  Answer now ".into()).with_text("Visible".into());
        assert_eq!(stream.send(reply(message, false), false).unwrap()["type"], "transcript");
        let mut message = ChatMessage::assistant().with_reasoning("Corrected".into()).with_text("Visible text".into());
        assert_eq!(stream.send(reply(message.clone(), false), false).unwrap()["type"], "transcript");
        message.metadata.values.insert("test".into(), serde_json::json!(true).into());
        assert_eq!(stream.send(reply(message.clone(), false), false).unwrap()["type"], "transcript");
        let mut terminal = reply(message, false);
        terminal.finish_reason = Some(ChatReplyFinishReason::Cancelled);
        assert_eq!(stream.send(terminal, false).unwrap()["type"], "transcript");
        stream.finish();

        // Cancellation and errors may arrive without a final reply. Flush the
        // latest streamed suffix before completing the current thinking item.
        for promote in [false, true] {
            let mut stream = StreamOracle::default();
            stream.send(reply(ChatMessage::assistant().with_reasoning("Partial".into()), false), promote);
            stream.send(reply(ChatMessage::assistant().with_reasoning("Partial answer".into()), false), promote);
            stream.finish();
        }
    }

    #[test]
    fn serialized_stream_grows_linearly_and_deltas_exclude_prior_charts() {
        let measure = |length: usize| {
            let mut stream = StreamOracle::default();
            stream.send(reply(ChatMessage::assistant().with_tool_call(call("show_chart", Some("chart"))), true), false);
            stream.tool_results(vec![
                ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                    identifier: Some("chart".into()),
                    name: Some("show_chart".into()),
                    value: serde_json::json!({
                        "type": "bar", "title": "Previously displayed chart", "labels": ["A"],
                        "datasets": [{ "label": "Values", "data": [2] }],
                    })
                    .into(),
                }),
            ]);
            assert!(matches!(stream.displayed[0], TranscriptItem::Chart { .. }));
            let mut text = String::new();
            for index in 0..length {
                text.push('x');
                let event = stream.send(reply(ChatMessage::assistant().with_text(text.clone()), false), false).unwrap();
                if index > 0 {
                    assert_eq!(event, serde_json::json!({ "type": "transcriptDelta", "index": 1, "delta": "x" }));
                }
            }
            stream.finish();
            stream.wire_bytes
        };
        let shorter = measure(512);
        let longer = measure(1024);
        assert!(longer < 2 * shorter);
        assert!(longer < 100 * 1024, "serialized {longer} bytes for 1024 generated characters");
    }

    #[test]
    fn chronological_transcript_completes_tools_before_another_assistant_token() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Need time".into()), false));
        assert_eq!(
            transcript.items(false),
            [TranscriptItem::Thinking {
                text: "Need time".into(),
                completed: false
            }]
        );
        transcript.update(&reply(
            ChatMessage::assistant()
                .with_reasoning("Need time".into())
                .with_text("Checking.".into())
                .with_tool_call(call("get_current_date_time", Some("date"))),
            true,
        ));
        let initial = transcript.items(false);
        assert_eq!(
            initial[0],
            TranscriptItem::Thinking {
                text: "Need time".into(),
                completed: true
            }
        );
        assert_eq!(
            initial[1],
            TranscriptItem::Text {
                text: "Checking.".into()
            }
        );
        assert_eq!(
            initial[2],
            TranscriptItem::ToolCall {
                name: "get_current_date_time".into(),
                called: false,
                failed: false
            }
        );
        transcript.tool_results(vec![result(Some("get_current_date_time"), Some("date"))]);
        assert_eq!(
            transcript.items(false)[2],
            TranscriptItem::ToolCall {
                name: "get_current_date_time".into(),
                called: true,
                failed: false
            }
        );
        transcript.update(&reply(ChatMessage::assistant().with_text("It is noon.".into()), false));
        let items = transcript.items(true);
        assert_eq!(items.len(), 4);
        assert_eq!(
            items[3],
            TranscriptItem::Text {
                text: "It is noon.".into()
            }
        );
    }

    #[test]
    fn hidden_naming_bridges_thinking_without_collapsing_between_turns() {
        let mut transcript = Transcript::default();
        let message = ChatMessage::assistant().with_reasoning("A name".into()).with_text("\n".into());
        transcript.update(&reply(message.with_tool_call_candidate(serde_json::json!("partial call").into()), false));
        let thinking = |text: &str, completed| {
            vec![TranscriptItem::Thinking {
                text: text.into(),
                completed,
            }]
        };
        assert_eq!(transcript.items(false), thinking("A name", false));
        transcript.update(&reply(message.with_tool_call(naming_call(serde_json::json!(true))), true));
        assert_eq!(transcript.items(false), thinking("A name", false));
        transcript.tool_results(vec![result(Some("set_chat_name"), None)]);
        assert_eq!(transcript.items(false), thinking("A name", false));
        assert_eq!(transcript.items(true), thinking("A name", true));
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Answer".into()), false));
        assert_eq!(transcript.items(false), thinking("A name\n\nAnswer", false));
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Answer reasoning".into()), false));
        assert_eq!(transcript.items(false), thinking("A name\n\nAnswer reasoning", false));
        transcript.update(&reply(
            ChatMessage::assistant().with_reasoning("Answer reasoning".into()).with_text("The answer.".into()),
            false,
        ));
        assert_eq!(
            transcript.items(false),
            [
                TranscriptItem::Thinking {
                    text: "A name\n\nAnswer reasoning".into(),
                    completed: true
                },
                TranscriptItem::Text {
                    text: "The answer.".into()
                }
            ]
        );
    }

    #[test]
    fn explicit_naming_stays_visible_and_separates_thinking() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(
            ChatMessage::assistant()
                .with_reasoning("Requested rename".into())
                .with_tool_call(naming_call(serde_json::json!(false))),
            true,
        ));
        transcript.tool_results(vec![result(Some("set_chat_name"), None)]);
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Now answer".into()), false));
        assert_eq!(
            transcript.items(false),
            [
                TranscriptItem::Thinking {
                    text: "Requested rename".into(),
                    completed: true
                },
                TranscriptItem::ToolCall {
                    name: "set_chat_name".into(),
                    called: true,
                    failed: false
                },
                TranscriptItem::Thinking {
                    text: "Now answer".into(),
                    completed: false
                },
            ]
        );
    }

    #[test]
    fn only_true_hidden_values_hide_naming_calls() {
        let calls = [
            call("set_chat_name", None),
            naming_call(serde_json::json!(false)),
            naming_call(serde_json::json!("false")),
            naming_call(serde_json::json!("maybe")),
            naming_call(serde_json::json!(1)),
            naming_call(serde_json::Value::Null),
            ToolCall {
                arguments: Value {
                    json: "not json".into(),
                },
                ..call("set_chat_name", None)
            },
            ToolCall {
                arguments: serde_json::json!({ "hidden": true }).into(),
                ..call("get_current_date_time", None)
            },
        ];
        for call in calls {
            assert!(!hidden_call(&call));
            let mut transcript = Transcript::default();
            transcript.update(&reply(ChatMessage::assistant().with_tool_call(call), true));
            assert!(matches!(
                transcript.items(false).as_slice(),
                [TranscriptItem::ToolCall {
                    called: false,
                    ..
                }]
            ));
        }
        assert!(hidden_call(&naming_call(serde_json::json!(true))));
        assert!(hidden_call(&naming_call(serde_json::json!(" TRUE "))));
    }

    #[test]
    fn terminal_assistant_reason_closes_thinking_without_visible_output() {
        let mut transcript = Transcript::default();
        let mut final_reply = reply(ChatMessage::assistant().with_reasoning("Finished".into()), false);
        final_reply.finish_reason = Some(ChatReplyFinishReason::Stop);
        transcript.update(&final_reply);
        assert_eq!(
            transcript.items(false),
            [TranscriptItem::Thinking {
                text: "Finished".into(),
                completed: true
            }]
        );
    }

    #[test]
    fn results_match_one_call_by_id_or_name_and_never_complete_unrelated_calls() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(
            ChatMessage::assistant()
                .with_tool_call(naming_call(serde_json::json!(true)))
                .with_tool_call(call("get_current_date_time", None))
                .with_tool_call(call("get_current_date_time", None))
                .with_tool_call(call("get_current_date_time", Some("final"))),
            true,
        ));
        transcript.tool_results(vec![result(Some("set_chat_name"), None), result(Some("get_current_date_time"), None)]);
        let called = |transcript: &Transcript| {
            transcript
                .items(false)
                .iter()
                .map(|item| match item {
                    TranscriptItem::ToolCall {
                        called,
                        ..
                    } => *called,
                    _ => panic!("only tool calls"),
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(called(&transcript), [true, false, false]);
        transcript.tool_results(vec![result(None, Some("final"))]);
        assert_eq!(called(&transcript), [true, false, true]);
        transcript.tool_results(vec![result(None, None)]);
        assert_eq!(called(&transcript), [true, false, true]);
        transcript.tool_results(vec![result(Some("get_current_date_time"), None)]);
        assert_eq!(called(&transcript), [true, true, true]);
    }

    #[test]
    fn mismatched_ids_do_not_fall_back_to_a_shared_name() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(ChatMessage::assistant().with_tool_call(call("date", Some("a"))), true));
        transcript.tool_results(vec![result(Some("date"), Some("b"))]);
        assert_eq!(
            transcript.items(false),
            [TranscriptItem::ToolCall {
                name: "date".into(),
                called: false,
                failed: false
            }]
        );
    }

    #[test]
    fn tool_errors_mark_only_the_matching_call_as_failed() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(
            ChatMessage::assistant()
                .with_tool_call(call("set_chat_name", Some("failed")))
                .with_tool_call(call("set_chat_name", Some("succeeded")))
                .with_tool_call(call("set_chat_name", Some("pending"))),
            true,
        ));
        transcript.tool_results(vec![
            result(Some("set_chat_name"), Some("succeeded")),
            ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: Some("failed".into()),
                name: Some("set_chat_name".into()),
                value: serde_json::json!({ "error": "Chat names must contain 3 to 50 characters" }).into(),
            }),
        ]);
        let items = transcript.items(false);
        assert_eq!(
            items,
            [
                TranscriptItem::ToolCall {
                    name: "set_chat_name".into(),
                    called: true,
                    failed: true,
                },
                TranscriptItem::ToolCall {
                    name: "set_chat_name".into(),
                    called: true,
                    failed: false,
                },
                TranscriptItem::ToolCall {
                    name: "set_chat_name".into(),
                    called: false,
                    failed: false,
                },
            ]
        );
        let serialized = serde_json::to_value(items).unwrap();
        assert_eq!(serialized[0]["failed"], true);
        assert!(serialized[1].get("failed").is_none());
        assert!(serialized[2].get("failed").is_none());
    }

    #[tokio::test]
    async fn completed_charts_replace_the_matching_call_in_transcript_order() {
        use uzu::session::tool::func_def::ToolDescriptor;

        let chart = |title: &str| {
            serde_json::json!({
                "type": "bar", "title": title, "labels": ["A"],
                "datasets": [{ "label": "Values", "data": [2] }],
            })
        };
        let first = chart("First chart");
        let second = chart("Second chart");
        let first_call = ToolCall {
            // The transcript must use the canonical result, including markup argument coercion.
            arguments: serde_json::json!({ "chart": first.to_string() }).into(),
            ..call("show_chart", Some("first"))
        };
        let second_call = ToolCall {
            arguments: serde_json::json!({ "chart": second }).into(),
            ..call("show_chart", Some("second"))
        };
        let tool: ToolDescriptor = super::super::chart::show_chart.into();
        let first_result = tool.execute(first_call.arguments.clone()).await.unwrap();
        let second_result = tool.execute(second_call.arguments.clone()).await.unwrap();
        let chart_result = |id: &str, value| {
            ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: Some(id.into()),
                name: Some("show_chart".into()),
                value,
            })
        };
        let mut transcript = Transcript::default();
        transcript.update(&reply(
            ChatMessage::assistant()
                .with_reasoning("Compare the data".into())
                .with_tool_call(first_call)
                .with_tool_call(call("get_current_date_time", Some("date")))
                .with_tool_call(second_call)
                .with_text("The comparison.".into()),
            true,
        ));
        transcript.tool_results(vec![chart_result("second", second_result)]);
        let items = transcript.items(false);
        assert!(matches!(
            items[1],
            TranscriptItem::ToolCall {
                called: false,
                ..
            }
        ));
        assert!(matches!(&items[3], TranscriptItem::Chart { chart } if chart.title == "Second chart"));
        transcript.tool_results(vec![
            chart_result("first", first_result),
            result(Some("get_current_date_time"), Some("date")),
        ]);
        let items = transcript.items(true);
        assert_eq!(items.len(), 5);
        assert!(matches!(
            items[0],
            TranscriptItem::Thinking {
                completed: true,
                ..
            }
        ));
        assert!(matches!(&items[1], TranscriptItem::Chart { chart } if chart.title == "First chart"));
        assert!(matches!(
            items[2],
            TranscriptItem::ToolCall {
                called: true,
                failed: false,
                ..
            }
        ));
        assert!(matches!(&items[3], TranscriptItem::Chart { chart } if chart.title == "Second chart"));
        assert!(matches!(&items[4], TranscriptItem::Text { text } if text == "The comparison."));
        let serialized = serde_json::to_value(&items[1]).unwrap();
        assert_eq!(serialized["type"], "chart");
        assert_eq!(serialized["chart"]["title"], "First chart");
    }

    #[test]
    fn chart_errors_and_invalid_results_remain_failed_tool_calls() {
        for value in [
            serde_json::json!({ "error": "Invalid chart" }),
            serde_json::json!("Chart displayed"),
            serde_json::json!({
                "type": "bar", "title": "Invalid", "labels": ["A", "B"],
                "datasets": [{ "label": "Values", "data": [1] }],
            }),
            serde_json::json!({
                "type": "bar", "title": "Invalid", "labels": ["A"],
                "datasets": [{ "label": "Values", "data": [1] }],
                "options": { "onClick": "alert(1)" },
            }),
        ] {
            let mut transcript = Transcript::default();
            transcript.update(&reply(ChatMessage::assistant().with_tool_call(call("show_chart", None)), true));
            transcript.tool_results(vec![ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: None,
                name: Some("show_chart".into()),
                value: value.into(),
            })]);
            assert_eq!(
                transcript.items(true),
                [TranscriptItem::ToolCall {
                    name: "show_chart".into(),
                    called: true,
                    failed: true,
                }]
            );
        }
    }

    #[test]
    fn promotion_only_changes_the_current_assistant_turn_and_finishing_keeps_partial_activity() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(
            ChatMessage::assistant().with_reasoning("Find time".into()).with_tool_call(call("date", None)),
            true,
        ));
        transcript.tool_results(vec![result(Some("date"), None)]);
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Actual answer".into()), false));
        transcript.set_reasoning_promotion(true);
        let items = transcript.items(true);
        assert_eq!(
            items[0],
            TranscriptItem::Thinking {
                text: "Find time".into(),
                completed: true
            }
        );
        assert_eq!(
            items[2],
            TranscriptItem::Text {
                text: "Actual answer".into()
            }
        );
        transcript.set_reasoning_promotion(false);
        assert_eq!(
            transcript.items(true)[2],
            TranscriptItem::Thinking {
                text: "Actual answer".into(),
                completed: true
            }
        );
    }

    #[test]
    fn error_keeps_reasoning_that_was_already_displayed_as_an_answer() {
        let mut transcript = Transcript::default();
        transcript.update(&reply(ChatMessage::assistant().with_reasoning("Partial answer".into()), false));
        transcript.set_reasoning_promotion(true);
        let displayed = transcript.items(false);
        assert_eq!(
            displayed,
            [TranscriptItem::Text {
                text: "Partial answer".into()
            }]
        );
        // An error finishes the existing view without reclassifying its text.
        assert_eq!(transcript.items(true), displayed);
    }
}
