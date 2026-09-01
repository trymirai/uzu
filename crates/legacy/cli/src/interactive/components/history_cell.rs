use iocraft::prelude::*;
use shoji::types::session::{
    chat::{ChatReplyJoulesPerToken, ChatReplyStats},
    classification::ClassificationOutput,
    text_to_speech::TextToSpeechStats,
};

use crate::interactive::{
    components::ApplicationState,
    helpers::{SYMBOL_COMMAND, SYMBOL_INPUT, SYMBOL_INPUT_RESULT, SYMBOL_LONG_DASH},
};

#[derive(Clone)]
pub enum TranscriptItem {
    Thinking(String),
    ToolCall {
        name: String,
        called: bool,
    },
    Text(String),
}

#[derive(Clone)]
pub enum HistoryCellType {
    Command {
        name: String,
    },
    CommandResult {
        result: String,
    },
    Request {
        text: String,
    },
    ChatTranscript {
        items: Vec<TranscriptItem>,
        stats: Option<ChatReplyStats>,
    },
    ClassificationOutput {
        output: ClassificationOutput,
    },
    TextToSpeechOutput {
        stats: TextToSpeechStats,
    },
}

#[derive(Default, Props)]
pub struct HistoryCellProps {
    pub r#type: Option<HistoryCellType>,
    pub live: Option<bool>,
}

#[component]
pub fn HistoryCell(
    props: &HistoryCellProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let theme = state.read().theme().clone();

    let view: AnyElement<'static> = match props.r#type.clone() {
        Some(HistoryCellType::Command {
            name,
        }) => element! {
            View(flex_direction: FlexDirection::Column) {
                View(height: theme.padding())
                View(flex_direction: FlexDirection::Row, column_gap: theme.padding()) {
                    Text(content: SYMBOL_INPUT, color: theme.accent_color)
                    Text(content: format!("{}{}", SYMBOL_COMMAND, name), weight: Weight::Bold, color: theme.accent_color)
                }
            }
        }
        .into(),
        Some(HistoryCellType::CommandResult {
            result,
        }) => element! {
            View(flex_direction: FlexDirection::Row, column_gap: theme.padding(), padding_left: 2 * theme.padding()) {
                Text(content: SYMBOL_INPUT_RESULT, color: theme.subtitle_color)
                Text(content: result)
            }
        }
        .into(),
        Some(HistoryCellType::Request {
            text,
        }) => element! {
            View(flex_direction: FlexDirection::Column) {
                View(height: theme.padding())
                View(
                    width: 100pct,
                    background_color: Some(theme.overlay_color()),
                    flex_direction: FlexDirection::Row,
                    column_gap: theme.padding(),
                ) {
                    Text(content: SYMBOL_INPUT, weight: Weight::Bold, color: theme.subtitle_color)
                    Text(content: text)
                }
            }
        }
        .into(),
        Some(HistoryCellType::ChatTranscript {
            items,
            stats,
        }) => chat_transcript_component(
            items,
            stats,
            theme.subtitle_color,
            theme.overlay_color(),
            theme.padding(),
            props.live.unwrap_or(false),
        ),
        Some(HistoryCellType::ClassificationOutput {
            output,
        }) => classification_output_component(output, theme.subtitle_color, theme.padding()),
        Some(HistoryCellType::TextToSpeechOutput {
            stats,
        }) => text_to_speech_output_component(stats, theme.subtitle_color, theme.padding()),
        None => element! { View }.into(),
    };
    view
}

fn chat_transcript_component(
    items: Vec<TranscriptItem>,
    stats: Option<ChatReplyStats>,
    subtitle_color: Color,
    overlay_color: Color,
    padding: u16,
    live: bool,
) -> AnyElement<'static> {
    // consecutive tool calls form one block so there's no padding between them
    let mut blocks: Vec<TranscriptBlock> = Vec::new();
    for item in items {
        match item {
            TranscriptItem::ToolCall {
                name,
                called,
            } => match blocks.last_mut() {
                Some(TranscriptBlock::ToolGroup(calls)) => calls.push((name, called)),
                _ => blocks.push(TranscriptBlock::ToolGroup(vec![(name, called)])),
            },
            TranscriptItem::Thinking(content) => blocks.push(TranscriptBlock::Thinking(content)),
            TranscriptItem::Text(content) => blocks.push(TranscriptBlock::Text(content)),
        }
    }

    element! {
        View(
            width: 100pct,
            border_style: BorderStyle::Single,
            border_color: subtitle_color,
            flex_direction: FlexDirection::Column,
            row_gap: padding,
            padding_left: padding,
            padding_right: padding,
        ) {
            #(blocks.into_iter().map(|block| match block {
                TranscriptBlock::Thinking(content) => element! {
                    View(
                        width: 100pct,
                        background_color: Some(overlay_color),
                    ) {
                        Text(content: content, color: subtitle_color)
                    }
                }
                .into(),
                TranscriptBlock::ToolGroup(calls) => element! {
                    View(flex_direction: FlexDirection::Column) {
                        #(calls.into_iter().map(|(name, called)| {
                            tool_call_line(name, called, live, subtitle_color)
                        }).collect::<Vec<AnyElement<'static>>>())
                    }
                }
                .into(),
                TranscriptBlock::Text(content) => element! {
                    Text(content: content)
                }
                .into(),
            }).collect::<Vec<AnyElement<'static>>>())
            #(stats.as_ref().map(|stats| chat_reply_stats_component(stats, subtitle_color)))
        }
    }
    .into()
}

enum TranscriptBlock {
    Thinking(String),
    ToolGroup(Vec<(String, bool)>),
    Text(String),
}

fn tool_call_line(
    name: String,
    called: bool,
    live: bool,
    color: Color,
) -> AnyElement<'static> {
    if called {
        element! {
            Text(
                content: format!("* called {}", name),
                color: color,
            )
        }
        .into()
    } else if live {
        element! {
            CallingToolLine(name: name, color: color)
        }
        .into()
    } else {
        element! {
            Text(
                content: format!("* calling {}", name),
                color: color,
            )
        }
        .into()
    }
}

#[derive(Default, Props)]
struct CallingToolLineProps {
    name: String,
    color: Option<Color>,
}

#[component]
fn CallingToolLine(
    props: &CallingToolLineProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let mut frame = hooks.use_state(|| 0usize);
    hooks.use_future(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            *frame.write() += 1;
        }
    });
    // starts empty, then cycles one/two/three dots; padded so the line doesn't shift
    let frame = *frame.read();
    let dots = if frame == 0 {
        "   "
    } else {
        [".  ", ".. ", "..."][(frame - 1) % 3]
    };
    element! {
        Text(
            content: format!("* calling {}{}", props.name, dots),
            color: props.color,
        )
    }
}

fn classification_output_component(
    output: ClassificationOutput,
    subtitle_color: Color,
    padding: u16,
) -> AnyElement<'static> {
    let feature_scores: Vec<(String, f64)> = output.probabilities.values.into_iter().collect();
    let duration = format!("{:.2} s", output.stats.total_duration);

    element! {
        View(
            width: 100pct,
            border_style: BorderStyle::Single,
            border_color: subtitle_color,
            flex_direction: FlexDirection::Column,
            row_gap: padding,
            padding_left: padding,
            padding_right: padding,
        ) {
            View(
                width: 100pct,
                flex_direction: FlexDirection::Column,
            ) {
                #(feature_scores.into_iter().map(|(feature, score)| element! {
                    Text(content: format!("{feature}: {score:.4}"))
                }))
            }
            View(
                width: 100pct,
                flex_direction: FlexDirection::Column,
            ) {
                Text(
                    content: format!("duration: {duration}"),
                    color: subtitle_color,
                )
            }
        }
    }
    .into()
}

fn chat_reply_stats_component(
    stats: &ChatReplyStats,
    subtitle_color: Color,
) -> AnyElement<'static> {
    let time_to_first_token = stats
        .time_to_first_token
        .map(|duration| format!("{duration:.2} s"))
        .unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let generation_speed = stats
        .generate_tokens_per_second
        .map(|tokens_per_second| format!("{tokens_per_second:.2} t/s"))
        .unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let prefill_speed = stats
        .prefill_tokens_per_second
        .map(|tokens_per_second| format!("{tokens_per_second:.2} t/s"))
        .unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let tokens_per_forward_pass = stats
        .speculator_stats
        .as_ref()
        .map(|speculator_stats| format!("{:.2} t/f", speculator_stats.tokens_per_forward_pass))
        .unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let memory_used = stats.memory_used_bytes.map(format_memory_used).unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let energy =
        stats.total_joules().map(|energy| format!("{energy:.2} J")).unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let input_energy_per_token =
        stats.input_joules_per_token().map(format_joules_per_token).unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let output_energy_per_token =
        stats.output_joules_per_token().map(format_joules_per_token).unwrap_or_else(|| SYMBOL_LONG_DASH.to_string());
    let duration = format!("{:.2} s", stats.duration);

    element! {
        View(
            width: 100pct,
            flex_direction: FlexDirection::Column,
        ) {
            Text(
                content: format!("time to first token: {time_to_first_token}"),
                color: subtitle_color,
            )
            Text(
                content: format!("prefill speed: {prefill_speed}"),
                color: subtitle_color,
            )
            Text(
                content: format!("generation speed: {generation_speed}"),
                color: subtitle_color,
            )
            Text(
                content: format!("tokens per forward pass: {tokens_per_forward_pass}"),
                color: subtitle_color,
            )
            Text(
                content: format!("memory used: {memory_used}"),
                color: subtitle_color,
            )
            Text(
                content: format!("total energy: {energy}"),
                color: subtitle_color,
            )
            Text(
                content: format!("input energy per token: {input_energy_per_token}"),
                color: subtitle_color,
            )
            Text(
                content: format!("output energy per token: {output_energy_per_token}"),
                color: subtitle_color,
            )
            Text(
                content: format!("duration: {duration}"),
                color: subtitle_color,
            )
        }
    }
    .into()
}

fn format_joules_per_token(stats: ChatReplyJoulesPerToken) -> String {
    let total = stats.total();
    match stats {
        ChatReplyJoulesPerToken::Total {
            ..
        } => format!("{total:.3} J/tok"),
        ChatReplyJoulesPerToken::Components {
            cpu,
            gpu,
            ane,
            dram,
        } => format!("CPU {cpu:.3}, GPU {gpu:.3}, ANE {ane:.3}, DRAM {dram:.3}, total {total:.3} J/tok"),
    }
}

fn format_memory_used(bytes: i64) -> String {
    let gibibytes = bytes.max(0) as f64 / 1024.0 / 1024.0 / 1024.0;
    format!("{gibibytes:.2} GB")
}

fn text_to_speech_output_component(
    stats: TextToSpeechStats,
    subtitle_color: Color,
    padding: u16,
) -> AnyElement<'static> {
    let text_length = stats.text_length;
    let first_chunk_seconds = format!("{:.2} s", stats.first_chunk_seconds);
    let generation_duration = format!("{:.2} s", stats.generation_duration);
    let audio_duration = format!("{:.2} s", stats.audio_duration);
    let real_time_factor = format!("{:.2}x", stats.real_time_factor());

    element! {
        View(
            width: 100pct,
            border_style: BorderStyle::Single,
            border_color: subtitle_color,
            flex_direction: FlexDirection::Column,
            row_gap: padding,
            padding_left: padding,
            padding_right: padding,
        ) {
            View(
                width: 100pct,
                flex_direction: FlexDirection::Column,
            ) {
                Text(content: format!("text length: {text_length}"), color: subtitle_color)
                Text(content: format!("first chunk: {first_chunk_seconds}"), color: subtitle_color)
                Text(content: format!("generation duration: {generation_duration}"), color: subtitle_color)
                Text(content: format!("audio duration: {audio_duration}"), color: subtitle_color)
                Text(content: format!("real-time factor: {real_time_factor}"), color: subtitle_color)
            }
        }
    }
    .into()
}
