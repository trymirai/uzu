use iocraft::prelude::*;
use shoji::types::session::chat::{ChatReply, ChatReplyStats};

use crate::cli::{
    components::ApplicationState,
    helpers::{SYMBOL_COMMAND, SYMBOL_INPUT, SYMBOL_INPUT_RESULT, SYMBOL_LONG_DASH},
};

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
    ChatReply {
        reply: ChatReply,
    },
}

#[derive(Default, Props)]
pub struct HistoryCellProps {
    pub r#type: Option<HistoryCellType>,
}

#[component]
pub fn HistoryCell(
    props: &HistoryCellProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let theme = state.read().theme.clone();

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
        Some(HistoryCellType::ChatReply {
            reply,
        }) => {
            chat_reply_component(reply, theme.subtitle_color, theme.overlay_color(), theme.padding())
                .into()
        },
        None => element! { View }.into(),
    };
    view
}

fn chat_reply_component(
    reply: ChatReply,
    subtitle_color: Color,
    overlay_color: Color,
    padding: u16,
) -> AnyElement<'static> {
    let text = reply.message.text();
    let reasoning = reply.message.reasoning();
    let stats = reply.stats.clone();

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
            #(reasoning.map(|content| element! {
                View(
                    width: 100pct,
                    background_color: Some(overlay_color),
                ) {
                    Text(content: content, color: subtitle_color)
                }
            }))
            #(text.map(|content| element! {
                Text(content: content)
            }))
            #(chat_reply_stats_component(&stats, subtitle_color))
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
                content: format!("generation speed: {generation_speed}"),
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
