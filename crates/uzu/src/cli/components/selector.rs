use iocraft::prelude::*;

use crate::cli::{components::Gradient, helpers::ColorRgb};

#[derive(Clone)]
pub struct SelectorItem {
    pub title: String,
    pub description: Option<String>,
    pub color: Option<Color>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SelectorStyle {
    #[default]
    Plain,
    WithIcon,
}

#[derive(Default, Props)]
pub struct SelectorProps {
    pub items: Vec<SelectorItem>,
    pub style: SelectorStyle,
    pub maximal_height: u16,
    pub accent_color: Option<Color>,
    pub on_submit: HandlerMut<'static, usize>,
}

#[component]
pub fn Selector(
    props: &mut SelectorProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let items = props.items.clone();
    let style = props.style;
    let maximal_height = (props.maximal_height as usize).max(1);
    let accent_color = props.accent_color;
    let mut on_submit = props.on_submit.take();

    let mut selected_index = hooks.use_state(|| 0usize);
    let scroll_handle = hooks.use_ref_default::<ScrollViewHandle>();

    let items_count = items.len();
    if items_count > 0 && selected_index.get() >= items_count {
        selected_index.set(items_count - 1);
    }

    hooks.use_terminal_events({
        let mut scroll_handle = scroll_handle;

        move |event| {
            let TerminalEvent::Key(KeyEvent {
                code,
                kind,
                ..
            }) = event
            else {
                return;
            };
            if kind == KeyEventKind::Release || items_count == 0 {
                return;
            }
            let next = match code {
                KeyCode::Up => Some(selected_index.get().saturating_sub(1)),
                KeyCode::Down => Some((selected_index.get() + 1).min(items_count - 1)),
                KeyCode::Enter => {
                    on_submit(selected_index.get());
                    None
                },
                _ => None,
            };

            let Some(next) = next else {
                return;
            };
            if next == selected_index.get() {
                return;
            }
            selected_index.set(next);

            let row = next as i32;
            let scroll_offset = scroll_handle.read().scroll_offset();
            let viewport_height = scroll_handle.read().viewport_height() as i32;
            if row < scroll_offset {
                scroll_handle.write().scroll_to(row);
            } else if viewport_height > 0 && row >= scroll_offset + viewport_height {
                scroll_handle.write().scroll_to(row - viewport_height + 1);
            }
        }
    });

    let selected = selected_index.get();
    let rows: Vec<AnyElement<'static>> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| item_component(item, index == selected, style, accent_color))
        .collect();

    element! {
        View(height: maximal_height as u32) {
            ScrollView(
                handle: Some(scroll_handle),
                keyboard_scroll: Some(false),
                scrollbar: Some(false),
            ) {
                View(flex_direction: FlexDirection::Column, width: 100pct) {
                    #(rows.into_iter())
                }
            }
        }
    }
}

fn item_component(
    item: SelectorItem,
    is_selected: bool,
    style: SelectorStyle,
    accent_color: Option<Color>,
) -> AnyElement<'static> {
    let icon = match style {
        SelectorStyle::WithIcon => Some(if is_selected {
            "[x]"
        } else {
            "[ ]"
        }),
        SelectorStyle::Plain => None,
    };
    let label = match (icon, item.description.as_deref()) {
        (Some(icon), Some(description)) => format!("{} {} {}", icon, item.title, description),
        (Some(icon), None) => format!("{} {}", icon, item.title),
        (None, Some(description)) => format!("{} {}", item.title, description),
        (None, None) => item.title.clone(),
    };

    if is_selected {
        if let Some(color) = item.color {
            return element! {
                Gradient(from_color: Some(color.darker(0.25)), to_color: Some(color), width: 100pct) {
                    Text(content: label, weight: Weight::Bold)
                }
            }
            .into();
        }
        return element! {
            Text(content: label, weight: Weight::Bold, color: accent_color)
        }
        .into();
    }

    element! {
        Text(content: label, color: item.color)
    }
    .into()
}
