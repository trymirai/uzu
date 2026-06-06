use iocraft::prelude::*;

use crate::cli::{
    components::{ApplicationState, Preferences, SamplingMode, Theme},
    flows::{Flow, FlowEvent},
};

const LABEL_WIDTH: u16 = 14;
const TEMPERATURE_STEP: f64 = 0.05;
const TEMPERATURE_MAX: f64 = 2.0;
const PROBABILITY_STEP: f64 = 0.05;
const PROBABILITY_MAX: f64 = 1.0;
const TOP_K_STEP: i64 = 1;
const TOP_K_MIN: i64 = 1;
const TOP_K_MAX: i64 = 4096;

pub struct SettingsFlow;

impl Flow for SettingsFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { SettingsFlowView(on_event: on_event) }.into()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Thinking,
    SamplingMode,
    Temperature,
    TopK,
    TopP,
    MinP,
}

fn visible_fields(preferences: &Preferences) -> Vec<Field> {
    let mut fields = vec![Field::Thinking, Field::SamplingMode];
    if preferences.sampling.mode == SamplingMode::Stochastic {
        fields.extend([Field::Temperature, Field::TopK, Field::TopP, Field::MinP]);
    }
    fields
}

fn adjust(
    preferences: &mut Preferences,
    field: Field,
    delta: i64,
) {
    let sampling = &mut preferences.sampling;
    match field {
        Field::Thinking => {
            preferences.thinking = if delta >= 0 {
                preferences.thinking.next()
            } else {
                preferences.thinking.previous()
            };
        },
        Field::SamplingMode => {
            sampling.mode = if delta >= 0 {
                sampling.mode.next()
            } else {
                sampling.mode.previous()
            };
        },
        Field::Temperature => {
            sampling.temperature = step_f64(sampling.temperature, delta, TEMPERATURE_STEP, 0.0, TEMPERATURE_MAX);
        },
        Field::TopK => {
            sampling.top_k = (sampling.top_k + delta * TOP_K_STEP).clamp(TOP_K_MIN, TOP_K_MAX);
        },
        Field::TopP => {
            sampling.top_p = step_f64(sampling.top_p, delta, PROBABILITY_STEP, 0.0, PROBABILITY_MAX);
        },
        Field::MinP => {
            sampling.min_p = step_f64(sampling.min_p, delta, PROBABILITY_STEP, 0.0, PROBABILITY_MAX);
        },
    }
}

fn toggle(
    preferences: &mut Preferences,
    field: Field,
) {
    let sampling = &mut preferences.sampling;
    match field {
        Field::Thinking => preferences.thinking = preferences.thinking.next(),
        Field::SamplingMode => sampling.mode = sampling.mode.next(),
        Field::Temperature => sampling.temperature_enabled = !sampling.temperature_enabled,
        Field::TopK => sampling.top_k_enabled = !sampling.top_k_enabled,
        Field::TopP => sampling.top_p_enabled = !sampling.top_p_enabled,
        Field::MinP => sampling.min_p_enabled = !sampling.min_p_enabled,
    }
}

fn step_f64(
    value: f64,
    delta: i64,
    step: f64,
    min: f64,
    max: f64,
) -> f64 {
    let next = (value + delta as f64 * step).clamp(min, max);
    (next * 100.0).round() / 100.0
}

#[derive(Default, Props)]
pub struct SettingsFlowViewProps {
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn SettingsFlowView(
    props: &mut SettingsFlowViewProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let state = *hooks.use_context::<State<ApplicationState>>();

    let mut draft = hooks.use_state(|| state.read().preferences);
    let mut selected_index = hooks.use_state(|| 0usize);

    let current = draft.get();
    let fields = visible_fields(&current);
    if selected_index.get() >= fields.len() {
        selected_index.set(fields.len().saturating_sub(1));
    }

    hooks.use_terminal_events(move |event| {
        let TerminalEvent::Key(KeyEvent {
            code,
            kind,
            ..
        }) = event
        else {
            return;
        };
        if kind == KeyEventKind::Release {
            return;
        }

        let preferences = draft.get();
        let fields = visible_fields(&preferences);
        if fields.is_empty() {
            return;
        }
        let current = selected_index.get().min(fields.len() - 1);
        let field = fields[current];

        match code {
            KeyCode::Up => {
                let next = if current == 0 {
                    fields.len() - 1
                } else {
                    current - 1
                };
                selected_index.set(next);
            },
            KeyCode::Down => {
                let next = if current + 1 >= fields.len() {
                    0
                } else {
                    current + 1
                };
                selected_index.set(next);
            },
            KeyCode::Left => {
                let mut preferences = draft.get();
                adjust(&mut preferences, field, -1);
                draft.set(preferences);
            },
            KeyCode::Right => {
                let mut preferences = draft.get();
                adjust(&mut preferences, field, 1);
                draft.set(preferences);
            },
            KeyCode::Char(' ') => {
                let mut preferences = draft.get();
                toggle(&mut preferences, field);
                draft.set(preferences);
            },
            KeyCode::Enter => {
                let mut state = state;
                let preferences = draft.get();
                let summary =
                    format!("thinking {} · sampling {}", preferences.thinking.label(), preferences.sampling.summary());
                let result = match state.read().settings.clone() {
                    Some(settings) => match preferences.save(&settings) {
                        Ok(()) => format!("Settings saved ({})", summary),
                        Err(error) => format!("Settings applied ({}), unable to save: {}", summary, error),
                    },
                    None => format!("Settings applied ({})", summary),
                };
                state.write().preferences = preferences;
                on_event(FlowEvent::finish(result));
            },
            _ => {},
        }
    });

    let theme = state.read().theme.clone();
    let preferences = current;
    let selected = fields.get(selected_index.get().min(fields.len().saturating_sub(1))).copied();
    let padding = theme.padding();

    let mut rows: Vec<AnyElement<'static>> = vec![
        section_header("Thinking", &theme),
        field_row(Field::Thinking, &preferences, selected, &theme),
        section_header("Sampling", &theme),
        field_row(Field::SamplingMode, &preferences, selected, &theme),
    ];
    if preferences.sampling.mode == SamplingMode::Stochastic {
        rows.push(field_row(Field::Temperature, &preferences, selected, &theme));
        rows.push(field_row(Field::TopK, &preferences, selected, &theme));
        rows.push(field_row(Field::TopP, &preferences, selected, &theme));
        rows.push(field_row(Field::MinP, &preferences, selected, &theme));
    }

    element! {
        View(flex_direction: FlexDirection::Column, padding_left: padding, padding_right: padding) {
            #(rows.into_iter())
            View(height: padding)
            Text(
                content: "↑↓ move · ←→ adjust · space toggle · enter save · esc cancel",
                color: theme.subtitle_color,
            )
        }
    }
}

fn section_header(
    title: &str,
    theme: &Theme,
) -> AnyElement<'static> {
    element! {
        View(margin_top: 1) {
            Text(content: title.to_string(), color: Some(theme.accent_color), weight: Weight::Bold)
        }
    }
    .into()
}

fn field_row(
    field: Field,
    preferences: &Preferences,
    selected: Option<Field>,
    theme: &Theme,
) -> AnyElement<'static> {
    let is_selected = selected == Some(field);
    let marker = if is_selected {
        "›"
    } else {
        " "
    };
    let value_color = if is_selected {
        Some(theme.accent_color)
    } else {
        None
    };

    let (label, control): (&str, AnyElement<'static>) = match field {
        Field::Thinking => ("Thinking", cycle_control(preferences.thinking.label(), value_color, theme)),
        Field::SamplingMode => ("Sampling", cycle_control(preferences.sampling.mode.label(), value_color, theme)),
        Field::Temperature => (
            "Temperature",
            toggle_control(
                preferences.sampling.temperature_enabled,
                format!("{:.2}", preferences.sampling.temperature),
                is_selected,
                theme,
            ),
        ),
        Field::TopK => (
            "Top K",
            toggle_control(
                preferences.sampling.top_k_enabled,
                preferences.sampling.top_k.to_string(),
                is_selected,
                theme,
            ),
        ),
        Field::TopP => (
            "Top P",
            toggle_control(
                preferences.sampling.top_p_enabled,
                format!("{:.2}", preferences.sampling.top_p),
                is_selected,
                theme,
            ),
        ),
        Field::MinP => (
            "Min P",
            toggle_control(
                preferences.sampling.min_p_enabled,
                format!("{:.2}", preferences.sampling.min_p),
                is_selected,
                theme,
            ),
        ),
    };

    element! {
        View(flex_direction: FlexDirection::Row, column_gap: 1u16) {
            Text(content: marker, color: Some(theme.accent_color))
            View(width: LABEL_WIDTH) {
                Text(
                    content: label.to_string(),
                    color: if is_selected { Some(theme.accent_color) } else { None },
                )
            }
            #(control)
        }
    }
    .into()
}

fn cycle_control(
    value: &str,
    value_color: Option<Color>,
    theme: &Theme,
) -> AnyElement<'static> {
    element! {
        View(flex_direction: FlexDirection::Row) {
            Text(content: "‹ ", color: Some(theme.subtitle_color))
            Text(content: value.to_string(), color: value_color, weight: Weight::Bold)
            Text(content: " ›", color: Some(theme.subtitle_color))
        }
    }
    .into()
}

fn toggle_control(
    enabled: bool,
    value: String,
    is_selected: bool,
    theme: &Theme,
) -> AnyElement<'static> {
    let checkbox = if enabled {
        "[x]"
    } else {
        "[ ]"
    };
    let checkbox_color = if enabled {
        Some(theme.accent_color)
    } else {
        Some(theme.subtitle_color)
    };
    let value_color = if is_selected {
        Some(theme.accent_color)
    } else if enabled {
        None
    } else {
        Some(theme.subtitle_color)
    };

    element! {
        View(flex_direction: FlexDirection::Row, column_gap: 1u16) {
            Text(content: checkbox, color: checkbox_color)
            Text(content: value, color: value_color, weight: Weight::Bold)
        }
    }
    .into()
}
