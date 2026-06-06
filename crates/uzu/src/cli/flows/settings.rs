use iocraft::prelude::*;

use crate::cli::{
    components::{ApplicationState, ModelSamplingDefaults, Preferences, SamplingMode, Theme, ThinkingSupport},
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

fn visible_fields(
    preferences: &Preferences,
    support: ThinkingSupport,
) -> Vec<Field> {
    let mut fields = Vec::new();
    if support.is_adjustable() {
        fields.push(Field::Thinking);
    }
    fields.push(Field::SamplingMode);
    if preferences.sampling.mode == SamplingMode::Stochastic {
        fields.extend([Field::Temperature, Field::TopK, Field::TopP, Field::MinP]);
    }
    fields
}

fn adjust(
    preferences: &mut Preferences,
    field: Field,
    delta: i64,
    support: ThinkingSupport,
) {
    match field {
        Field::Thinking => {
            support.with_preference(&preferences.thinking).cycled(delta).write_back(&mut preferences.thinking);
        },
        Field::SamplingMode => {
            preferences.sampling.mode = if delta >= 0 {
                preferences.sampling.mode.next()
            } else {
                preferences.sampling.mode.previous()
            };
        },
        Field::Temperature => {
            preferences.sampling.temperature =
                step_f64(preferences.sampling.temperature, delta, TEMPERATURE_STEP, 0.0, TEMPERATURE_MAX);
        },
        Field::TopK => {
            preferences.sampling.top_k = (preferences.sampling.top_k + delta * TOP_K_STEP).clamp(TOP_K_MIN, TOP_K_MAX);
        },
        Field::TopP => {
            preferences.sampling.top_p =
                step_f64(preferences.sampling.top_p, delta, PROBABILITY_STEP, 0.0, PROBABILITY_MAX);
        },
        Field::MinP => {
            preferences.sampling.min_p =
                step_f64(preferences.sampling.min_p, delta, PROBABILITY_STEP, 0.0, PROBABILITY_MAX);
        },
    }
}

fn toggle(
    preferences: &mut Preferences,
    field: Field,
    support: ThinkingSupport,
) {
    match field {
        Field::Thinking => adjust(preferences, Field::Thinking, 1, support),
        Field::SamplingMode => preferences.sampling.mode = preferences.sampling.mode.next(),
        Field::Temperature => preferences.sampling.temperature_enabled = !preferences.sampling.temperature_enabled,
        Field::TopK => preferences.sampling.top_k_enabled = !preferences.sampling.top_k_enabled,
        Field::TopP => preferences.sampling.top_p_enabled = !preferences.sampling.top_p_enabled,
        Field::MinP => preferences.sampling.min_p_enabled = !preferences.sampling.min_p_enabled,
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

fn float_value(
    enabled: bool,
    value: f64,
    default: Option<f64>,
) -> String {
    if enabled {
        format!("{value:.2}")
    } else {
        default.map(|value| format!("{value:.2}")).unwrap_or_else(|| "—".to_string())
    }
}

fn int_value(
    enabled: bool,
    value: i64,
    default: Option<i64>,
) -> String {
    if enabled {
        value.to_string()
    } else {
        default.map(|value| value.to_string()).unwrap_or_else(|| "—".to_string())
    }
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

    let capabilities =
        state.read().model_state.as_ref().map(|model_state| model_state.capabilities).unwrap_or_default();
    let support = capabilities.thinking;
    let defaults = capabilities.sampling_defaults;

    let mut draft = hooks.use_state(|| state.read().preferences);
    let mut selected_index = hooks.use_state(|| 0usize);

    let preferences = draft.get();
    let fields = visible_fields(&preferences, support);
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
        let fields = visible_fields(&preferences, support);
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
                adjust(&mut preferences, field, -1, support);
                draft.set(preferences);
            },
            KeyCode::Right => {
                let mut preferences = draft.get();
                adjust(&mut preferences, field, 1, support);
                draft.set(preferences);
            },
            KeyCode::Char(' ') => {
                let mut preferences = draft.get();
                toggle(&mut preferences, field, support);
                draft.set(preferences);
            },
            KeyCode::Enter => {
                let mut state = state;
                let preferences = draft.get();
                let summary = format!(
                    "thinking {} · sampling {}",
                    thinking_summary(support, &preferences),
                    preferences.sampling.summary()
                );
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
    let selected = fields.get(selected_index.get().min(fields.len().saturating_sub(1))).copied();
    let padding = theme.padding();

    let mut rows: Vec<AnyElement<'static>> = vec![section_header("Thinking", &theme)];
    if support.is_adjustable() {
        rows.push(field_row(Field::Thinking, &preferences, selected, &theme, support, defaults));
    } else {
        rows.push(info_row("Thinking", support.value_label(), &theme));
    }

    rows.push(section_header("Sampling", &theme));
    rows.push(field_row(Field::SamplingMode, &preferences, selected, &theme, support, defaults));
    match preferences.sampling.mode {
        SamplingMode::ModelDefault => rows.push(info_row("", &format!("uses {}", defaults.summary()), &theme)),
        SamplingMode::Greedy => {},
        SamplingMode::Stochastic => {
            rows.push(field_row(Field::Temperature, &preferences, selected, &theme, support, defaults));
            rows.push(field_row(Field::TopK, &preferences, selected, &theme, support, defaults));
            rows.push(field_row(Field::TopP, &preferences, selected, &theme, support, defaults));
            rows.push(field_row(Field::MinP, &preferences, selected, &theme, support, defaults));
        },
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

fn thinking_summary(
    support: ThinkingSupport,
    preferences: &Preferences,
) -> &'static str {
    support.with_preference(&preferences.thinking).value_label()
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

fn info_row(
    label: &str,
    value: &str,
    theme: &Theme,
) -> AnyElement<'static> {
    element! {
        View(flex_direction: FlexDirection::Row, column_gap: 1u16) {
            Text(content: " ", color: Some(theme.subtitle_color))
            View(width: LABEL_WIDTH) {
                Text(content: label.to_string(), color: Some(theme.subtitle_color))
            }
            Text(content: value.to_string(), color: Some(theme.subtitle_color))
        }
    }
    .into()
}

fn field_row(
    field: Field,
    preferences: &Preferences,
    selected: Option<Field>,
    theme: &Theme,
    support: ThinkingSupport,
    defaults: ModelSamplingDefaults,
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
        Field::Thinking => (
            "Thinking",
            cycle_control(support.with_preference(&preferences.thinking).value_label(), value_color, theme),
        ),
        Field::SamplingMode => ("Sampling", cycle_control(preferences.sampling.mode.label(), value_color, theme)),
        Field::Temperature => (
            "Temperature",
            toggle_control(
                preferences.sampling.temperature_enabled,
                float_value(
                    preferences.sampling.temperature_enabled,
                    preferences.sampling.temperature,
                    defaults.temperature,
                ),
                is_selected,
                theme,
            ),
        ),
        Field::TopK => (
            "Top K",
            toggle_control(
                preferences.sampling.top_k_enabled,
                int_value(preferences.sampling.top_k_enabled, preferences.sampling.top_k, defaults.top_k),
                is_selected,
                theme,
            ),
        ),
        Field::TopP => (
            "Top P",
            toggle_control(
                preferences.sampling.top_p_enabled,
                float_value(preferences.sampling.top_p_enabled, preferences.sampling.top_p, defaults.top_p),
                is_selected,
                theme,
            ),
        ),
        Field::MinP => (
            "Min P",
            toggle_control(
                preferences.sampling.min_p_enabled,
                float_value(preferences.sampling.min_p_enabled, preferences.sampling.min_p, defaults.min_p),
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
    let hint: Option<AnyElement<'static>> =
        (!enabled).then(|| element! { Text(content: "(default)", color: Some(theme.subtitle_color)) }.into());

    element! {
        View(flex_direction: FlexDirection::Row, column_gap: 1u16) {
            Text(content: checkbox, color: checkbox_color)
            Text(content: value, color: value_color, weight: Weight::Bold)
            #(hint.into_iter())
        }
    }
    .into()
}
