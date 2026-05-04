use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;

use crate::{
    cli::{
        components::{ApplicationState, InputType, TextInput, TextInputFocus},
        flows::{Flow, FlowEvent},
        helpers::SYMBOL_INPUT_RESULT,
    },
    settings::SettingType,
};

const SAFE_PADDING: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthProvider {
    pub title: String,
    pub key: String,
    pub value_title: String,
    pub input_type: InputType,
    pub setting_type: SettingType,
}

impl AuthProvider {
    pub fn key(
        title: impl Into<String>,
        key: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            key: key.into(),
            value_title: "API_KEY".to_string(),
            input_type: InputType::Secret,
            setting_type: SettingType::Secret,
        }
    }

    pub fn path(
        title: impl Into<String>,
        key: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            key: key.into(),
            value_title: "PATH".to_string(),
            input_type: InputType::Secret,
            setting_type: SettingType::Config,
        }
    }
}

impl Default for AuthProvider {
    fn default() -> Self {
        Self {
            title: String::new(),
            key: String::new(),
            value_title: String::new(),
            input_type: InputType::Text,
            setting_type: SettingType::Config,
        }
    }
}

pub struct AuthKeyFlow {
    provider: AuthProvider,
}

impl AuthKeyFlow {
    pub fn new(provider: AuthProvider) -> Self {
        Self {
            provider,
        }
    }
}

impl Flow for AuthKeyFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { AuthKeyFlowView(provider: self.provider.clone(), on_event: on_event) }.into()
    }
}

#[derive(Default, Props)]
struct AuthKeyFlowViewProps {
    provider: AuthProvider,
    on_event: Handler<FlowEvent>,
}

#[component]
fn AuthKeyFlowView(
    props: &mut AuthKeyFlowViewProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let provider = props.provider.clone();
    let on_event = std::mem::take(&mut props.on_event);
    let state = *hooks.use_context::<State<ApplicationState>>();
    let (width, _) = hooks.use_terminal_size();

    let padding = state.read().theme.padding();
    let subtitle_color = state.read().theme.subtitle_color;
    let label = format!("{} {}", SYMBOL_INPUT_RESULT, provider.value_title);
    let maximal_width = width
        .saturating_sub(UnicodeWidthStr::width(label.as_str()) as u16)
        .saturating_sub(padding)
        .saturating_sub(SAFE_PADDING)
        .max(1);

    element! {
        View(flex_direction: FlexDirection::Row, width: 100pct, column_gap: padding) {
            Text(content: label, color: subtitle_color)
            TextInput(
                maximal_width: maximal_width,
                focus: TextInputFocus::Full,
                r#type: provider.input_type,
                on_change: move |_text: String| {},
                on_submit: move |value: String| {
                    let value = value.trim().to_string();
                    if value.is_empty() {
                        on_event(FlowEvent::finish("Error: value is empty"));
                        return;
                    }

                    let settings = state.read().settings.clone();
                    let result = match settings {
                        Some(settings) => settings
                            .save(provider.setting_type.clone(), &provider.key, Some(value))
                            .map(|()| "Saved, please restart to apply changes".to_string())
                            .unwrap_or_else(|error| format!("Error: {}", error)),
                        None => "Error: settings are not available".to_string(),
                    };
                    on_event(FlowEvent::finish(result));
                },
            )
        }
    }
}
