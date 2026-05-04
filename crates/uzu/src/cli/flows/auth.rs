use iocraft::prelude::*;

use crate::{
    cli::{
        components::{ApplicationState, Selector, SelectorItem, SelectorStyle},
        flows::{
            Flow, FlowEvent,
            auth_key::{AuthKeyFlow, AuthProvider},
        },
    },
    engine::config::{
        KEY_ANTHROPIC_API_KEY, KEY_BASETEN_API_KEY, KEY_GEMINI_API_KEY, KEY_HF_TOKEN, KEY_LALAMO_PATH,
        KEY_MIRAI_API_KEY, KEY_OPENAI_API_KEY, KEY_OPENROUTER_API_KEY, KEY_XAI_API_KEY,
    },
};

pub struct AuthFlow;

impl Flow for AuthFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { AuthProviderFlowView(on_event: on_event) }.into()
    }
}

#[derive(Default, Props)]
struct AuthProviderFlowViewProps {
    on_event: Handler<FlowEvent>,
}

#[component]
fn AuthProviderFlowView(
    props: &mut AuthProviderFlowViewProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let state = *hooks.use_context::<State<ApplicationState>>();

    let providers = auth_providers();
    let items: Vec<SelectorItem> = providers
        .iter()
        .map(|provider| SelectorItem {
            title: provider.title.clone(),
            description: None,
            color: None,
        })
        .collect();

    element! {
        Selector(
            items: items,
            style: SelectorStyle::WithIcon,
            maximal_height: (providers.len() as u16).min(5),
            accent_color: state.read().theme.accent_color,
            subtitle_color: state.read().theme.subtitle_color,
            columns_padding: state.read().theme.padding_wide(),
            on_submit: move |index: usize| {
                if let Some(provider) = providers.get(index) {
                    on_event(FlowEvent::transition(provider.title.clone(), Box::new(AuthKeyFlow::new(provider.clone()))));
                }
            },
        )
    }
}

fn auth_providers() -> Vec<AuthProvider> {
    vec![
        AuthProvider::key("Mirai", KEY_MIRAI_API_KEY),
        AuthProvider::key("OpenAI", KEY_OPENAI_API_KEY),
        AuthProvider::key("Anthropic", KEY_ANTHROPIC_API_KEY),
        AuthProvider::key("Google Gemini", KEY_GEMINI_API_KEY),
        AuthProvider::key("xAI", KEY_XAI_API_KEY),
        AuthProvider::key("Baseten", KEY_BASETEN_API_KEY),
        AuthProvider::key("OpenRouter", KEY_OPENROUTER_API_KEY),
        AuthProvider::key("HuggingFace", KEY_HF_TOKEN),
        AuthProvider::path("lalamo", KEY_LALAMO_PATH),
    ]
}
