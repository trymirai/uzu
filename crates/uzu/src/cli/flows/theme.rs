use iocraft::prelude::*;

use crate::cli::{
    components::{ApplicationState, Selector, SelectorItem, SelectorStyle, Theme},
    flows::{Flow, FlowEvent},
};

pub struct ThemeFlow;

impl Flow for ThemeFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { ThemeFlowView(on_event: on_event) }.into()
    }
}

#[derive(Default, Props)]
pub struct ThemeFlowViewProps {
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn ThemeFlowView(
    props: &mut ThemeFlowViewProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let state = *hooks.use_context::<State<ApplicationState>>();

    let themes = Theme::all();
    let items: Vec<SelectorItem> = themes
        .iter()
        .map(|theme| SelectorItem {
            title: theme.name.clone(),
            description: None,
            color: Some(theme.accent_color),
        })
        .collect();

    element! {
        Selector(
            items: items,
            style: SelectorStyle::Plain,
            maximal_height: (themes.len() as u16).min(5),
            accent_color: state.read().theme.accent_color,
            subtitle_color: state.read().theme.subtitle_color,
            columns_padding: state.read().theme.padding_wide(),
            on_submit: move |index: usize| {
                let mut state = state;
                if let Some(theme) = themes.get(index) {
                    let settings_result = match state.read().settings.as_ref() {
                        Some(settings) => theme.save(settings),
                        None => Ok(()),
                    };
                    state.write().theme = theme.clone();
                    let result = match settings_result {
                        Ok(()) => format!("Theme set to {}", theme.name),
                        Err(error) => format!("Theme set to {}, unable to save preference: {}", theme.name, error),
                    };
                    on_event(FlowEvent::finish(result));
                }
            },
        )
    }
}
